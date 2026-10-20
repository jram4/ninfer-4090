"""Shadow-first, compare-and-swap promotion of a qualified local commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

from .contracts import canonical_hash, read_json, validate_policy, validate_receipt, sha256_file
from .submit import SubmissionError, _manifest_from_commit


_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
MAIN_REF = "refs/heads/main"


class PromotionError(RuntimeError):
    """Promotion is unsafe, stale, or not backed by the required evidence."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        raise PromotionError(
            f"git {' '.join(args)} failed: {detail.decode('utf-8', errors='replace').strip()}"
        ) from exc
    return result


def _resolve(repo: Path, revision: str, label: str) -> str:
    if not isinstance(revision, str) or not _OBJECT_ID.fullmatch(revision):
        raise PromotionError(f"{label} must be a full lowercase Git object ID")
    result = _git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    resolved = result.stdout.decode("ascii").strip()
    if resolved != revision:
        raise PromotionError(f"{label} does not resolve to the supplied commit")
    return resolved


def _is_bare(repo: Path) -> bool:
    return _git(repo, "rev-parse", "--is-bare-repository").stdout.strip() == b"true"


def _repo_boundary(repo: Path) -> Path:
    top = _git(repo, "rev-parse", "--show-toplevel", check=False)
    if top.returncode == 0:
        return Path(top.stdout.decode("utf-8").strip()).resolve()
    _git(repo, "rev-parse", "--git-dir")
    return repo.resolve()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_symlink_component(path: Path, *, stop: Path | None = None) -> bool:
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor /= component
        if stop is not None and cursor == stop:
            continue
        if cursor.is_symlink():
            return True
    return False


def _trusted_promotion_reasons(
    repository_boundary: Path,
    receipt: dict[str, Any],
    *,
    receipt_path: Path | None,
    trusted_receipt_root: Path | None,
    trusted_policy_path: Path | None,
    authorize: bool,
) -> list[str]:
    reasons: list[str] = []
    if not authorize:
        reasons.append("actual promotion requires explicit caller authorization (authorize=True)")
    if trusted_receipt_root is None:
        reasons.append("actual promotion requires trusted_receipt_root")
    if trusted_policy_path is None:
        reasons.append("actual promotion requires trusted_policy_path")
    if receipt_path is None:
        reasons.append("actual promotion requires receipt_path inside the trusted judge root")
    if trusted_receipt_root is None or trusted_policy_path is None or receipt_path is None:
        return reasons

    normalized_receipt: dict[str, Any] | None = None
    try:
        normalized_receipt = validate_receipt(receipt)
        root = Path(trusted_receipt_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("trusted_receipt_root is not a directory")
        if _under(root, repository_boundary):
            raise ValueError("trusted_receipt_root must be outside the candidate repository")
        raw_receipt_path = Path(receipt_path).expanduser()
        if ".." in raw_receipt_path.parts:
            raise ValueError("receipt_path must not contain parent traversal")
        lexical_receipt = (root / raw_receipt_path) if not raw_receipt_path.is_absolute() else raw_receipt_path
        if _has_symlink_component(lexical_receipt.absolute(), stop=root):
            raise ValueError("receipt_path must not traverse symlinks")
        resolved_receipt = lexical_receipt.resolve(strict=True)
        if not _under(resolved_receipt, root):
            raise ValueError("receipt_path is outside trusted_receipt_root")
        receipt_rel = resolved_receipt.relative_to(root)
        if not resolved_receipt.is_file():
            raise ValueError("receipt_path must resolve to a regular non-symlink file")
        disk_receipt = validate_receipt(read_json(resolved_receipt))
        if canonical_hash(disk_receipt) != canonical_hash(normalized_receipt):
            raise ValueError("supplied receipt differs from the trusted receipt file")
        # Keep the path binding explicit even when a caller supplied an absolute
        # path; relative paths are rooted at the operator-controlled judge store.
        if not receipt_rel.parts:
            raise ValueError("receipt_path must name a file below trusted_receipt_root")
        for reference in normalized_receipt["evidence"]:
            evidence_path = Path(reference["path"])
            if (
                evidence_path.is_absolute()
                or not evidence_path.parts
                or ".." in evidence_path.parts
                or "\\" in reference["path"]
                or evidence_path.as_posix() != reference["path"]
            ):
                raise ValueError("evidence paths must be safe relative paths under trusted_receipt_root")
            lexical_evidence = root / evidence_path
            if _has_symlink_component(lexical_evidence, stop=root):
                raise ValueError("evidence reference traverses a symlink in trusted_receipt_root")
            resolved_evidence = lexical_evidence.resolve(strict=True)
            if not _under(resolved_evidence, root):
                raise ValueError("evidence reference escapes or traverses a symlink in trusted_receipt_root")
            metadata = resolved_evidence.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("evidence reference is not a regular file")
            if sha256_file(resolved_evidence) != reference["sha256"]:
                raise ValueError(f"evidence hash mismatch: {reference['path']}")
    except (OSError, ValueError) as exc:
        reasons.append(f"trusted receipt/evidence verification failed: {exc}")

    try:
        if normalized_receipt is None:
            raise ValueError("receipt is not valid under the shared receipt contract")
        raw_policy_path = Path(trusted_policy_path).expanduser()
        if _has_symlink_component(raw_policy_path.absolute()):
            raise ValueError("trusted policy path must not traverse symlinks")
        policy_path = raw_policy_path.resolve(strict=True)
        if _under(policy_path, repository_boundary):
            raise ValueError("trusted policy must be a non-symlink file outside the candidate repository")
        if not policy_path.is_file():
            raise ValueError("trusted_policy_path is not a file")
        policy = validate_policy(read_json(policy_path))
        if policy["policy_id"] != normalized_receipt["policy_id"]:
            raise ValueError("trusted policy_id does not match receipt")
        if policy["contract_id"] != normalized_receipt["contract_id"]:
            raise ValueError("trusted policy contract_id does not match receipt")
        if policy["qualification_enabled"] is not True:
            raise ValueError("trusted policy does not explicitly enable qualification")
    except (OSError, ValueError) as exc:
        reasons.append(f"trusted policy verification failed: {exc}")
    return reasons


def _receipt_reasons(
    repo: Path,
    candidate_commit: str,
    expected_main: str,
    receipt: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    try:
        normalized = validate_receipt(receipt)
    except ValueError as exc:
        return [f"invalid receipt: {exc}"]
    if normalized["verdict"] != "qualified":
        reasons.append("receipt verdict is not qualified")
    if normalized["stage"] != "full":
        reasons.append("receipt did not complete the full stage")
    if normalized["failure_class"] is not None:
        reasons.append("receipt has a failure_class")
    if normalized.get("candidate_commit") != candidate_commit:
        reasons.append("receipt candidate_commit does not bind the candidate")
    if normalized.get("base_commit") != expected_main:
        reasons.append("receipt base_commit does not bind expected_main")
    if not normalized["evidence"]:
        reasons.append("receipt has no evidence references")
    try:
        manifest, _ = _manifest_from_commit(repo, candidate_commit)
        actual_snapshot = canonical_hash(manifest)
        if normalized["source_snapshot_sha256"] != actual_snapshot:
            reasons.append("receipt source_snapshot_sha256 does not match candidate source")
    except (PromotionError, SubmissionError, OSError, ValueError) as exc:
        reasons.append(f"candidate source cannot be verified: {exc}")
    return reasons


def _plan(
    repo: Path,
    candidate_commit: str,
    expected_main: str,
    receipt: dict[str, Any],
    *,
    shadow: bool,
    receipt_path: Path | None,
    trusted_receipt_root: Path | None,
    trusted_policy_path: Path | None,
    authorize: bool,
) -> dict[str, Any]:
    repository = Path(repo).expanduser().resolve(strict=True)
    top = _git(repository, "rev-parse", "--show-toplevel", check=False)
    if top.returncode != 0:
        # Bare repositories do not have a worktree top-level.
        _git(repository, "rev-parse", "--git-dir")
    candidate = _resolve(repository, candidate_commit, "candidate_commit")
    expected = _resolve(repository, expected_main, "expected_main")
    current_result = _git(repository, "rev-parse", "--verify", MAIN_REF, check=False)
    current = current_result.stdout.decode("ascii").strip() if current_result.returncode == 0 else None
    reasons: list[str] = []
    if current is None:
        reasons.append("local main ref does not exist")
    elif current != expected:
        reasons.append("stale compare-and-swap base: current main differs from expected_main")
    if candidate == expected:
        reasons.append("candidate commit is not ahead of expected_main")
    ancestor = _git(repository, "merge-base", "--is-ancestor", expected, candidate, check=False)
    if ancestor.returncode != 0:
        reasons.append("candidate is unrelated to expected_main or does not descend from it")
    reasons.extend(_receipt_reasons(repository, candidate, expected, receipt))
    reasons.extend(_trusted_promotion_reasons(
        _repo_boundary(repository),
        receipt,
        receipt_path=receipt_path,
        trusted_receipt_root=trusted_receipt_root,
        trusted_policy_path=trusted_policy_path,
        authorize=authorize,
    ))
    bare = _is_bare(repository)
    if not shadow and not bare:
        reasons.append("actual promotion requires a dedicated bare target; checked-out refs are never advanced")
    return {
        "schema_version": 1,
        "shadow": shadow,
        "main_ref": MAIN_REF,
        "candidate_commit": candidate,
        "expected_main": expected,
        "current_main": current,
        "ready": not reasons,
        "reasons": reasons,
        "mutation": "none",
    }


def plan_promotion(
    repo: Path,
    candidate_commit: str,
    expected_main: str,
    receipt: dict[str, Any],
    *,
    shadow: bool = True,
    receipt_path: Path | None = None,
    trusted_receipt_root: Path | None = None,
    trusted_policy_path: Path | None = None,
    authorize: bool = False,
) -> dict[str, Any]:
    """Return a side-effect-free plan; shadow mode is the default."""
    if type(shadow) is not bool:
        raise ValueError("shadow must be boolean")
    if type(authorize) is not bool:
        raise ValueError("authorize must be boolean")
    return _plan(
        repo, candidate_commit, expected_main, receipt, shadow=shadow,
        receipt_path=receipt_path, trusted_receipt_root=trusted_receipt_root,
        trusted_policy_path=trusted_policy_path, authorize=authorize,
    )


def promote(
    repo: Path,
    candidate_commit: str,
    expected_main: str,
    receipt: dict[str, Any],
    *,
    shadow: bool = True,
    receipt_path: Path | None = None,
    trusted_receipt_root: Path | None = None,
    trusted_policy_path: Path | None = None,
    authorize: bool = False,
) -> dict[str, Any]:
    """Plan by default, or CAS-update main in an explicitly selected bare repo."""
    if type(shadow) is not bool:
        raise ValueError("shadow must be boolean")
    if type(authorize) is not bool:
        raise ValueError("authorize must be boolean")
    plan = _plan(
        repo, candidate_commit, expected_main, receipt, shadow=shadow,
        receipt_path=receipt_path, trusted_receipt_root=trusted_receipt_root,
        trusted_policy_path=trusted_policy_path, authorize=authorize,
    )
    if shadow:
        return plan
    if not plan["ready"]:
        raise PromotionError("promotion denied: " + "; ".join(plan["reasons"]))
    repository = Path(repo).expanduser().resolve(strict=True)
    try:
        _git(repository, "update-ref", MAIN_REF, plan["candidate_commit"], plan["expected_main"])
    except PromotionError as exc:
        raise PromotionError("promotion compare-and-swap failed; main changed after planning") from exc
    plan["mutation"] = "updated-main-ref"
    plan["current_main"] = plan["candidate_commit"]
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--expected-main", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--trusted-receipt-root", type=Path)
    parser.add_argument("--trusted-policy", type=Path)
    parser.add_argument("--promote", action="store_true", help="CAS-update main in a bare target (default: shadow plan)")
    parser.add_argument("--authorize", action="store_true", help="explicit caller authorization required with --promote")
    args = parser.parse_args(argv)
    try:
        result = promote(
            args.repo,
            args.candidate_commit,
            args.expected_main,
            read_json(args.receipt),
            shadow=not args.promote,
            receipt_path=args.receipt,
            trusted_receipt_root=args.trusted_receipt_root,
            trusted_policy_path=args.trusted_policy,
            authorize=args.authorize,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, PromotionError) as exc:
        print(f"promotion.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
