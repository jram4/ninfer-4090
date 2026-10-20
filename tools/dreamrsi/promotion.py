"""Shadow-first, compare-and-swap promotion of a qualified local commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from .contracts import canonical_hash, read_json, validate_receipt
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
    if normalized.get("qualification_enabled") is not True:
        reasons.append("receipt does not prove qualification was explicitly enabled")
    if normalized.get("promotion_authorized") is not True:
        reasons.append("receipt does not include explicit promotion authorization")
    if not normalized["evidence"]:
        reasons.append("receipt has no immutable evidence references")
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
) -> dict[str, Any]:
    """Return a side-effect-free plan; shadow mode is the default."""
    if type(shadow) is not bool:
        raise ValueError("shadow must be boolean")
    return _plan(repo, candidate_commit, expected_main, receipt, shadow=shadow)


def promote(
    repo: Path,
    candidate_commit: str,
    expected_main: str,
    receipt: dict[str, Any],
    *,
    shadow: bool = True,
) -> dict[str, Any]:
    """Plan by default, or CAS-update main in an explicitly selected bare repo."""
    if type(shadow) is not bool:
        raise ValueError("shadow must be boolean")
    plan = _plan(repo, candidate_commit, expected_main, receipt, shadow=shadow)
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
    parser.add_argument("--promote", action="store_true", help="CAS-update main in a bare target (default: shadow plan)")
    args = parser.parse_args(argv)
    try:
        result = promote(
            args.repo,
            args.candidate_commit,
            args.expected_main,
            read_json(args.receipt),
            shadow=not args.promote,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, PromotionError) as exc:
        print(f"promotion.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
