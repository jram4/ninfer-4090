#!/usr/bin/env python3
"""Evaluate one frozen DreamRSI submission under a trusted staged policy.

Policy stages are argv arrays owned outside candidate source, never shell text
or submission-provided commands.  Each stage writes one JSON report at the
trusted ``evidence.path`` and echoes the evaluator-provided attempt, stage,
submission, policy, contract, and source-snapshot identities.  The policy's
``evidence`` object must contain ``path``, ``required_fields``, and an
``equals`` mapping whose ``status`` is ``"pass"``; optional ``minimum`` and
``maximum`` maps use dotted JSON paths. Each stage report also references at
least one raw artifact by run-relative path and SHA-256 in ``raw_artifacts``;
the evaluator independently checks and snapshots those bytes. Stage ``gates``
can name semantic, task, and holdout gates and require non-status evidence
criteria. Exactly one stage must set ``observes_contract`` and report the
actual runtime contract as ``observed_contract``; it is compared to the policy
contract with the shared contracts API.

Policies and receipts use the exact shared contracts schema. The policy
contract locks model_artifact, runtime, template, and tokenizer; the runtime
lock must specify k8v4 and context/KV capacity 188416. Each stage uses the
shared ``name/argv/timeout_seconds/required_evidence/criteria`` shape. Criteria
may specify equality/numeric gates, semantic/task/holdout gates, contract
observation, GPU admission, and CMake/CUDA toolchain locks. Stage reports bind
their attempt identities and reference each required raw artifact by run-
relative path and SHA-256. Those bytes are independently re-hashed and
snapshotted; receipts contain only immutable path/SHA references, while full
diagnostics stay in an evaluator-owned JSON artifact.

Qualification additionally requires all six ordered stages (admission, build,
correctness, smoke, screen, full), distinct semantic/task/holdout evidence,
nonempty corpus and toolchain identities, at least one exclusively admitted
GPU stage, and every configured evidence criterion.  Qualification is disabled
unless the trusted policy explicitly enables it.  Live-endpoint stages are
labelled and can never establish candidate performance qualification.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
from typing import Any, Callable, Mapping
import uuid

try:
    from . import lifecycle
except ImportError:  # direct module execution
    import lifecycle  # type: ignore[no-redef]


SCHEMA_VERSION = 1
STAGE_ORDER = ("admission", "build", "correctness", "smoke", "screen", "full")
QUALIFICATION_GATES = frozenset({"semantic", "task", "holdout"})
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_ARCHIVE_MEMBERS = 500_000
_MAX_UNPACKED_BYTES = 16 * 1024**3
_HASH_CHUNK_BYTES = 1024 * 1024

Cancelled = Callable[[], bool]


class EvaluationError(RuntimeError):
    """A trusted policy or evaluation setup problem."""


class _SourceInputError(EvaluationError):
    """An invalid, changed, or unsafe frozen source input."""


class _EvidenceError(EvaluationError):
    def __init__(self, message: str, *, inconclusive: bool = False,
                 failure_class: str | None = None) -> None:
        self.inconclusive = inconclusive
        self.failure_class = failure_class
        super().__init__(message)


class _PipelineFault(Exception):
    def __init__(
        self,
        verdict: str,
        stage: str,
        failure_class: str | None,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.verdict = verdict
        self.stage = stage
        self.failure_class = failure_class
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


def _contracts_api() -> Any:
    try:
        from . import contracts
    except ImportError:  # direct module execution
        try:
            import contracts  # type: ignore[no-redef]
        except ImportError as exc:
            raise EvaluationError("shared tools.dreamrsi.contracts module is unavailable") from exc
        return contracts
    return contracts


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{field} must be a nonempty string")
    return value


def _validate_locked_contract(contract: Mapping[str, Any], *, require_runtime_identity: bool = True) -> None:
    model = contract.get("model_artifact")
    template = contract.get("template")
    tokenizer = contract.get("tokenizer")
    runtime = contract.get("runtime")
    if not all(isinstance(item, Mapping) for item in (model, template, tokenizer, runtime)):
        raise EvaluationError("qualification contract locks must be structured identity objects")
    for name, lock in (("model_artifact", model), ("template", template)):
        _sha256_text(lock.get("sha256"), f"contract.{name}.sha256")
    if type(model.get("size_bytes")) is not int or model["size_bytes"] <= 0:
        raise EvaluationError("contract.model_artifact.size_bytes must be positive")
    tokenizer_hash = tokenizer.get("sha256", tokenizer.get("embedded_in_model_artifact_sha256"))
    if not isinstance(tokenizer_hash, str) or not _HEX_SHA256.fullmatch(tokenizer_hash):
        raise EvaluationError("contract.tokenizer must have an independently locked SHA-256")
    model_hash = model.get("sha256")
    if tokenizer.get("embedded_in_model_artifact_sha256") is not None and tokenizer_hash != model_hash:
        raise EvaluationError("embedded tokenizer lock differs from model_artifact.sha256")
    if require_runtime_identity:
        # The observed runtime must identify the exact binary it loaded and the
        # source it was built from. The expected contract is the invariant one
        # and deliberately carries no candidate identity.
        _sha256_text(runtime.get("binary", {}).get("sha256")
                     if isinstance(runtime.get("binary"), Mapping) else None,
                     "contract.runtime.binary.sha256")
        _identity_text(runtime.get("release_source_commit"), "contract.runtime.release_source_commit")
    if runtime.get("kv_dtype") != "k8v4":
        raise EvaluationError("contract.runtime.kv_dtype must be k8v4")
    if runtime.get("max_context") != 188416 or runtime.get("kv_capacity") != 188416:
        raise EvaluationError("contract.runtime max_context and kv_capacity must both be 188416")
    sampling = runtime.get("sampling_capabilities")
    if not isinstance(sampling, Mapping) or not sampling:
        raise EvaluationError("contract.runtime.sampling_capabilities must be locked")
    if any(type(value) is not bool for value in sampling.values()):
        raise EvaluationError("contract.runtime.sampling_capabilities values must be booleans")


def _compatibility_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Keep runtime compatibility locks; candidate binary/source are checked separately."""
    result = json.loads(json.dumps(contract))
    runtime = result.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("binary", None)
        runtime.pop("release_source_commit", None)
    return result


def _candidate_binary_identity(core: Any, stage: Mapping[str, Any], report: Mapping[str, Any],
                               paths: Mapping[str, Path | str]) -> dict[str, Any]:
    """Return the independently hashed identity of the built candidate binary."""
    references = report.get("evidence")
    if not isinstance(references, Mapping) or "candidate_binary" not in references:
        raise _PipelineFault("inconclusive", stage["name"], None,
                             "build stage did not report a candidate binary")
    reference = references["candidate_binary"]
    if not isinstance(reference, Mapping):
        raise _PipelineFault("inconclusive", stage["name"], None,
                             "build stage reported an invalid candidate binary reference")
    attempt_dir = paths["attempt_dir"]
    assert isinstance(attempt_dir, Path)
    try:
        source = _safe_relative_file(attempt_dir, str(reference.get("path")))
        info = source.lstat()
    except (OSError, ValueError, _EvidenceError) as exc:
        raise _PipelineFault("inconclusive", stage["name"], None,
                             f"candidate binary could not be identified: {exc}") from exc
    return {
        "sha256": core.sha256_file(source).lower(),
        "size_bytes": info.st_size,
    }


def _verify_observed_contract(core: Any, expected: Mapping[str, Any], observed: Mapping[str, Any],
                              candidate_binary: Mapping[str, Any] | None,
                              source_commit: str, stage_name: str) -> dict[str, Any]:
    # A deviation from the locked model, compression, capability, or sampling
    # contract is a candidate regression, never an infrastructure fault.
    mismatches = core.compare_contract(
        _compatibility_contract(expected), _compatibility_contract(observed)
    )
    if not isinstance(mismatches, list):
        raise EvaluationError("shared contract comparison returned an invalid result")
    if mismatches:
        raise _PipelineFault("regression", stage_name, "candidate",
                             "observed model/runtime compatibility locks differ: "
                             + ", ".join(str(path) for path in mismatches),
                             details={"compatibility_matched": False, "mismatches": mismatches})
    try:
        _validate_locked_contract(observed)
    except EvaluationError as exc:
        raise _PipelineFault("regression", stage_name, "candidate",
                             f"observed runtime violates the locked contract: {exc}") from exc
    runtime = observed.get("runtime")
    binary = runtime.get("binary") if isinstance(runtime, Mapping) else None
    if not isinstance(binary, Mapping) or candidate_binary is None:
        raise _PipelineFault("inconclusive", stage_name, None,
                             "actual candidate runtime binary identity is not bound to build evidence")
    if (binary.get("sha256") != candidate_binary.get("sha256")
            or binary.get("size_bytes") != candidate_binary.get("size_bytes")):
        raise _PipelineFault("regression", stage_name, "candidate",
                             "loaded runtime binary differs from the independently hashed build output")
    if runtime.get("release_source_commit") != source_commit:
        raise _PipelineFault("regression", stage_name, "candidate",
                             "loaded runtime source commit differs from the frozen submission")
    baseline_runtime = expected.get("runtime")
    baseline_binary = baseline_runtime.get("binary") if isinstance(baseline_runtime, Mapping) else None
    return {
        "compatibility_matched": True,
        "mismatches": [],
        "candidate_binary_sha256": binary["sha256"],
        "candidate_binary_size_bytes": binary["size_bytes"],
        "release_source_commit": runtime["release_source_commit"],
        "baseline_binary_sha256": baseline_binary.get("sha256") if isinstance(baseline_binary, Mapping) else None,
    }


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise EvaluationError(f"{field} must be a SHA-256 hex digest")
    return value.lower()


def _make_attempt_dirs(run_root: Path, submission_id: str) -> dict[str, Path | str]:
    root = Path(run_root).expanduser()
    if root.exists() and root.is_symlink():
        raise EvaluationError("run_root must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise EvaluationError("run_root is not a directory")
    root.chmod(0o700)
    attempts = root / "attempts"
    attempts.mkdir(mode=0o700, exist_ok=True)
    if attempts.is_symlink() or not attempts.is_dir():
        raise EvaluationError("run_root/attempts must be a real directory")
    try:
        attempts.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvaluationError("run_root/attempts resolves outside run_root") from exc
    attempts.chmod(0o700)
    attempt_id = uuid.uuid4().hex
    submission_key = hashlib.sha256(submission_id.encode("utf-8", errors="replace")).hexdigest()[:16]
    attempt = attempts / f"{submission_key}-{attempt_id}"
    attempt.mkdir(mode=0o700)
    paths: dict[str, Path | str] = {
        "run_root": root,
        "attempt_id": attempt_id,
        "attempt_dir": attempt,
        "archive": attempt / "source.tar",
        "source_dir": attempt / "source",
        "build_dir": attempt / "build",
        "run_dir": attempt / "run",
        "evidence_dir": attempt / "evidence",
        "artifacts_dir": attempt / "artifacts",
        "receipt_path": attempt / "receipt.json",
    }
    for key in ("build_dir", "run_dir", "evidence_dir", "artifacts_dir"):
        path = paths[key]
        assert isinstance(path, Path)
        path.mkdir(mode=0o700)
    return paths


def _basic_receipt(
    submission: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None,
    *,
    verdict: str,
    stage: str,
    failure_class: str | None,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    submission = submission or {}
    policy = policy or {}
    source = submission.get("source") if isinstance(submission, Mapping) else None
    source = source if isinstance(source, Mapping) else {}
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": submission.get("submission_id"),
        "policy_id": policy.get("policy_id"),
        "contract_id": policy.get("contract_id", submission.get("contract_id")),
        "source_snapshot_sha256": source.get("snapshot_sha256"),
        "verdict": verdict,
        "stage": stage,
        "failure_class": failure_class,
        "evidence": list(evidence or []),
        "created_at": _utc_now(),
    }
    candidate_commit = submission.get("candidate_commit")
    if isinstance(candidate_commit, str) and candidate_commit:
        receipt["candidate_commit"] = candidate_commit
    base_commit = submission.get("base_commit")
    if isinstance(base_commit, str) and base_commit:
        receipt["base_commit"] = base_commit
    qualification_enabled = policy.get("qualification_enabled")
    if type(qualification_enabled) is bool:
        receipt["qualification_enabled"] = qualification_enabled
    return receipt


def _persist_receipt(core: Any, paths: Mapping[str, Path | str] | None,
                     receipt: dict[str, Any], *, validated: bool) -> dict[str, Any]:
    if paths is None:
        return receipt
    final = receipt
    if validated:
        final = core.validate_receipt(receipt)
        if not isinstance(final, dict):
            raise EvaluationError("shared receipt validator did not return a JSON object")
    receipt_path = paths["receipt_path"]
    assert isinstance(receipt_path, Path)
    core.write_json(receipt_path, final)
    return final


def _artifact_reference(core: Any, path: Path, *, kind: str, stage: str,
                        media_type: str = "application/octet-stream") -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": core.sha256_file(path).lower(),
        "kind": kind,
        "stage": stage,
        "media_type": media_type,
    }


def _write_diagnostics(core: Any, paths: Mapping[str, Path | str],
                       diagnostics: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
    artifacts_dir = paths["artifacts_dir"]
    assert isinstance(artifacts_dir, Path)
    path = artifacts_dir / "evaluation-diagnostics.json"
    core.write_json(path, diagnostics)
    path.chmod(0o400)
    evidence.append(_artifact_reference(
        core, path, kind="evaluator_diagnostics", stage=str(diagnostics.get("stage", "full")),
        media_type="application/json",
    ))


def _create_verified_archive(core: Any, source: Mapping[str, Any], destination: Path,
                             cancelled: Cancelled) -> tuple[str, dict[str, Any]]:
    raw_path = source.get("archive_path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise _SourceInputError("source archive_path must be an absolute path")
    source_path = Path(raw_path)
    try:
        link_info = source_path.lstat()
    except OSError as exc:
        raise _SourceInputError("source archive is missing") from exc
    if stat.S_ISLNK(link_info.st_mode) or not stat.S_ISREG(link_info.st_mode):
        raise _SourceInputError("source archive must be a regular non-symlink file")
    try:
        expected_hash = _sha256_text(source.get("archive_sha256"), "source.archive_sha256")
        snapshot_hash = _sha256_text(source.get("snapshot_sha256"), "source.snapshot_sha256")
    except EvaluationError as exc:
        raise _SourceInputError(str(exc)) from exc
    resolved = source_path.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise _SourceInputError("resolved source archive is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(resolved, flags)
    except OSError as exc:
        raise _SourceInputError("source archive could not be opened without following links") from exc
    digest = hashlib.sha256()
    copied_bytes = 0
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    target_fd: int | None = None
    try:
        opened_before = os.fstat(source_fd)
        if not stat.S_ISREG(opened_before.st_mode):
            raise _SourceInputError("opened source archive is not a regular file")
        target_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        while True:
            if cancelled():
                raise lifecycle.ProcessCancelled()
            chunk = os.read(source_fd, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            copied_bytes += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
        os.fsync(target_fd)
        opened_after = os.fstat(source_fd)
        file_identity = lambda info: (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
        )
        if file_identity(opened_before) != file_identity(opened_after):
            raise _SourceInputError("source archive changed while it was being frozen")
        if file_identity(before) != file_identity(opened_before):
            raise _SourceInputError("source archive path changed while it was opened")
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise _SourceInputError("source archive SHA-256 does not match the submission")
        os.fchmod(target_fd, 0o400)
        os.close(target_fd)
        target_fd = None
        os.replace(temporary, destination)
        staged_hash = core.sha256_file(destination)
        if staged_hash.lower() != expected_hash:
            raise _SourceInputError("staged source archive failed its SHA-256 recheck")
        archive_evidence = {
            "kind": "source_archive",
            "path": str(destination),
            "sha256": staged_hash.lower(),
            "size_bytes": copied_bytes,
            "source_path": str(resolved),
            "snapshot_sha256": snapshot_hash,
        }
        return snapshot_hash, archive_evidence
    except lifecycle.ProcessCancelled:
        raise
    except _SourceInputError:
        raise
    except OSError as exc:
        raise _SourceInputError("source archive could not be frozen safely") from exc
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)
        temporary.unlink(missing_ok=True)


def _archive_member_path(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\0" in name:
        raise _SourceInputError(f"source archive has an unsafe path: {name!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        return ()
    if any(part in {"", ".", ".."} for part in parts):
        raise _SourceInputError(f"source archive has an unsafe path: {name!r}")
    return parts


def _safe_extract_archive(archive_path: Path, destination: Path, cancelled: Cancelled) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(mode=0o700)
    try:
        try:
            archive = tarfile.open(archive_path, mode="r:*")
        except (OSError, tarfile.TarError) as exc:
            raise _SourceInputError("source archive is not a readable tar archive") from exc
        with archive:
            members = archive.getmembers()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise _SourceInputError("source archive contains too many members")
            by_path: dict[tuple[str, ...], tarfile.TarInfo] = {}
            paths: list[tuple[tuple[str, ...], tarfile.TarInfo]] = []
            symlink_paths: set[tuple[str, ...]] = set()
            total_size = 0
            for member in members:
                if cancelled():
                    raise lifecycle.ProcessCancelled()
                parts = _archive_member_path(member.name)
                if not parts:
                    if member.isdir():
                        continue
                    raise _SourceInputError("source archive has a non-directory root member")
                prior = by_path.get(parts)
                if prior is not None:
                    if prior.isdir() and member.isdir():
                        continue
                    raise _SourceInputError(f"source archive repeats member path: {member.name!r}")
                if member.issym():
                    target = PurePosixPath(member.linkname)
                    normalized_target = posixpath.normpath(
                        posixpath.join("/".join(parts[:-1]), member.linkname)
                    )
                    if target.is_absolute() or normalized_target == ".." or normalized_target.startswith("../"):
                        raise _SourceInputError(f"source archive has an escaping symlink: {member.name!r}")
                    symlink_paths.add(parts)
                elif member.islnk():
                    raise _SourceInputError(f"source archive hard links are unsupported: {member.name!r}")
                elif not (member.isdir() or member.isfile()):
                    raise _SourceInputError(f"source archive has an unsupported file type: {member.name!r}")
                if member.isfile():
                    if member.size < 0:
                        raise _SourceInputError("source archive has a negative member size")
                    total_size += member.size
                    if total_size > _MAX_UNPACKED_BYTES:
                        raise _SourceInputError("source archive exceeds the unpacked byte limit")
                by_path[parts] = member
                paths.append((parts, member))

            for parts, member in paths:
                if any(parts[:length] in symlink_paths for length in range(1, len(parts))):
                    raise _SourceInputError("source archive places a member below a symlink")

            directories: set[tuple[str, ...]] = {()}
            for parts, member in paths:
                if member.isdir():
                    directories.add(parts)
                directories.update(parts[:length] for length in range(1, len(parts)))
            for parts in sorted(directories, key=lambda value: (len(value), value)):
                if not parts:
                    continue
                path = temporary.joinpath(*parts)
                path.mkdir(mode=0o700, parents=True, exist_ok=True)
                if path.is_symlink() or not path.is_dir():
                    raise _SourceInputError("source archive directory conflicts with a file")

            for parts, member in paths:
                if cancelled():
                    raise lifecycle.ProcessCancelled()
                path = temporary.joinpath(*parts)
                if member.isdir() or member.issym():
                    continue
                if not member.isfile():
                    raise _SourceInputError(f"unsupported source archive entry: {member.name!r}")
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise _SourceInputError(f"source archive member could not be read: {member.name!r}")
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(path, flags, 0o600)
                copied = 0
                try:
                    with source_file:
                        while copied < member.size:
                            if cancelled():
                                raise lifecycle.ProcessCancelled()
                            chunk = source_file.read(min(_HASH_CHUNK_BYTES, member.size - copied))
                            if not chunk:
                                raise _SourceInputError("source archive member ended before its declared size")
                            copied += len(chunk)
                            view = memoryview(chunk)
                            while view:
                                written = os.write(descriptor, view)
                                view = view[written:]
                    if copied != member.size:
                        raise _SourceInputError("source archive member size changed while extracting")
                    os.fsync(descriptor)
                    mode = 0o500 if member.mode & 0o111 else 0o400
                    os.fchmod(descriptor, mode)
                finally:
                    os.close(descriptor)

            for parts, member in paths:
                if not member.issym():
                    continue
                if cancelled():
                    raise lifecycle.ProcessCancelled()
                path = temporary.joinpath(*parts)
                if path.exists() or path.is_symlink():
                    raise _SourceInputError("source archive symlink conflicts with another entry")
                os.symlink(member.linkname, path)

        for directory in sorted(
            (item for item in temporary.rglob("*") if item.is_dir() and not item.is_symlink()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            directory.chmod(0o500)
        temporary.chmod(0o500)
        os.replace(temporary, destination)
    except lifecycle.ProcessCancelled:
        raise
    except _SourceInputError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise _SourceInputError("source archive could not be extracted safely") from exc
    finally:
        if temporary.exists():
            _make_writable_and_remove(temporary)


def _make_writable_and_remove(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        for child in path.iterdir():
            _make_writable_and_remove(child)
        path.chmod(0o700)
        path.rmdir()
    else:
        path.chmod(0o600)
        path.unlink(missing_ok=True)


def _extracted_source_manifest(core: Any, source_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = path.relative_to(source_dir).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries.append({
                    "path": relative,
                    "type": "symlink",
                    "mode": 0o120000,
                    "target": os.readlink(path),
                })
            elif stat.S_ISDIR(info.st_mode):
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                entries.append({
                    "path": relative,
                    "type": "file",
                    "mode": 0o100755 if info.st_mode & 0o111 else 0o100644,
                    "sha256": core.sha256_file(path),
                })
            else:
                raise _SourceInputError(f"unsupported extracted source object: {relative}")

    visit(source_dir)
    # The frozen manifest is ordered by path, which is not the same order a
    # directory walk produces. Compare like with like.
    entries.sort(key=lambda entry: entry["path"])
    return entries


def _policy_stages(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_stages = policy.get("stages")
    if not isinstance(raw_stages, list) or [
        item.get("name") if isinstance(item, Mapping) else None for item in raw_stages
    ] != list(STAGE_ORDER):
        raise EvaluationError("trusted policy must define all six ordered stages")
    stages: list[dict[str, Any]] = []
    contract_observers = 0
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, Mapping):
            raise EvaluationError("each policy stage must be a JSON object")
        stage = dict(raw_stage)
        name = stage.get("name")
        if set(stage) != {"name", "argv", "timeout_seconds", "required_evidence", "criteria"}:
            raise EvaluationError(f"{name} stage does not match the shared policy-stage schema")
        argv = stage.get("argv")
        if not isinstance(argv, list) or not argv or any(
            not isinstance(item, str) or not item or "\0" in item for item in argv
        ):
            raise EvaluationError(f"{name} stage argv must be a nonempty string array")
        timeout = stage.get("timeout_seconds")
        if type(timeout) not in {int, float} or not 0 < timeout <= 86_400:
            raise EvaluationError(f"{name} stage timeout_seconds must be within (0, 86400]")
        required_evidence = stage.get("required_evidence")
        if not isinstance(required_evidence, list) or not required_evidence:
            raise EvaluationError(f"{name} stage requires named evidence artifacts")
        if any(not _is_nonempty(item) for item in required_evidence) \
                or len(set(required_evidence)) != len(required_evidence):
            raise EvaluationError(f"{name} required_evidence names must be unique nonempty strings")
        raw_criteria = stage.get("criteria")
        if not isinstance(raw_criteria, Mapping):
            raise EvaluationError(f"{name} criteria must be a JSON object")
        criteria = dict(raw_criteria)
        allowed_criteria = {
            "equals", "minimum", "maximum", "gates", "observes_contract",
            "execution_mode", "label", "requires_gpu", "gpu_admission",
            "toolchain", "build_identity", "cwd", "env",
        }
        if criteria.keys() - allowed_criteria:
            raise EvaluationError(f"{name} criteria contain unsupported fields")
        mode = criteria.get("execution_mode", "isolated_candidate")
        if mode not in {"isolated_candidate", "live_endpoint"}:
            raise EvaluationError(f"{name} has an unsupported execution_mode")
        if mode == "live_endpoint" and not _is_nonempty(criteria.get("label")):
            raise EvaluationError("live_endpoint stages require an explicit nonempty label")
        if type(criteria.get("requires_gpu", False)) is not bool:
            raise EvaluationError(f"{name} requires_gpu must be a boolean criterion")
        if type(criteria.get("observes_contract", False)) is not bool:
            raise EvaluationError("observes_contract must be a boolean")
        if criteria.get("observes_contract", False):
            contract_observers += 1
        cwd = criteria.get("cwd", "run")
        if cwd not in {"source", "build", "run"}:
            raise EvaluationError(f"{name} cwd must be source, build, or run")
        criteria["cwd"] = cwd
        criteria["execution_mode"] = mode
        stage["timeout_seconds"] = float(timeout)
        equals = criteria.get("equals", {})
        minimum = criteria.get("minimum", {})
        maximum = criteria.get("maximum", {})
        if not isinstance(equals, Mapping) or equals.get("status") != "pass":
            raise EvaluationError(f"{name} criteria.equals must require status=pass")
        if not isinstance(minimum, Mapping) or not isinstance(maximum, Mapping):
            raise EvaluationError(f"{name} criteria minimum/maximum must be objects")
        if any(not isinstance(key, str) for key in (*equals.keys(), *minimum.keys(), *maximum.keys())):
            raise EvaluationError(f"{name} criteria paths must be strings")
        if any(type(value) not in {int, float} for value in (*minimum.values(), *maximum.values())):
            raise EvaluationError(f"{name} criteria numeric bounds must be numbers")
        if any(
            key in maximum and minimum[key] > maximum[key]
            for key in minimum
        ):
            raise EvaluationError(f"{name} criteria minimum exceeds its maximum")
        gates = criteria.get("gates", [])
        if not isinstance(gates, list) or any(
            not isinstance(gate, str) or gate not in QUALIFICATION_GATES for gate in gates
        ) or len(gates) != len(set(gates)):
            raise EvaluationError(f"{name} gates must be unique semantic/task/holdout labels")
        if (gates or name in {"screen", "full"}) and not (
            any(key != "status" for key in equals) or minimum or maximum
        ):
            raise EvaluationError(f"{name} quality/performance gate needs criteria beyond status=pass")
        if not equals and not minimum and not maximum:
            raise EvaluationError(f"{name} criteria cannot be empty")
        if criteria.get("requires_gpu", False):
            gpu = criteria.get("gpu_admission")
            if not isinstance(gpu, Mapping) or not {
                "lock_path", "production_units", "target_gpu_uuid"
            } <= gpu.keys():
                raise EvaluationError(f"{name} GPU stage requires trusted gpu_admission criteria")
            if not isinstance(gpu.get("lock_path"), str) or not Path(gpu["lock_path"]).is_absolute():
                raise EvaluationError("gpu_admission.lock_path must be absolute")
            if not isinstance(gpu.get("production_units"), list) or not gpu["production_units"]:
                raise EvaluationError("gpu_admission.production_units must be nonempty")
            if any(not _is_nonempty(unit) for unit in gpu["production_units"]):
                raise EvaluationError("gpu_admission.production_units entries must be nonempty")
            if not _is_nonempty(gpu.get("target_gpu_uuid")):
                raise EvaluationError("gpu_admission.target_gpu_uuid must be nonempty")
        env = criteria.get("env", {})
        if not isinstance(env, Mapping):
            raise EvaluationError(f"{name} criteria.env must be an object")
        criteria["gates"] = list(gates)
        stage["criteria"] = criteria
        stage["required_evidence"] = list(required_evidence)
        stage["argv"] = list(argv)
        stage["evidence_path"] = f"{name}.json"
        stages.append(stage)
    if contract_observers > 1:
        raise EvaluationError("at most one stage may observe the runtime contract")
    if any(stage["name"] == "build" for stage in stages):
        toolchain = stages[0]["criteria"].get("toolchain")
        if not isinstance(toolchain, Mapping) or any(
            not _is_nonempty(toolchain.get(key)) for key in ("cmake", "cuda")
        ):
            raise EvaluationError("admission criteria must pin explicit CMake and CUDA toolchains")
        if (_version_tokens(toolchain["cmake"])[:2] != (4, 4)
                or _version_tokens(toolchain["cuda"])[:1] != (13,)):
            raise EvaluationError("NInfer bench requires explicitly pinned CMake 4.4 and CUDA 13")
        build_identity = next(stage for stage in stages if stage["name"] == "build")["criteria"].get(
            "build_identity"
        )
        if not isinstance(build_identity, Mapping) or not isinstance(
            build_identity.get("flags", {}), Mapping
        ) or not isinstance(build_identity.get("dependencies", []), list):
            raise EvaluationError("build criteria must define flags and a dependency list")
        if "candidate_binary" not in next(
            stage for stage in stages if stage["name"] == "build"
        )["required_evidence"]:
            raise EvaluationError("build stage must provide candidate_binary evidence")
    for stage in stages:
        if (stage["criteria"].get("observes_contract") is True
                and "contract_observation" not in stage["required_evidence"]):
            raise EvaluationError("contract observer must provide contract_observation evidence")
    return stages


def _is_nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _verify_build_dependencies(core: Any, dependencies: Any) -> list[dict[str, Any]]:
    if not isinstance(dependencies, list):
        raise EvaluationError("trusted build dependencies must be a list")
    verified: list[dict[str, Any]] = []
    for dependency in dependencies:
        if isinstance(dependency, str) and dependency:
            verified.append({"identity": dependency})
            continue
        if not isinstance(dependency, Mapping):
            raise EvaluationError("build dependency entries must be identities or path/hash objects")
        raw_path = dependency.get("path")
        expected = _sha256_text(dependency.get("sha256"), "build dependency sha256")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise EvaluationError("build dependency paths must be absolute")
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise EvaluationError("build dependency must be a regular non-symlink file")
        actual = core.sha256_file(path).lower()
        if actual != expected:
            raise EvaluationError("build dependency SHA-256 differs from the trusted policy")
        verified.append({"path": str(path.resolve(strict=True)), "sha256": actual})
    return verified


def _version_tokens(value: str) -> tuple[int, ...]:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+){0,3})", value)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _toolchain_value_matches(expected: Any, observed: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and all(key in observed and _toolchain_value_matches(value, observed[key])
                    for key, value in expected.items())
        )
    if isinstance(expected, str) and isinstance(observed, str):
        expected_version = _version_tokens(expected)
        observed_version = _version_tokens(observed)
        if expected_version:
            return observed_version[:len(expected_version)] == expected_version
        return expected == observed
    return expected == observed


def _build_identity(core: Any, submission: Mapping[str, Any], policy: Mapping[str, Any],
                    stages: list[dict[str, Any]], source_sha: str,
                    build_dir: Path, source_dir: Path, run_dir: Path,
                    attempt_id: str, evidence_dir: Path) -> tuple[str, dict[str, Any]]:
    build_stage = next(stage for stage in stages if stage["name"] == "build")
    build_identity_config = build_stage["criteria"].get("build_identity")
    if not isinstance(build_identity_config, Mapping):
        raise EvaluationError("build criteria must define trusted build_identity")
    dependencies = _verify_build_dependencies(core, build_identity_config.get("dependencies", []))
    admission_toolchain = stages[0]["criteria"].get("toolchain", {})
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_archive_sha256": source_sha,
        "source_snapshot_sha256": submission["source"]["snapshot_sha256"],
        "toolchain_id": policy.get("toolchain_id"),
        "toolchain": admission_toolchain,
        "build_flags": build_identity_config.get("flags", {}),
        "build_dependencies": dependencies,
        "build_command_sha256": core.canonical_hash(build_stage["argv"]),
        "build_environment_sha256": core.canonical_hash(build_stage["criteria"].get("env", {})),
    }
    build_key = core.canonical_hash(identity)
    if not isinstance(build_key, str) or not _HEX_SHA256.fullmatch(build_key):
        raise EvaluationError("shared canonical_hash returned an invalid build key")
    return build_key.lower(), identity


def _expand_argv(stage: Mapping[str, Any], paths: Mapping[str, Path | str],
                 submission: Mapping[str, Any], policy: Mapping[str, Any],
                 build_key: str, raw_dir: Path) -> list[str]:
    evidence_dir = paths["evidence_dir"]
    assert isinstance(evidence_dir, Path)
    evidence_path = evidence_dir / stage["evidence_path"]
    values = {
        "source_dir": str(paths["source_dir"]),
        "build_dir": str(paths["build_dir"]),
        "run_dir": str(paths["run_dir"]),
        "attempt_dir": str(paths["attempt_dir"]),
        "evidence_dir": str(evidence_dir),
        "evidence_path": str(evidence_path),
        "raw_dir": str(raw_dir),
        "attempt_id": str(paths["attempt_id"]),
        "submission_id": str(submission["submission_id"]),
        "policy_id": str(policy["policy_id"]),
        "contract_id": str(policy["contract_id"]),
        "source_snapshot_sha256": str(submission["source"]["snapshot_sha256"]),
        "build_key": build_key,
        "stage": str(stage["name"]),
    }
    return [
        _replace_known_placeholders(argument, values)
        for argument in stage["argv"]
    ]


def _replace_known_placeholders(argument: str, values: Mapping[str, str]) -> str:
    # Literal braces in Python/CMake snippets are common; replace only explicit
    # tokens so trusted argv can safely embed arbitrary code without a shell.
    for name, value in values.items():
        argument = argument.replace("{" + name + "}", value)
    return argument


def _stage_environment(stage: Mapping[str, Any], paths: Mapping[str, Path | str],
                       submission: Mapping[str, Any], policy: Mapping[str, Any],
                       build_key: str, attempt_id: str,
                       raw_dir: Path) -> dict[str, str]:
    evidence_dir = paths["evidence_dir"]
    assert isinstance(evidence_dir, Path)
    evidence_path = evidence_dir / stage["evidence_path"]
    environment = lifecycle.sanitized_environment()
    environment.update({
        "NINFER_SOURCE_DIR": str(paths["source_dir"]),
        "NINFER_BUILD_DIR": str(paths["build_dir"]),
        "NINFER_RUN_DIR": str(paths["run_dir"]),
        "NINFER_ATTEMPT_DIR": str(paths["attempt_dir"]),
        "NINFER_EVIDENCE_DIR": str(evidence_dir),
        "NINFER_STAGE_EVIDENCE_PATH": str(evidence_path),
        "NINFER_STAGE_RAW_DIR": str(raw_dir),
        "NINFER_STAGE": str(stage["name"]),
        "NINFER_REPORT_SCHEMA": str(SCHEMA_VERSION),
        "NINFER_ATTEMPT_ID": attempt_id,
        "NINFER_SUBMISSION_ID": str(submission["submission_id"]),
        "NINFER_POLICY_ID": str(policy["policy_id"]),
        "NINFER_CONTRACT_ID": str(policy["contract_id"]),
        "NINFER_SOURCE_SNAPSHOT_SHA256": str(submission["source"]["snapshot_sha256"]),
        "NINFER_BUILD_KEY": build_key,
        "NINFER_TOOLCHAIN_ID": str(policy.get("toolchain_id", "")),
        # The stage program must publish exactly these evidence names.
        "NINFER_STAGE_REQUIRED_EVIDENCE": json.dumps(list(stage["required_evidence"])),
        "NINFER_STAGE_CRITERIA": json.dumps(stage["criteria"], sort_keys=True),
    })
    configured = stage["criteria"].get("env", {})
    if not isinstance(configured, Mapping):
        raise EvaluationError(f"{stage['name']} env must be a JSON object")
    for name, value in configured.items():
        if not isinstance(name, str) or not isinstance(value, str) or "\0" in name + value:
            raise EvaluationError(f"{stage['name']} env has an invalid string entry")
        if any(marker in name.upper() for marker in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH")):
            raise EvaluationError(f"{stage['name']} env cannot provide credential-bearing variables")
        environment[name] = value
    return environment


def _prepare_stage_evidence_path(stage: Mapping[str, Any], evidence_dir: Path) -> Path:
    path = evidence_dir / stage["evidence_path"]
    if path.exists() or path.is_symlink():
        raise EvaluationError("stage evidence output already exists before the stage")
    return path


def _immutable_snapshot(core: Any, source: Path, destination: Path) -> dict[str, Any]:
    try:
        source_info = source.lstat()
    except OSError as exc:
        raise _EvidenceError("required raw evidence artifact is missing", inconclusive=True) from exc
    if stat.S_ISLNK(source_info.st_mode) or not stat.S_ISREG(source_info.st_mode):
        raise _EvidenceError("raw evidence artifact must be a regular non-symlink file", inconclusive=True)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    target_fd: int | None = None
    size = 0
    try:
        before = os.fstat(source_fd)
        target_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            chunk = os.read(source_fd, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
        os.fsync(target_fd)
        after = os.fstat(source_fd)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if identity(before) != identity(after):
            raise _EvidenceError("raw evidence changed while it was frozen", inconclusive=True)
        os.fchmod(target_fd, 0o400)
        os.close(target_fd)
        target_fd = None
    except BaseException:
        if target_fd is not None:
            os.close(target_fd)
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
    digest = core.sha256_file(destination).lower()
    return {"path": str(destination), "sha256": digest, "size_bytes": size}


def _safe_relative_file(root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str):
        raise _EvidenceError("raw evidence path is not a string", inconclusive=True)
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise _EvidenceError("raw evidence path escapes the run directory", inconclusive=True)
    root = root.resolve(strict=True)
    candidate = root.joinpath(*relative.parts)
    cursor = root
    try:
        for index, part in enumerate(relative.parts):
            cursor = cursor / part
            info = cursor.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise _EvidenceError("raw evidence path contains a symbolic link", inconclusive=True)
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise _EvidenceError("raw evidence parent is not a directory", inconclusive=True)
            if index == len(relative.parts) - 1 and not stat.S_ISREG(info.st_mode):
                raise _EvidenceError("raw evidence must be a regular file", inconclusive=True)
        candidate.resolve(strict=True).relative_to(root)
    except _EvidenceError:
        raise
    except (OSError, ValueError) as exc:
        raise _EvidenceError("raw evidence file is missing or escaped", inconclusive=True) from exc
    return candidate


def _snapshot_required_evidence(core: Any, report: Mapping[str, Any], *, stage: Mapping[str, Any],
                                attempt_dir: Path, artifacts_dir: Path,
                                receipt_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references = report.get("evidence")
    required = stage["required_evidence"]
    if not isinstance(references, Mapping) or set(references) != set(required):
        raise _EvidenceError("stage report does not provide exactly its required evidence names",
                             inconclusive=True)
    snapshots: list[dict[str, Any]] = []
    for index, name in enumerate(required):
        reference = references[name]
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise _EvidenceError(f"raw evidence reference {name!r} must contain path and sha256",
                                 inconclusive=True)
        relative_path = reference.get("path")
        parts = PurePosixPath(relative_path).parts if isinstance(relative_path, str) else ()
        if not parts or parts[0] not in {"build", "run"}:
            raise _EvidenceError(f"raw evidence {name!r} must be inside build or run directory",
                                 inconclusive=True)
        if name == "candidate_binary" and (stage["name"] != "build" or parts[0] != "build"):
            raise _EvidenceError("candidate_binary evidence must be a build-stage file",
                                 inconclusive=True)
        source = _safe_relative_file(attempt_dir, relative_path)
        if name == "candidate_binary" and not source.stat().st_mode & 0o111:
            raise _EvidenceError("candidate_binary evidence is not executable", inconclusive=True)
        try:
            expected_hash = _sha256_text(reference.get("sha256"), f"{name} sha256")
            actual_hash = core.sha256_file(source).lower()
        except (EvaluationError, OSError) as exc:
            raise _EvidenceError(f"raw evidence {name!r} hash could not be verified",
                                 inconclusive=True) from exc
        if actual_hash != expected_hash:
            raise _EvidenceError(f"raw evidence {name!r} SHA-256 differs from its stage report",
                                 inconclusive=True)
        snapshot = _immutable_snapshot(
            core, source, artifacts_dir / f"{stage['name']}.evidence-{index:03d}"
        )
        if snapshot["sha256"] != expected_hash:
            raise _EvidenceError(f"raw evidence {name!r} changed while it was snapshotted",
                                 inconclusive=True)
        if snapshot["size_bytes"] == 0:
            raise _EvidenceError(f"raw evidence {name!r} is empty", inconclusive=True)
        reference_snapshot = _artifact_reference(
            core,
            Path(snapshot["path"]),
            kind=name,
            stage=str(stage["name"]),
        )
        receipt_evidence.append(reference_snapshot)
        snapshots.append({"name": name, **snapshot})
    return snapshots


def _path_value(value: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
    current: Any = value
    for part in dotted_path.split("."):
        if not part or not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _validate_evidence_report(core: Any, *, report_path: Path, artifact_path: Path,
                              evidence_root: Path,
                              attempt_dir: Path, artifacts_dir: Path,
                              stage: Mapping[str, Any], submission: Mapping[str, Any],
                              policy: Mapping[str, Any], attempt_id: str,
                              build_key: str,
                              receipt_evidence: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        relative = report_path.relative_to(evidence_root)
        cursor = evidence_root
        for index, part in enumerate(relative.parts):
            cursor = cursor / part
            info = cursor.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise _EvidenceError("stage evidence path contains a symbolic link", inconclusive=True)
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise _EvidenceError("stage evidence parent is not a directory", inconclusive=True)
        report_path.resolve(strict=True).relative_to(evidence_root.resolve(strict=True))
    except _EvidenceError:
        raise
    except (OSError, ValueError) as exc:
        raise _EvidenceError("stage evidence path escaped or is missing", inconclusive=True) from exc
    if report_path.is_symlink() or not report_path.is_file():
        raise _EvidenceError("successful stage omitted its required evidence report", inconclusive=True)
    artifact = _immutable_snapshot(core, report_path, artifact_path)
    try:
        report = core.read_json(artifact_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise _EvidenceError("stage evidence report is not valid JSON", inconclusive=True) from exc
    if not isinstance(report, Mapping):
        raise _EvidenceError("stage evidence report must be a JSON object", inconclusive=True)
    report = dict(report)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise _EvidenceError("stage evidence report has an unsupported schema_version", inconclusive=True)
    binding = {
        "submission_id": submission["submission_id"],
        "policy_id": policy["policy_id"],
        "contract_id": policy["contract_id"],
        "source_snapshot_sha256": submission["source"]["snapshot_sha256"],
        "attempt_id": attempt_id,
        "stage": stage["name"],
    }
    for field, expected in binding.items():
        if report.get(field) != expected:
            raise _EvidenceError(f"stage evidence is missing or misbound at {field}", inconclusive=True)
    snapshots = _snapshot_required_evidence(
        core,
        report,
        stage=stage,
        attempt_dir=attempt_dir,
        artifacts_dir=artifacts_dir,
        receipt_evidence=receipt_evidence,
    )
    if stage["name"] == "build" and report.get("build_key") != build_key:
        raise _EvidenceError("build evidence does not bind the complete build identity", inconclusive=True)
    if stage["name"] == "admission" and _is_nonempty(policy.get("toolchain_id")):
        if report.get("toolchain_id") != policy.get("toolchain_id"):
            raise _EvidenceError("host admission evidence does not match the trusted toolchain_id",
                                 failure_class="infrastructure")
        expected_toolchain = stage["criteria"].get("toolchain")
        observed_toolchain = report.get("toolchain")
        if not isinstance(expected_toolchain, Mapping) or not isinstance(observed_toolchain, Mapping):
            raise _EvidenceError("host admission did not report actual CMake/CUDA toolchain identity",
                                 failure_class="infrastructure")
        for name in ("cmake", "cuda"):
            if not _is_nonempty(expected_toolchain.get(name)):
                raise _EvidenceError(f"trusted toolchain policy omits explicit {name} identity",
                                     failure_class="infrastructure")
            if not _toolchain_value_matches(expected_toolchain[name], observed_toolchain.get(name)):
                raise _EvidenceError(f"actual {name} toolchain differs from the trusted policy",
                                     failure_class="infrastructure")

    report_status = report.get("status")
    if isinstance(report_status, str) and report_status.lower() in {
        "inconclusive", "noise", "noisy", "underpowered", "insufficient"
    }:
        raise _EvidenceError("stage evidence is noisy or inconclusive", inconclusive=True)
    if isinstance(report_status, str) and report_status.lower() in {"regression", "fail", "failed"}:
        raise _EvidenceError("stage evidence reports a failing gate")

    criteria = stage["criteria"]
    for path, expected in criteria["equals"].items():
        exists, actual = _path_value(report, path)
        if not exists:
            raise _EvidenceError(f"stage evidence is missing required criterion {path}", inconclusive=True)
        if actual != expected:
            raise _EvidenceError(f"stage evidence did not meet trusted equality criterion {path}")
    for path, lower in criteria.get("minimum", {}).items():
        exists, actual = _path_value(report, path)
        if not exists:
            raise _EvidenceError(f"stage evidence is missing numeric criterion {path}", inconclusive=True)
        if type(actual) not in {int, float} or actual < lower:
            raise _EvidenceError(f"stage evidence did not meet trusted minimum criterion {path}")
    for path, upper in criteria.get("maximum", {}).items():
        exists, actual = _path_value(report, path)
        if not exists:
            raise _EvidenceError(f"stage evidence is missing numeric criterion {path}", inconclusive=True)
        if type(actual) not in {int, float} or actual > upper:
            raise _EvidenceError(f"stage evidence did not meet trusted maximum criterion {path}")
    return report, artifact, snapshots


def _stage_cwd(stage: Mapping[str, Any], paths: Mapping[str, Path | str]) -> Path:
    directory = paths[f"{stage['criteria']['cwd']}_dir"]
    assert isinstance(directory, Path)
    return directory


def _record_partial_log(core: Any, log_path: Path, artifacts_dir: Path,
                        stage_name: str) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    destination = artifacts_dir / f"{stage_name}.log"
    try:
        if destination.exists():
            return None
        artifact = _immutable_snapshot(core, log_path, destination)
    except (OSError, EvaluationError):
        return None
    return artifact


def _run_stage(core: Any, stage: Mapping[str, Any], *, paths: Mapping[str, Path | str],
               submission: Mapping[str, Any], policy: Mapping[str, Any], attempt_id: str,
               build_key: str, cancelled: Cancelled,
               receipt_evidence: list[dict[str, Any]],
               stage_diagnostics: list[dict[str, Any]],
               gpu_admissions: list[dict[str, Any]]) -> dict[str, Any]:
    source_dir = paths["source_dir"]
    build_dir = paths["build_dir"]
    run_dir = paths["run_dir"]
    evidence_dir = paths["evidence_dir"]
    artifacts_dir = paths["artifacts_dir"]
    assert all(isinstance(value, Path) for value in (source_dir, build_dir, run_dir, evidence_dir, artifacts_dir))
    report_path = _prepare_stage_evidence_path(stage, evidence_dir)
    raw_parent = run_dir / "raw"
    if raw_parent.exists() and (raw_parent.is_symlink() or not raw_parent.is_dir()):
        raise EvaluationError("stage raw-evidence root is not a safe directory")
    raw_parent.mkdir(mode=0o700, exist_ok=True)
    raw_dir = raw_parent / str(stage["name"])
    if raw_dir.exists() or raw_dir.is_symlink():
        raise EvaluationError("stage raw-evidence directory already exists")
    raw_dir.mkdir(mode=0o700)
    argv = _expand_argv(stage, paths, submission, policy, build_key, raw_dir)
    environment = _stage_environment(
        stage, paths, submission, policy, build_key, attempt_id, raw_dir
    )
    log_path = run_dir / f"{stage['name']}.log"
    criteria = stage["criteria"]
    gpu_required = criteria.get("requires_gpu", False)
    if gpu_required:
        admission = criteria.get("gpu_admission")
        if not isinstance(admission, Mapping):
            raise _PipelineFault("blocked", stage["name"], "infrastructure",
                                 "GPU stage has no trusted admission criteria")
        lock_path = admission.get("lock_path")
        units = admission.get("production_units")
        target_uuid = admission.get("target_gpu_uuid")
        if not isinstance(lock_path, str) or not Path(lock_path).is_absolute():
            raise _PipelineFault("blocked", stage["name"], "infrastructure",
                                 "GPU stage requires an absolute trusted lock_path")
        admission_context = lifecycle.gpu_admission(
            Path(lock_path),
            units,
            target_uuid=target_uuid,
            lease_wait_seconds=float(admission.get("lease_wait_seconds", 30.0)),
            cancelled=cancelled,
        )
    else:
        admission_context = nullcontext(None)

    stage_result: dict[str, Any] | None = None
    admission_record: dict[str, Any] | None = None
    try:
        with admission_context as admission:
            if admission is not None:
                admission_record = {"stage": stage["name"], **admission}
                gpu_admissions.append(admission_record)
            stage_result = lifecycle.run_managed(
                argv,
                log_path=log_path,
                cwd=_stage_cwd(stage, paths),
                env=environment,
                timeout_seconds=stage["timeout_seconds"],
                cancelled=cancelled,
            )
    except lifecycle.ProcessCancelled as exc:
        cleanup = getattr(exc, "cleanup", None)
        log_artifact = _record_partial_log(core, log_path, artifacts_dir, stage["name"])
        if log_artifact is not None:
            receipt_evidence.append(_artifact_reference(
                core, Path(log_artifact["path"]), kind="stage_log", stage=stage["name"],
                media_type="text/plain",
            ))
        stage_diagnostics.append({
            "result": "cancelled",
            "stage": stage["name"],
            "command_sha256": core.canonical_hash(argv),
            "process_cleanup": cleanup,
            "log": log_artifact,
        })
        if cleanup and cleanup.get("process_group_alive"):
            raise _PipelineFault("blocked", stage["name"], "infrastructure",
                                 "cancellation teardown left a process group alive") from exc
        raise _PipelineFault("blocked", stage["name"], None,
                             "evaluation was cancelled after process-group teardown") from exc
    except lifecycle.ProcessTimeout as exc:
        cleanup = exc.cleanup
        log_artifact = _record_partial_log(core, log_path, artifacts_dir, stage["name"])
        if log_artifact is not None:
            receipt_evidence.append(_artifact_reference(
                core, Path(log_artifact["path"]), kind="stage_log", stage=stage["name"],
                media_type="text/plain",
            ))
        stage_diagnostics.append({
            "result": "timeout",
            "stage": stage["name"],
            "command_sha256": core.canonical_hash(argv),
            "process_cleanup": cleanup,
            "log": log_artifact,
        })
        if cleanup.get("process_group_alive"):
            raise _PipelineFault("blocked", stage["name"], "infrastructure",
                                 "timed-out stage left a process group alive") from exc
        raise _PipelineFault(
            "blocked" if stage["name"] == "admission" else "regression",
            stage["name"],
            "infrastructure" if stage["name"] == "admission" else "candidate",
            f"trusted {stage['name']} stage exceeded its timeout; cleanup={cleanup}",
        ) from exc
    except (lifecycle.GpuAdmissionError, lifecycle.ProductionServiceActive) as exc:
        stage_diagnostics.append({
            "result": "blocked_gpu_admission",
            "stage": stage["name"],
            "reason": str(exc),
        })
        raise _PipelineFault("blocked", stage["name"], "infrastructure", str(exc)) from exc
    except lifecycle.ProcessCleanupError as exc:
        log_artifact = _record_partial_log(core, log_path, artifacts_dir, stage["name"])
        if log_artifact is not None:
            receipt_evidence.append(_artifact_reference(
                core, Path(log_artifact["path"]), kind="stage_log", stage=stage["name"],
                media_type="text/plain",
            ))
        stage_diagnostics.append({
            "result": "unclean",
            "stage": stage["name"],
            "command_sha256": core.canonical_hash(argv),
            "process_cleanup": exc.cleanup,
            "log": log_artifact,
        })
        raise _PipelineFault("blocked", stage["name"], "infrastructure",
                             f"stage process-group teardown failed: {exc.cleanup}") from exc
    except (OSError, subprocess.SubprocessError, EvaluationError) as exc:
        stage_diagnostics.append({
            "result": "infrastructure_error",
            "stage": stage["name"],
            "exception_type": type(exc).__name__,
        })
        raise _PipelineFault("blocked", stage["name"], "infrastructure",
                             f"trusted {stage['name']} stage could not run: {type(exc).__name__}") from exc

    assert stage_result is not None
    log_artifact: dict[str, Any] | None = None
    if log_path.is_file():
        log_artifact = _immutable_snapshot(core, log_path, artifacts_dir / f"{stage['name']}.log")
        receipt_evidence.append(_artifact_reference(
            core, Path(log_artifact["path"]), kind="stage_log", stage=stage["name"],
            media_type="text/plain",
        ))
    if stage_result["returncode"] != 0:
        failure_class = "infrastructure" if stage["name"] == "admission" else "candidate"
        verdict = "blocked" if failure_class == "infrastructure" else (
            "rejected" if stage["name"] == "build" else "regression"
        )
        stage_diagnostics.append({
            "result": "failed",
            "stage": stage["name"],
            "command_sha256": core.canonical_hash(argv),
            "executable": argv[0],
            "argument_count": len(argv),
            "returncode": stage_result["returncode"],
            "elapsed_seconds": stage_result["elapsed_seconds"],
            "log": log_artifact,
            "cleanup": stage_result["cleanup"],
        })
        raise _PipelineFault(verdict, stage["name"], failure_class,
                             f"trusted {stage['name']} stage exited {stage_result['returncode']}")
    report_artifact_path = artifacts_dir / f"{stage['name']}.report.json"
    try:
        report, report_artifact, required_artifacts = _validate_evidence_report(
            core,
            report_path=report_path,
            artifact_path=report_artifact_path,
            evidence_root=evidence_dir,
            attempt_dir=Path(paths["attempt_dir"]),
            artifacts_dir=artifacts_dir,
            stage=stage,
            submission=submission,
            policy=policy,
            attempt_id=attempt_id,
            build_key=build_key,
            receipt_evidence=receipt_evidence,
        )
    except _EvidenceError as exc:
        if report_artifact_path.is_file():
            receipt_evidence.append(_artifact_reference(
                core, report_artifact_path, kind="stage_report", stage=stage["name"],
                media_type="application/json",
            ))
        stage_diagnostics.append({
            "result": "evidence_rejected",
            "stage": stage["name"],
            "command_sha256": core.canonical_hash(argv),
            "executable": argv[0],
            "argument_count": len(argv),
            "returncode": stage_result["returncode"],
            "elapsed_seconds": stage_result["elapsed_seconds"],
            "log": log_artifact,
            "cleanup": stage_result["cleanup"],
        })
        if stage["name"] == "admission" and exc.failure_class is None:
            verdict, failure_class = "blocked", "infrastructure"
        else:
            verdict = (
                "blocked" if exc.failure_class == "infrastructure"
                else "inconclusive" if exc.inconclusive
                else "regression"
            )
            failure_class = exc.failure_class or (None if exc.inconclusive else "candidate")
        raise _PipelineFault(verdict, stage["name"], failure_class, str(exc)) from exc
    receipt_evidence.append(_artifact_reference(
        core, report_artifact_path, kind="stage_report", stage=stage["name"],
        media_type="application/json",
    ))
    stage_diagnostic = {
        "result": "passed",
        "stage": stage["name"],
        "command_sha256": core.canonical_hash(argv),
        "executable": argv[0],
        "argument_count": len(argv),
        "returncode": stage_result["returncode"],
        "elapsed_seconds": stage_result["elapsed_seconds"],
        "execution_mode": criteria["execution_mode"],
        "label": criteria.get("label"),
        "requires_gpu": gpu_required,
        "gates": list(criteria["gates"]),
        "log": log_artifact,
        "report": report_artifact,
        "cleanup": stage_result["cleanup"],
    }
    stage_diagnostics.append(stage_diagnostic)
    if criteria.get("observes_contract") is True:
        observed = report.get("observed_contract")
        if not isinstance(observed, Mapping):
            raise _PipelineFault("inconclusive", stage["name"], None,
                                 "contract-observation stage omitted observed_contract")
    stage_diagnostic["required_evidence"] = required_artifacts
    if admission_record is not None:
        stage_diagnostic["gpu_admission"] = admission_record
    return report


def _qualification_gaps(policy: Mapping[str, Any], stages: list[dict[str, Any]],
                        gates_seen: set[str], contract_seen: bool,
                        gpu_admissions: list[dict[str, Any]]) -> list[str]:
    gaps: list[str] = []
    names = {stage["name"] for stage in stages}
    missing_stages = [name for name in STAGE_ORDER if name not in names]
    if missing_stages:
        gaps.append("missing stages: " + ", ".join(missing_stages))
    missing_gates = sorted(QUALIFICATION_GATES - gates_seen)
    if missing_gates:
        gaps.append("missing semantic/task/holdout gates: " + ", ".join(missing_gates))
    for gate in QUALIFICATION_GATES & gates_seen:
        owners = [stage["name"] for stage in stages if gate in stage["criteria"]["gates"]]
        if len(owners) != 1:
            gaps.append(f"{gate} gate must have exactly one trusted evidence stage")
    owners_by_gate = [
        next((stage["name"] for stage in stages if gate in stage["criteria"]["gates"]), None)
        for gate in sorted(QUALIFICATION_GATES)
        if gate in gates_seen
    ]
    if len(owners_by_gate) != len(set(owners_by_gate)):
        gaps.append("semantic, task, and holdout gates require distinct evidence stages")
    if not contract_seen:
        gaps.append("missing trusted observed-contract verification")
    if not _is_nonempty(policy.get("corpus_id")):
        gaps.append("qualification has no trusted corpus_id")
    if not _is_nonempty(policy.get("toolchain_id")):
        gaps.append("qualification has no trusted toolchain_id")
    if not _is_nonempty(policy.get("baseline_id")):
        gaps.append("qualification has no trusted baseline_id")
    if not gpu_admissions:
        gaps.append("qualification has no exclusively admitted model-GPU stage")
    if any(stage["criteria"]["execution_mode"] == "live_endpoint" for stage in stages):
        gaps.append("live-endpoint plumbing cannot establish candidate performance")
    return gaps


def evaluate_submission(
    submission: dict[str, Any],
    policy: dict[str, Any],
    run_root: Path,
    *,
    cancelled: Cancelled = lambda: False,
) -> dict[str, Any]:
    """Evaluate one frozen source archive under independently trusted policy gates."""
    if not callable(cancelled):
        raise TypeError("cancelled must be a callable")
    if not isinstance(submission, Mapping) or not isinstance(policy, Mapping):
        raise TypeError("submission and policy must be JSON objects")
    core = _contracts_api()
    paths: dict[str, Path | str] | None = None
    receipt_evidence: list[dict[str, Any]] = []
    stage_diagnostics: list[dict[str, Any]] = []
    submission_validated = False
    policy_validated = False
    current_stage = "admission"
    verdict = "inconclusive"
    failure_class: str | None = None
    detail = "evaluation did not complete"
    normalized_submission: dict[str, Any] = dict(submission)
    normalized_policy: dict[str, Any] = dict(policy)
    build_key: str | None = None
    build_identity: dict[str, Any] | None = None
    gates_seen: set[str] = set()
    gpu_admissions: list[dict[str, Any]] = []
    contract_seen = False
    stages: list[dict[str, Any]] = []

    raw_submission_id = normalized_submission.get("submission_id")
    if not isinstance(raw_submission_id, str):
        raw_submission_id = "invalid-submission"
    try:
        paths = _make_attempt_dirs(Path(run_root), raw_submission_id)
    except (OSError, EvaluationError, ValueError) as exc:
        receipt = _basic_receipt(
            normalized_submission, normalized_policy, verdict="blocked", stage="admission",
            failure_class="infrastructure",
        )
        return receipt

    try:
        try:
            normalized_submission = core.validate_submission(dict(submission))
            if not isinstance(normalized_submission, dict):
                raise ValueError("validator did not return an object")
            submission_validated = True
        except (ValueError, TypeError, KeyError) as exc:
            raise _PipelineFault("rejected", "admission", "candidate",
                                 f"submission validation failed: {type(exc).__name__}") from exc
        try:
            normalized_policy = core.validate_policy(dict(policy))
            if not isinstance(normalized_policy, dict):
                raise ValueError("validator did not return an object")
            policy_validated = True
        except (ValueError, TypeError, KeyError) as exc:
            raise _PipelineFault("blocked", "admission", "infrastructure",
                                 f"trusted policy validation failed: {exc}") from exc

        submission_id = _identity_text(normalized_submission.get("submission_id"), "submission_id")
        policy_id = _identity_text(normalized_policy.get("policy_id"), "policy_id")
        contract_id = _identity_text(normalized_policy.get("contract_id"), "contract_id")
        if normalized_submission.get("contract_id") != contract_id:
            raise _PipelineFault("rejected", "admission", "candidate",
                                 "submission contract_id differs from trusted policy")
        if normalized_submission.get("policy_id") != policy_id:
            raise _PipelineFault("rejected", "admission", "candidate",
                                 "submission policy_id differs from trusted policy")
        source = normalized_submission.get("source")
        if not isinstance(source, Mapping):
            raise _PipelineFault("rejected", "admission", "candidate", "submission has no source identity")
        if not isinstance(normalized_policy.get("contract"), Mapping):
            raise _PipelineFault("blocked", "admission", "infrastructure",
                                 "trusted policy has no expected runtime/model contract")
        try:
            _validate_locked_contract(normalized_policy["contract"],
                                     require_runtime_identity=False)
        except EvaluationError as exc:
            raise _PipelineFault("blocked", "admission", "infrastructure", str(exc)) from exc
        try:
            stages = _policy_stages(normalized_policy)
        except EvaluationError as exc:
            raise _PipelineFault("blocked", "admission", "infrastructure", str(exc)) from exc
        if normalized_policy.get("qualification_enabled", False) is True:
            for key in ("corpus_id", "toolchain_id", "baseline_id"):
                if not _is_nonempty(normalized_policy.get(key)):
                    raise _PipelineFault("blocked", "admission", "infrastructure",
                                         f"qualification is enabled without trusted {key}")

        if cancelled():
            raise lifecycle.ProcessCancelled()
        archive_path = paths["archive"]
        source_dir = paths["source_dir"]
        assert isinstance(archive_path, Path) and isinstance(source_dir, Path)
        snapshot_sha, archive_evidence = _create_verified_archive(core, source, archive_path, cancelled)
        diagnostics_archive = archive_evidence
        receipt_evidence.append(_artifact_reference(
            core, archive_path, kind="source_archive", stage="admission",
            media_type="application/x-tar",
        ))
        if snapshot_sha != str(source.get("snapshot_sha256", "")).lower():
            raise _SourceInputError("source snapshot identity changed during archive validation")
        _safe_extract_archive(archive_path, source_dir, cancelled)
        actual_manifest = _extracted_source_manifest(core, source_dir)
        if actual_manifest != source.get("manifest") or core.canonical_hash(actual_manifest) != snapshot_sha:
            raise _SourceInputError("extracted source tree does not match its frozen source manifest")

        build_dir = paths["build_dir"]
        run_dir = paths["run_dir"]
        evidence_dir = paths["evidence_dir"]
        attempt_dir = paths["attempt_dir"]
        assert all(isinstance(value, Path) for value in (build_dir, run_dir, evidence_dir, attempt_dir))
        build_key, build_identity = _build_identity(
            core,
            normalized_submission,
            normalized_policy,
            stages,
            archive_evidence["sha256"],
            build_dir,
            source_dir,
            run_dir,
            str(paths["attempt_id"]),
            evidence_dir,
        )
        diagnostics_build = {"build_key": build_key, "identity": build_identity}

        contract_observer_count = sum(
            stage["criteria"].get("observes_contract") is True for stage in stages
        )
        candidate_binary: dict[str, Any] | None = None
        for stage in stages:
            current_stage = stage["name"]
            if cancelled():
                raise lifecycle.ProcessCancelled()
            report = _run_stage(
                core,
                stage,
                paths=paths,
                submission=normalized_submission,
                policy=normalized_policy,
                attempt_id=str(paths["attempt_id"]),
                build_key=build_key,
                cancelled=cancelled,
                receipt_evidence=receipt_evidence,
                stage_diagnostics=stage_diagnostics,
                gpu_admissions=gpu_admissions,
            )
            gates_seen.update(stage["criteria"]["gates"])
            if stage["criteria"].get("observes_contract") is True:
                observed = report.get("observed_contract")
                if not isinstance(observed, Mapping):
                    raise _PipelineFault("inconclusive", stage["name"], None,
                                         "contract-observation stage omitted observed_contract")
                try:
                    contract_verification = _verify_observed_contract(
                        core,
                        normalized_policy["contract"],
                        observed,
                        candidate_binary,
                        str(normalized_submission["source"]["git_commit"]),
                        stage["name"],
                    )
                except _PipelineFault as exc:
                    stage_diagnostics[-1]["contract_verification"] = exc.details
                    raise
                stage_diagnostics[-1]["contract_verification"] = contract_verification
                contract_seen = True
            elif stage["name"] == "build":
                candidate_binary = _candidate_binary_identity(core, stage, report, paths)

        if build_identity is not None:
            build_stage = next(stage for stage in stages if stage["name"] == "build")
            build_config = build_stage["criteria"]["build_identity"]
            dependencies_after = _verify_build_dependencies(core, build_config.get("dependencies", []))
            if dependencies_after != build_identity["build_dependencies"]:
                raise _PipelineFault("blocked", "build", "infrastructure",
                                     "trusted build dependency identity changed during evaluation")

        if contract_observer_count != 1:
            raise _PipelineFault("inconclusive", "full", None,
                                 "trusted policy did not provide exactly one contract-observation gate")
        required_gaps: list[str] = []
        missing_gates = sorted(QUALIFICATION_GATES - gates_seen)
        if missing_gates:
            required_gaps.append("missing semantic/task/holdout gates: " + ", ".join(missing_gates))
        owners_by_gate = [
            next((stage["name"] for stage in stages if gate in stage["criteria"]["gates"]), None)
            for gate in sorted(QUALIFICATION_GATES)
        ]
        if len(owners_by_gate) != len(set(owners_by_gate)):
            required_gaps.append("semantic, task, and holdout gates require distinct evidence stages")
        if not contract_seen:
            required_gaps.append("missing trusted observed-contract verification")
        if required_gaps:
            raise _PipelineFault("inconclusive", "full", None, "; ".join(required_gaps))
        if normalized_policy.get("qualification_enabled", False) is True:
            gaps = _qualification_gaps(
                normalized_policy, stages, gates_seen, contract_seen, gpu_admissions
            )
            if gaps:
                raise _PipelineFault("inconclusive", "full", None, "; ".join(gaps))
            verdict = "qualified"
            failure_class = None
            detail = "all trusted qualification stages and gates passed; this is not a promotion"
        else:
            verdict = "promising"
            failure_class = None
            detail = "all configured trusted stages passed; qualification is disabled by policy"
    except _PipelineFault as exc:
        verdict, current_stage, failure_class, detail = (
            exc.verdict, exc.stage, exc.failure_class, exc.reason
        )
    except _SourceInputError as exc:
        verdict, current_stage, failure_class, detail = "rejected", "admission", "candidate", str(exc)
    except lifecycle.ProcessCancelled:
        verdict, failure_class, detail = "blocked", None, "evaluation cancelled; active process groups were stopped"
    except (lifecycle.GpuAdmissionError, lifecycle.ProductionServiceActive) as exc:
        verdict, failure_class, detail = "blocked", "infrastructure", str(exc)
    except _EvidenceError as exc:
        verdict = "inconclusive" if exc.inconclusive else "regression"
        failure_class = None if exc.inconclusive else "candidate"
        detail = str(exc)
    except (EvaluationError, OSError, ValueError, TypeError, KeyError,
            subprocess.SubprocessError, json.JSONDecodeError) as exc:
        verdict, failure_class = "blocked", "infrastructure"
        detail = f"evaluation infrastructure failed: {type(exc).__name__}"

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": None if paths is None else paths.get("attempt_id"),
        "submission_id": normalized_submission.get("submission_id"),
        "policy_id": normalized_policy.get("policy_id"),
        "contract_id": normalized_policy.get("contract_id"),
        "source_archive": locals().get("diagnostics_archive"),
        "build": locals().get("diagnostics_build"),
        "stages": stage_diagnostics,
        "gpu_admissions": gpu_admissions,
        "verdict": verdict,
        "stage": current_stage,
        "failure_class": failure_class,
        "detail": detail,
        "candidate_performance_eligible": not any(
            stage["criteria"]["execution_mode"] == "live_endpoint" for stage in stages
        ),
    }
    if paths is not None:
        _write_diagnostics(core, paths, diagnostics, receipt_evidence)
    receipt = _basic_receipt(
        normalized_submission,
        normalized_policy,
        verdict=verdict,
        stage=current_stage,
        failure_class=failure_class,
        evidence=receipt_evidence,
    )
    return _persist_receipt(
        core,
        paths,
        receipt,
        validated=submission_validated and policy_validated,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True,
                        help="trusted frozen submission metadata JSON")
    parser.add_argument("--policy", type=Path, required=True,
                        help="trusted policy JSON stored outside candidate source")
    parser.add_argument("--run-root", type=Path, required=True,
                        help="owner-only directory for isolated attempts and receipt artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        core = _contracts_api()
        submission = core.read_json(args.submission)
        policy = core.read_json(args.policy)
        receipt = evaluate_submission(submission, policy, args.run_root)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        if receipt.get("verdict") == "qualified":
            return 0
        return 1
    except KeyboardInterrupt:
        print("evaluator.py: interrupted after process-group teardown", file=sys.stderr)
        return 130
    except (EvaluationError, OSError, ValueError, TypeError, subprocess.SubprocessError,
            json.JSONDecodeError) as exc:
        print(f"evaluator.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
