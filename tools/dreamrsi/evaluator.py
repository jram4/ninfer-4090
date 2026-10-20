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

The expected contract must contain SHA-256 identities for the model artifact,
chat template, and tokenizer, a nonempty runtime identity, ``kv_dtype=k8v4``,
and context capacity 188416 (direct or nested lock fields are accepted). A
build policy pins CMake 4.4 and CUDA 13, and its admission report must observe
those actual host toolchain versions before candidate build results are
classified.

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
    ) -> None:
        self.verdict = verdict
        self.stage = stage
        self.failure_class = failure_class
        self.reason = reason
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


def _contract_lock_observations(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Extract required lock values without accepting a public model label as identity."""
    found: dict[str, Any] = {}

    def visit(value: Any, parents: tuple[str, ...] = ()) -> None:
        if not isinstance(value, Mapping):
            return
        for raw_key, child in value.items():
            key = str(raw_key).lower().replace("-", "_")
            path = parents + (key,)
            joined = ".".join(path)
            if isinstance(child, str):
                if any(identity in joined for identity in ("artifact", "model")) and (
                    "sha256" in joined or "hash" in joined
                ):
                    found.setdefault("artifact_sha256", child.lower())
                if "template" in joined and ("sha256" in joined or "hash" in joined):
                    found.setdefault("template_sha256", child.lower())
                if "tokenizer" in joined and ("sha256" in joined or "hash" in joined):
                    found.setdefault("tokenizer_sha256", child.lower())
                if "runtime" in joined and key in {"id", "runtime_id", "name"}:
                    found.setdefault("runtime_id", child)
                if ("kv" in joined
                        and ("dtype" in joined or key in {"cache", "kv_cache"})
                        and child.lower() == "k8v4"):
                    found.setdefault("kv_dtype", child.lower())
            if type(child) is int and child == 188416 and "context" in joined:
                found.setdefault("context_capacity", child)
            visit(child, path)

    visit(contract)
    return found


def _validate_locked_contract(contract: Mapping[str, Any]) -> None:
    observed = _contract_lock_observations(contract)
    missing = [
        name for name in (
            "artifact_sha256", "template_sha256", "tokenizer_sha256", "runtime_id",
            "kv_dtype", "context_capacity",
        )
        if name not in observed
    ]
    for name in ("artifact_sha256", "template_sha256", "tokenizer_sha256"):
        if name in observed and not _HEX_SHA256.fullmatch(str(observed[name])):
            missing.append(f"valid {name}")
    if "runtime_id" in observed and not _is_nonempty(observed["runtime_id"]):
        missing.append("nonempty runtime_id")
    if missing:
        raise EvaluationError(
            "trusted contract must lock artifact, runtime, template, tokenizer, k8v4, and context 188416; "
            "missing " + ", ".join(missing)
        )


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
    detail: str,
    evidence: list[dict[str, Any]] | None = None,
    paths: Mapping[str, Path | str] | None = None,
    candidate_performance_eligible: bool | None = None,
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
        "evidence": evidence or [],
        "created_at": _utc_now(),
    }
    outcome = {
        "kind": "evaluator_outcome",
        "stage": stage,
        "detail": detail,
        "candidate_performance_eligible": (
            candidate_performance_eligible
            if candidate_performance_eligible is not None
            else not any(
                item.get("execution_mode") == "live_endpoint"
                for item in (evidence or [])
                if item.get("kind") == "stage_result"
            )
        ),
    }
    if paths is not None:
        outcome["attempt_id"] = paths.get("attempt_id")
        outcome["attempt_dir"] = str(paths.get("attempt_dir"))
    receipt["evidence"].append(outcome)
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


def _policy_stages(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_stages = policy.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise EvaluationError("trusted policy must define a nonempty ordered stages list")
    stages: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_evidence_paths: set[str] = set()
    last_rank = -1
    contract_observers = 0
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, Mapping):
            raise EvaluationError("each policy stage must be a JSON object")
        stage = dict(raw_stage)
        name = stage.get("name")
        if name not in STAGE_ORDER:
            raise EvaluationError("policy stage name is not in the supported staged pipeline")
        rank = STAGE_ORDER.index(name)
        if name in seen_names or rank <= last_rank:
            raise EvaluationError("policy stages must be unique and in admission/build/correctness/smoke/screen/full order")
        seen_names.add(name)
        last_rank = rank
        argv = stage.get("argv")
        if not isinstance(argv, list) or not argv or any(
            not isinstance(item, str) or not item or "\0" in item for item in argv
        ):
            raise EvaluationError(f"{name} stage argv must be a nonempty string array")
        timeout = stage.get("timeout_seconds")
        if type(timeout) not in {int, float} or not 0 < timeout <= 86_400:
            raise EvaluationError(f"{name} stage timeout_seconds must be within (0, 86400]")
        if type(stage.get("requires_gpu", False)) is not bool:
            raise EvaluationError(f"{name} requires_gpu must be a boolean")
        mode = stage.get("execution_mode", "isolated_candidate")
        if mode not in {"isolated_candidate", "live_endpoint"}:
            raise EvaluationError(f"{name} has an unsupported execution_mode")
        if mode == "live_endpoint" and not _is_nonempty(stage.get("label")):
            raise EvaluationError("live_endpoint stages require an explicit nonempty label")
        if type(stage.get("observes_contract", False)) is not bool:
            raise EvaluationError("observes_contract must be a boolean")
        if stage.get("observes_contract", False):
            contract_observers += 1
        cwd = stage.get("cwd", "run")
        if cwd not in {"source", "build", "run"}:
            raise EvaluationError(f"{name} cwd must be source, build, or run")
        stage["cwd"] = cwd
        stage["execution_mode"] = mode
        stage["timeout_seconds"] = float(timeout)
        evidence = stage.get("evidence")
        if not isinstance(evidence, Mapping):
            raise EvaluationError(f"{name} stage must define trusted evidence criteria")
        evidence = dict(evidence)
        evidence_path = evidence.get("path")
        if not isinstance(evidence_path, str):
            raise EvaluationError(f"{name} evidence.path must be a relative path")
        relative = PurePosixPath(evidence_path)
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise EvaluationError(f"{name} evidence.path must stay inside its evidence directory")
        normalized_path = relative.as_posix()
        if normalized_path in seen_evidence_paths:
            raise EvaluationError("each stage must write to a distinct evidence path")
        seen_evidence_paths.add(normalized_path)
        required = evidence.get("required_fields")
        equals = evidence.get("equals")
        minimum = evidence.get("minimum", {})
        maximum = evidence.get("maximum", {})
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise EvaluationError(f"{name} evidence.required_fields must be a string array")
        if "raw_artifacts" not in required:
            raise EvaluationError(f"{name} evidence.required_fields must include raw_artifacts")
        if not isinstance(equals, Mapping) or equals.get("status") != "pass":
            raise EvaluationError(f"{name} evidence.equals must require status=pass")
        if not isinstance(minimum, Mapping) or not isinstance(maximum, Mapping):
            raise EvaluationError(f"{name} evidence minimum/maximum criteria must be objects")
        if any(not isinstance(key, str) for key in (*equals.keys(), *minimum.keys(), *maximum.keys())):
            raise EvaluationError(f"{name} evidence criteria paths must be strings")
        if any(type(value) not in {int, float} for value in (*minimum.values(), *maximum.values())):
            raise EvaluationError(f"{name} evidence numeric bounds must be numbers")
        if any(
            key in maximum and minimum[key] > maximum[key]
            for key in minimum
        ):
            raise EvaluationError(f"{name} evidence minimum exceeds its maximum")
        if (stage.get("gates") or name in {"screen", "full"}) and not (
            any(key != "status" for key in equals) or minimum or maximum
        ):
            raise EvaluationError(f"{name} quality/performance gate needs criteria beyond status=pass")
        if not equals and not minimum and not maximum:
            raise EvaluationError(f"{name} evidence criteria cannot be empty")
        evidence["path"] = normalized_path
        stage["evidence"] = evidence
        gates = stage.get("gates", [])
        if not isinstance(gates, list) or any(
            not isinstance(gate, str) or gate not in QUALIFICATION_GATES for gate in gates
        ) or len(gates) != len(set(gates)):
            raise EvaluationError(f"{name} gates must be unique semantic/task/holdout labels")
        stage["gates"] = list(gates)
        stage["argv"] = list(argv)
        stages.append(stage)
    if stages[0]["name"] != "admission":
        raise EvaluationError("the first policy stage must be admission")
    if contract_observers > 1:
        raise EvaluationError("exactly one stage may observe the runtime contract")
    return stages


def _is_nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _verify_build_dependencies(core: Any, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    dependencies = policy.get("build_dependencies", [])
    if not isinstance(dependencies, list):
        raise EvaluationError("build_dependencies must be a list")
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
    build_stage = next((stage for stage in stages if stage["name"] == "build"), None)
    command_template = [] if build_stage is None else build_stage["argv"]
    dependencies = _verify_build_dependencies(core, policy)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_archive_sha256": source_sha,
        "source_snapshot_sha256": submission["source"]["snapshot_sha256"],
        "toolchain_id": policy.get("toolchain_id"),
        "toolchain": policy.get("toolchain", {}),
        "build_flags": policy.get("build_flags", {}),
        "build_dependencies": dependencies,
        "build_command_sha256": core.canonical_hash(command_template),
        "build_environment_sha256": core.canonical_hash(
            {} if build_stage is None else build_stage.get("env", {})
        ),
    }
    build_key = core.canonical_hash(identity)
    if not isinstance(build_key, str) or not _HEX_SHA256.fullmatch(build_key):
        raise EvaluationError("shared canonical_hash returned an invalid build key")
    return build_key.lower(), identity


def _expand_argv(stage: Mapping[str, Any], paths: Mapping[str, Path | str],
                 submission: Mapping[str, Any], policy: Mapping[str, Any],
                 build_key: str, raw_dir: Path) -> list[str]:
    evidence_config = stage["evidence"]
    evidence_dir = paths["evidence_dir"]
    assert isinstance(evidence_dir, Path)
    evidence_path = evidence_dir.joinpath(*PurePosixPath(evidence_config["path"]).parts)
    values = {
        "source_dir": str(paths["source_dir"]),
        "build_dir": str(paths["build_dir"]),
        "run_dir": str(paths["run_dir"]),
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
    evidence_path = evidence_dir.joinpath(*PurePosixPath(stage["evidence"]["path"]).parts)
    environment = lifecycle.sanitized_environment()
    environment.update({
        "NINFER_SOURCE_DIR": str(paths["source_dir"]),
        "NINFER_BUILD_DIR": str(paths["build_dir"]),
        "NINFER_RUN_DIR": str(paths["run_dir"]),
        "NINFER_EVIDENCE_DIR": str(evidence_dir),
        "NINFER_STAGE_EVIDENCE_PATH": str(evidence_path),
        "NINFER_STAGE_RAW_DIR": str(raw_dir),
        "NINFER_STAGE": str(stage["name"]),
        "NINFER_ATTEMPT_ID": attempt_id,
        "NINFER_SUBMISSION_ID": str(submission["submission_id"]),
        "NINFER_POLICY_ID": str(policy["policy_id"]),
        "NINFER_CONTRACT_ID": str(policy["contract_id"]),
        "NINFER_SOURCE_SNAPSHOT_SHA256": str(submission["source"]["snapshot_sha256"]),
        "NINFER_BUILD_KEY": build_key,
    })
    configured = stage.get("env", {})
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
    path = evidence_dir.joinpath(*PurePosixPath(stage["evidence"]["path"]).parts)
    cursor = evidence_dir
    for part in PurePosixPath(stage["evidence"]["path"]).parts[:-1]:
        cursor = cursor / part
        if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
            raise EvaluationError("stage evidence parent is not a safe directory")
        cursor.mkdir(mode=0o700, exist_ok=True)
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


def _snapshot_raw_artifacts(core: Any, report: Mapping[str, Any], *, stage_name: str,
                            run_dir: Path, artifacts_dir: Path,
                            evidence: list[dict[str, Any]]) -> None:
    references = report.get("raw_artifacts")
    if not isinstance(references, list) or not references or len(references) > 128:
        raise _EvidenceError("stage omitted its bounded raw-artifact references", inconclusive=True)
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise _EvidenceError("raw artifact reference must be a JSON object", inconclusive=True)
        source = _safe_relative_file(run_dir, reference.get("path"))
        try:
            expected_hash = _sha256_text(reference.get("sha256"), "raw artifact sha256")
            actual_hash = core.sha256_file(source).lower()
        except (EvaluationError, OSError) as exc:
            raise _EvidenceError("raw artifact hash could not be verified", inconclusive=True) from exc
        if actual_hash != expected_hash:
            raise _EvidenceError("raw artifact SHA-256 differs from its stage report", inconclusive=True)
        snapshot = _immutable_snapshot(
            core, source, artifacts_dir / f"{stage_name}.raw-{index:03d}"
        )
        if snapshot["sha256"] != expected_hash:
            raise _EvidenceError("raw artifact changed while it was snapshotted", inconclusive=True)
        if snapshot["size_bytes"] == 0:
            raise _EvidenceError("raw evidence artifact is empty", inconclusive=True)
        evidence.append({"kind": "raw_stage_artifact", "stage": stage_name, **snapshot})


def _path_value(value: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
    current: Any = value
    for part in dotted_path.split("."):
        if not part or not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _validate_evidence_report(core: Any, *, report_path: Path, artifact_path: Path,
                              evidence_root: Path,
                              run_dir: Path, artifacts_dir: Path,
                              stage: Mapping[str, Any], submission: Mapping[str, Any],
                              policy: Mapping[str, Any], attempt_id: str,
                              build_key: str,
                              evidence: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
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
    _snapshot_raw_artifacts(
        core,
        report,
        stage_name=str(stage["name"]),
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        evidence=evidence,
    )
    if stage["name"] == "build" and report.get("build_key") != build_key:
        raise _EvidenceError("build evidence does not bind the complete build identity", inconclusive=True)
    if stage["name"] == "admission" and _is_nonempty(policy.get("toolchain_id")):
        if report.get("toolchain_id") != policy.get("toolchain_id"):
            raise _EvidenceError("host admission evidence does not match the trusted toolchain_id",
                                 failure_class="infrastructure")
        expected_toolchain = policy.get("toolchain")
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

    criteria = stage["evidence"]
    for path in criteria["required_fields"]:
        exists, _ = _path_value(report, path)
        if not exists:
            raise _EvidenceError(f"stage evidence is missing required field {path}", inconclusive=True)
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
    return report, artifact


def _stage_cwd(stage: Mapping[str, Any], paths: Mapping[str, Path | str]) -> Path:
    directory = paths[f"{stage['cwd']}_dir"]
    assert isinstance(directory, Path)
    return directory


def _record_partial_log(core: Any, log_path: Path, artifacts_dir: Path,
                        stage_name: str, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    destination = artifacts_dir / f"{stage_name}.log"
    try:
        artifact = (
            {"path": str(destination), "sha256": core.sha256_file(destination).lower(),
             "size_bytes": destination.stat().st_size}
            if destination.is_file()
            else _immutable_snapshot(core, log_path, destination)
        )
    except (OSError, EvaluationError):
        return None
    evidence.append({"kind": "stage_log", "stage": stage_name, **artifact})
    return artifact


def _run_stage(core: Any, stage: Mapping[str, Any], *, paths: Mapping[str, Path | str],
               submission: Mapping[str, Any], policy: Mapping[str, Any], attempt_id: str,
               build_key: str, cancelled: Cancelled,
               evidence: list[dict[str, Any]], gpu_admissions: list[dict[str, Any]]) -> dict[str, Any]:
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
    gpu_required = stage.get("requires_gpu", False)
    if gpu_required:
        lock_path = policy.get("gpu_lock_path")
        units = policy.get("production_units")
        if not isinstance(lock_path, str) or not Path(lock_path).is_absolute():
            raise _PipelineFault("blocked", stage["name"], "infrastructure",
                                 "GPU stage requires an absolute trusted gpu_lock_path")
        if not isinstance(units, list) or any(not isinstance(unit, str) or not unit for unit in units):
            raise _PipelineFault("blocked", stage["name"], "infrastructure",
                                 "GPU stage requires trusted production_units")
        target_uuid = policy.get("target_gpu_uuid")
        if not _is_nonempty(target_uuid):
            raise _PipelineFault("blocked", stage["name"], "infrastructure",
                                 "GPU stage requires a trusted target_gpu_uuid")
        admission_context = lifecycle.gpu_admission(
            Path(lock_path),
            units,
            target_uuid=target_uuid,
            lease_wait_seconds=float(policy.get("gpu_lease_wait_seconds", 30.0)),
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
        log_artifact = _record_partial_log(core, log_path, artifacts_dir,
                                           stage["name"], evidence)
        evidence.append({
            "kind": "cancelled_stage",
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
        log_artifact = _record_partial_log(core, log_path, artifacts_dir,
                                           stage["name"], evidence)
        evidence.append({
            "kind": "timed_out_stage",
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
        raise _PipelineFault("blocked", stage["name"], "infrastructure", str(exc)) from exc
    except lifecycle.ProcessCleanupError as exc:
        log_artifact = _record_partial_log(core, log_path, artifacts_dir,
                                           stage["name"], evidence)
        evidence.append({
            "kind": "unclean_stage",
            "stage": stage["name"],
            "command_sha256": core.canonical_hash(argv),
            "process_cleanup": exc.cleanup,
            "log": log_artifact,
        })
        raise _PipelineFault("blocked", stage["name"], "infrastructure",
                             f"stage process-group teardown failed: {exc.cleanup}") from exc
    except (OSError, subprocess.SubprocessError, EvaluationError) as exc:
        raise _PipelineFault("blocked", stage["name"], "infrastructure",
                             f"trusted {stage['name']} stage could not run: {type(exc).__name__}") from exc

    assert stage_result is not None
    if admission_record is not None:
        evidence.append({"kind": "gpu_admission", **admission_record})
    log_artifact: dict[str, Any] | None = None
    if log_path.is_file():
        log_artifact = _immutable_snapshot(core, log_path, artifacts_dir / f"{stage['name']}.log")
        evidence.append({"kind": "stage_log", "stage": stage["name"], **log_artifact})
    if stage_result["returncode"] != 0:
        failure_class = "infrastructure" if stage["name"] == "admission" else "candidate"
        verdict = "blocked" if failure_class == "infrastructure" else (
            "rejected" if stage["name"] == "build" else "regression"
        )
        evidence.append({
            "kind": "stage_result",
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
    report_artifact_path = artifacts_dir / f"{stage['name']}.evidence.json"
    try:
        report, report_artifact = _validate_evidence_report(
            core,
            report_path=report_path,
            artifact_path=report_artifact_path,
            evidence_root=evidence_dir,
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            stage=stage,
            submission=submission,
            policy=policy,
            attempt_id=attempt_id,
            build_key=build_key,
            evidence=evidence,
        )
    except _EvidenceError as exc:
        if report_artifact_path.is_file():
            evidence.append({
                "kind": "stage_evidence",
                "stage": stage["name"],
                "path": str(report_artifact_path),
                "sha256": core.sha256_file(report_artifact_path).lower(),
            })
        evidence.append({
            "kind": "stage_result",
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
    evidence_item = {
        "kind": "stage_result",
        "stage": stage["name"],
        "command_sha256": core.canonical_hash(argv),
        "executable": argv[0],
        "argument_count": len(argv),
        "returncode": stage_result["returncode"],
        "elapsed_seconds": stage_result["elapsed_seconds"],
        "execution_mode": stage["execution_mode"],
        "label": stage.get("label"),
        "requires_gpu": gpu_required,
        "gates": list(stage["gates"]),
        "log": log_artifact,
        "report": report_artifact,
        "cleanup": stage_result["cleanup"],
    }
    evidence.append(evidence_item)
    if stage.get("observes_contract") is True:
        observed = report.get("observed_contract")
        if not isinstance(observed, Mapping):
            raise _PipelineFault("inconclusive", stage["name"], None,
                                 "contract-observation stage omitted observed_contract")
        mismatch = core.compare_contract(policy.get("contract"), dict(observed))
        if not isinstance(mismatch, list):
            raise _PipelineFault("inconclusive", stage["name"], None,
                                 "shared contract comparison returned an invalid result")
        evidence_item["contract_verification"] = {"matched": not mismatch, "mismatches": mismatch}
        if mismatch:
            raise _PipelineFault("regression", stage["name"], "candidate",
                                 "observed runtime/model contract differs from the trusted policy")
        try:
            _validate_locked_contract(observed)
        except EvaluationError as exc:
            raise _PipelineFault("inconclusive", stage["name"], None,
                                 "observed contract omitted an independent lock: " + str(exc)) from exc
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
        owners = [stage["name"] for stage in stages if gate in stage["gates"]]
        if len(owners) != 1:
            gaps.append(f"{gate} gate must have exactly one trusted evidence stage")
    owners_by_gate = [
        next((stage["name"] for stage in stages if gate in stage["gates"]), None)
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
    if any(stage["execution_mode"] == "live_endpoint" for stage in stages):
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
    evidence: list[dict[str, Any]] = []
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
            failure_class="infrastructure", detail=f"run directory setup failed: {type(exc).__name__}",
            candidate_performance_eligible=not any(
                isinstance(stage, Mapping) and stage.get("execution_mode") == "live_endpoint"
                for stage in normalized_policy.get("stages", [])
            ) if isinstance(normalized_policy.get("stages", []), list) else True,
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
                                 f"trusted policy validation failed: {type(exc).__name__}") from exc

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
            _validate_locked_contract(normalized_policy["contract"])
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
        if any(stage["name"] == "build" for stage in stages):
            toolchain = normalized_policy.get("toolchain")
            if not _is_nonempty(normalized_policy.get("toolchain_id")):
                raise _PipelineFault("blocked", "admission", "infrastructure",
                                     "build policy has no explicit trusted toolchain_id")
            if not isinstance(toolchain, Mapping) or any(
                not _is_nonempty(toolchain.get(key)) for key in ("cmake", "cuda")
            ):
                raise _PipelineFault("blocked", "admission", "infrastructure",
                                     "build policy must explicitly identify its CMake and CUDA toolchains")
            if (_version_tokens(toolchain["cmake"])[:2] != (4, 4)
                    or _version_tokens(toolchain["cuda"])[:1] != (13,)):
                raise _PipelineFault("blocked", "admission", "infrastructure",
                                     "NInfer bench requires explicitly pinned CMake 4.4 and CUDA 13")

        if cancelled():
            raise lifecycle.ProcessCancelled()
        archive_path = paths["archive"]
        source_dir = paths["source_dir"]
        assert isinstance(archive_path, Path) and isinstance(source_dir, Path)
        snapshot_sha, archive_evidence = _create_verified_archive(core, source, archive_path, cancelled)
        evidence.append(archive_evidence)
        if snapshot_sha != str(source.get("snapshot_sha256", "")).lower():
            raise _SourceInputError("source snapshot identity changed during archive validation")
        _safe_extract_archive(archive_path, source_dir, cancelled)

        build_dir = paths["build_dir"]
        run_dir = paths["run_dir"]
        evidence_dir = paths["evidence_dir"]
        assert isinstance(build_dir, Path) and isinstance(run_dir, Path) and isinstance(evidence_dir, Path)
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
        evidence.append({"kind": "build_identity", "build_key": build_key, "identity": build_identity})

        contract_observer_count = sum(stage.get("observes_contract") is True for stage in stages)
        if contract_observer_count != 1:
            detail = "policy must name exactly one observed-contract stage"
        for stage in stages:
            current_stage = stage["name"]
            if cancelled():
                raise lifecycle.ProcessCancelled()
            _run_stage(
                core,
                stage,
                paths=paths,
                submission=normalized_submission,
                policy=normalized_policy,
                attempt_id=str(paths["attempt_id"]),
                build_key=build_key,
                cancelled=cancelled,
                evidence=evidence,
                gpu_admissions=gpu_admissions,
            )
            gates_seen.update(stage["gates"])
            if stage.get("observes_contract") is True:
                contract_seen = True

        if build_identity is not None:
            dependencies_after = _verify_build_dependencies(core, normalized_policy)
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
            next((stage["name"] for stage in stages if gate in stage["gates"]), None)
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

    receipt = _basic_receipt(
        normalized_submission,
        normalized_policy,
        verdict=verdict,
        stage=current_stage,
        failure_class=failure_class,
        detail=detail,
        evidence=evidence,
        paths=paths,
        candidate_performance_eligible=not any(
            stage.get("execution_mode") == "live_endpoint" for stage in stages
        ),
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
