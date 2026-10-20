#!/usr/bin/env python3
"""Freeze a worktree into an immutable, FIFO evaluator submission."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from typing import Any
import uuid


SCHEMA_VERSION = 1
OVERRIDE_TYPES: dict[str, str] = {
    "max_context": "positive_int",
    "kv_capacity": "capacity",
    "max_concurrency": "concurrency",
    "max_pending_requests": "positive_int",
    "pending_timeout_ms": "positive_int",
    "prefill_chunk": "positive_int",
    "kv_dtype": "kv_dtype",
    "spec": "spec",
    "draft_tokens": "draft_tokens",
    "lm_head_draft": "bool",
    "preserve_thinking": "bool",
    "default_max_tokens": "positive_int",
    "seed": "positive_int",
    "no_cuda_graph": "bool",
    "no_prefix_reuse": "bool",
}
OVERRIDE_RANGES = {
    "max_context": (1, 1_048_576),
    "max_pending_requests": (1, 64),
    "pending_timeout_ms": (1, 3_600_000),
    "prefill_chunk": (1, 16_384),
    "default_max_tokens": (1, 8192),
    "seed": (0, 2**63 - 1),
}
EXCLUDED_ROOT_PARTS = {
    ".git",
    ".venv",
    "build",
    "build-sm89",
    "models",
    "node_modules",
    "out",
    "profiles",
}
EXCLUDED_ANYWHERE_PARTS = {".git", ".venv", "__pycache__", "node_modules"}


class SubmissionError(RuntimeError):
    pass


def _run_git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        details = getattr(exc, "stderr", b"")
        raise SubmissionError(f"git {' '.join(args)} failed: {details.decode(errors='replace')}") from exc


def _included_path(rel: PurePosixPath) -> bool:
    if rel.is_absolute() or ".." in rel.parts:
        return False
    if not rel.parts or rel.parts[0] in EXCLUDED_ROOT_PARTS:
        return False
    return not any(part in EXCLUDED_ANYWHERE_PARTS for part in rel.parts)


def _source_entries(root: Path) -> list[tuple[PurePosixPath, Path]]:
    raw = _run_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    entries: dict[str, tuple[PurePosixPath, Path]] = {}
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        name = os.fsdecode(encoded)
        rel = PurePosixPath(name)
        if not _included_path(rel):
            continue
        source = root.joinpath(*rel.parts)
        if not source.exists() and not source.is_symlink():
            # Deleted tracked paths are represented by the worktree patch identity.
            continue
        if source.is_symlink():
            target = source.resolve(strict=False)
            try:
                target.relative_to(root.resolve())
            except ValueError as exc:
                raise SubmissionError(f"source symlink escapes worktree: {name}") from exc
        elif not source.is_file():
            # Git submodules are directories in the checkout. Include their checked-out files
            # while still omitting their private .git metadata.
            if source.is_dir():
                for current, dirs, files in os.walk(source, followlinks=False):
                    current_path = Path(current)
                    dirs[:] = sorted(
                        d for d in dirs
                        if _included_path(
                            PurePosixPath((current_path / d).relative_to(root).as_posix())
                        )
                    )
                    for filename in sorted(files):
                        child = current_path / filename
                        child_rel = PurePosixPath(child.relative_to(root).as_posix())
                        if _included_path(child_rel):
                            entries[child_rel.as_posix()] = (child_rel, child)
            continue
        entries[rel.as_posix()] = (rel, source)
    return [entries[name] for name in sorted(entries)]


def _hash_snapshot(entries: list[tuple[PurePosixPath, Path]]) -> str:
    digest = hashlib.sha256()
    for rel, source in entries:
        rel_bytes = rel.as_posix().encode("utf-8", errors="surrogateescape")
        mode = source.lstat().st_mode
        digest.update(len(rel_bytes).to_bytes(4, "big"))
        digest.update(rel_bytes)
        if stat.S_ISLNK(mode):
            digest.update(b"L")
            digest.update(os.fsencode(os.readlink(source)))
        else:
            digest.update(b"F")
            digest.update((0o755 if mode & 0o111 else 0o644).to_bytes(4, "big"))
            with source.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
    return digest.hexdigest()


def _worktree_patch_identity(root: Path, entries: list[tuple[PurePosixPath, Path]]) -> str:
    tracked = _run_git(root, "diff", "--binary", "HEAD", "--")
    digest = hashlib.sha256()
    digest.update(tracked)
    tracked_paths = set(
        os.fsdecode(part)
        for part in _run_git(root, "ls-files", "-z", "--cached").split(b"\0")
        if part
    )
    for rel, source in entries:
        if rel.as_posix() in tracked_paths:
            continue
        rel_bytes = rel.as_posix().encode("utf-8", errors="surrogateescape")
        digest.update(len(rel_bytes).to_bytes(4, "big"))
        digest.update(rel_bytes)
        if source.is_symlink():
            digest.update(os.fsencode(os.readlink(source)))
        else:
            with source.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
    return digest.hexdigest()


def _copy_entries(root: Path, destination: Path,
                  entries: list[tuple[PurePosixPath, Path]]) -> None:
    for rel, source in entries:
        target = destination.joinpath(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copyfile(source, target)
            target.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)


def _write_deterministic_tar(source_root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source_root.rglob("*"), key=lambda item: item.relative_to(source_root).as_posix()):
            rel = path.relative_to(source_root).as_posix()
            info = archive.gettarinfo(str(path), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise argparse.ArgumentTypeError("override must be NAME=VALUE")
    name, raw = item.split("=", 1)
    kind = OVERRIDE_TYPES.get(name)
    if kind is None:
        allowed = ", ".join(sorted(OVERRIDE_TYPES))
        raise argparse.ArgumentTypeError(f"override {name!r} is not allowlisted (allowed: {allowed})")
    try:
        if kind in {"positive_int", "concurrency", "draft_tokens"}:
            value: Any = int(raw)
            lower, upper = OVERRIDE_RANGES.get(name, (1, 2**31 - 1))
            if value < lower or value > upper:
                raise ValueError
            if kind == "concurrency" and value > 8:
                raise ValueError
            if kind == "draft_tokens" and value > 15:
                raise ValueError
        elif kind == "capacity":
            value = raw if raw == "auto" else int(raw)
            if isinstance(value, int) and not 1 <= value <= 1_048_576:
                raise ValueError
        elif kind == "bool":
            lowered = raw.lower()
            if lowered not in {"true", "false"}:
                raise ValueError
            value = lowered == "true"
        elif kind == "kv_dtype":
            if raw not in {"bf16", "int8", "fp8", "nvfp4", "k8v4"}:
                raise ValueError
            value = raw
        else:
            if raw not in {"none", "mtp", "dflash", "dflash2"}:
                raise ValueError
            value = raw
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid value for override {name!r}: {raw!r}") from exc
    return name, value


def validate_overrides(overrides: dict[str, Any]) -> None:
    if not isinstance(overrides, dict):
        raise SubmissionError("overrides must be an object")
    for name, value in overrides.items():
        try:
            parsed_name, parsed_value = parse_override(f"{name}={value}")
        except argparse.ArgumentTypeError as exc:
            raise SubmissionError(str(exc)) from exc
        if parsed_name != name or type(parsed_value) is not type(value) or parsed_value != value:
            raise SubmissionError(f"invalid override value for {name!r}")


def freeze_submission(
    *, source: Path, package_dir: Path, role: str, baseline_id: str,
    artifact_path: str, artifact_sha256: str, runtime_id: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = source.resolve()
    _run_git(root, "rev-parse", "--show-toplevel")
    if role not in {"baseline", "control", "candidate"}:
        raise SubmissionError(f"invalid submission role: {role!r}")
    if not baseline_id.strip() or not runtime_id.strip():
        raise SubmissionError("baseline_id and runtime_id must be nonempty")
    if not Path(artifact_path).is_absolute():
        raise SubmissionError("artifact_path must be absolute")
    if len(artifact_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in artifact_sha256):
        raise SubmissionError("artifact_sha256 must be a 64-digit hexadecimal SHA-256")
    overrides = overrides or {}
    validate_overrides(overrides)
    commit = _run_git(root, "rev-parse", "HEAD").decode().strip()
    entries = _source_entries(root)
    if not entries:
        raise SubmissionError(f"worktree contains no source files: {root}")
    snapshot_sha256 = _hash_snapshot(entries)
    patch_sha256 = _worktree_patch_identity(root, entries)
    submission_id = uuid.uuid4().hex
    package_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    package_dir.chmod(0o700)
    package = package_dir / submission_id
    package.mkdir(parents=True, mode=0o700)
    source_tree = package / "source"
    source_tree.mkdir(mode=0o700)
    _copy_entries(root, source_tree, entries)
    archive_path = package / "source.tar"
    _write_deterministic_tar(source_tree, archive_path)
    shutil.rmtree(source_tree)
    archive_sha256 = _file_sha256(archive_path)
    source_info = {
        "git_commit": commit,
        "worktree_patch_sha256": patch_sha256,
        "snapshot_sha256": snapshot_sha256,
        "archive_sha256": archive_sha256,
        "file_count": len(entries),
        "archive": archive_path.name,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": submission_id,
        "role": role,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_id": baseline_id,
        "source": source_info,
        "artifact": {"path": artifact_path, "sha256": artifact_sha256},
        "runtime_id": runtime_id,
        "overrides": overrides,
    }
    _write_json(package / "manifest.json", manifest, mode=0o600)
    archive_path.chmod(0o400)
    return {"package_dir": str(package), "manifest_path": str(package / "manifest.json"),
            "archive_path": str(archive_path), "manifest": manifest}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def enqueue_submission(*, queue_root: Path, manifest_path: Path,
                       archive_path: Path) -> dict[str, Any]:
    """Atomically append one frozen submission to the filesystem FIFO."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SubmissionError("unsupported submission schema_version")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise SubmissionError("submission is missing source identity")
    archive_sha = _file_sha256(archive_path)
    if archive_sha != source.get("archive_sha256"):
        raise SubmissionError("frozen source archive hash does not match manifest")
    overrides = manifest.get("overrides", {})
    validate_overrides(overrides)

    queue_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    queue_root.chmod(0o700)
    for name in ("pending", "running", "completed", "failed", "snapshots"):
        directory = queue_root / name
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    lock_path = queue_root / ".queue.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        sequence_path = queue_root / ".sequence"
        try:
            sequence = int(sequence_path.read_text(encoding="ascii"))
        except FileNotFoundError:
            sequence = 0
        sequence += 1
        snapshot_path = queue_root / "snapshots" / f"{source['snapshot_sha256']}.tar"
        if snapshot_path.exists():
            if _file_sha256(snapshot_path) != archive_sha:
                raise SubmissionError("snapshot identity collision in queue")
        else:
            tmp_snapshot = snapshot_path.with_suffix(".tar.tmp")
            shutil.copyfile(archive_path, tmp_snapshot)
            tmp_snapshot.chmod(0o400)
            os.replace(tmp_snapshot, snapshot_path)
        manifest["source"]["archive"] = str(snapshot_path.relative_to(queue_root))
        manifest["queue"] = {
            "sequence": sequence,
            "enqueued_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        pending_path = queue_root / "pending" / f"{sequence:012d}-{manifest['submission_id']}.json"
        sequence_tmp = sequence_path.with_suffix(".tmp")
        sequence_tmp.write_text(str(sequence), encoding="ascii")
        sequence_tmp.chmod(0o600)
        os.replace(sequence_tmp, sequence_path)
        _write_json(pending_path, manifest)
        return {"sequence": sequence, "submission_id": manifest["submission_id"],
                "manifest_path": str(pending_path), "snapshot_path": str(snapshot_path)}
    finally:
        os.close(lock_fd)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="capture an immutable worktree package")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--package-dir", type=Path, required=True)
    freeze.add_argument("--role", choices=("baseline", "control", "candidate"), required=True)
    freeze.add_argument("--baseline-id", required=True)
    freeze.add_argument("--artifact-path", required=True)
    freeze.add_argument("--artifact-sha256", required=True)
    freeze.add_argument("--runtime-id", required=True)
    freeze.add_argument("--override", action="append", type=parse_override, default=[])

    enqueue = subparsers.add_parser("enqueue", help="append a frozen package to a FIFO queue")
    enqueue.add_argument("--queue-root", type=Path, required=True)
    enqueue.add_argument("--manifest", type=Path, required=True)
    enqueue.add_argument("--archive", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "freeze":
            overrides: dict[str, Any] = {}
            for name, value in args.override:
                if name in overrides:
                    raise SubmissionError(f"override supplied more than once: {name}")
                overrides[name] = value
            result = freeze_submission(
                source=args.source,
                package_dir=args.package_dir,
                role=args.role,
                baseline_id=args.baseline_id,
                artifact_path=args.artifact_path,
                artifact_sha256=args.artifact_sha256,
                runtime_id=args.runtime_id,
                overrides=overrides,
            )
        else:
            result = enqueue_submission(
                queue_root=args.queue_root,
                manifest_path=args.manifest,
                archive_path=args.archive,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, SubmissionError) as exc:
        print(f"submit.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
