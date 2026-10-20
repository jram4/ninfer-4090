"""Freeze a precise Git/worktree source snapshot for the local evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

from .contracts import (
    SCHEMA_VERSION,
    canonical_hash,
    sha256_file,
    submission_identity,
    validate_submission,
    write_json,
)


class SubmissionError(ValueError):
    """Source cannot be safely and unambiguously frozen."""


# These are intentionally anchored at the checkout root.  In particular,
# excluding the repository's root ``models`` directory must not exclude
# ``src/models`` or any other source path with the same basename.
_EXCLUDED_ROOTS = {
    ".git", ".slim", ".local", ".codex", ".venv", "venv", "build",
    "out", "profiles", "node_modules", "models", "research", "private",
    "secrets", "credentials", "temp", "tmp", "Dream-RSI", "Dream-RSI.zip",
    ".obsidian", ".omo",
}
_EXCLUDED_ANYWHERE = {".git", "__pycache__", "node_modules", ".venv", "venv"}
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _git(root: Path, *args: str, check: bool = True) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        raise SubmissionError(
            f"git {' '.join(args)} failed: {detail.decode('utf-8', errors='replace').strip()}"
        ) from exc
    return completed.stdout


def _git_succeeds(root: Path, *args: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise SubmissionError(f"git {' '.join(args)} failed: {exc}") from exc
    return result.returncode == 0


def _included(path: PurePosixPath) -> bool:
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return False
    if path.parts[0] in _EXCLUDED_ROOTS:
        return False
    if any(part in _EXCLUDED_ANYWHERE for part in path.parts):
        return False
    # Keep common local credentials/private key formats out even if not ignored
    # by the repository.  Matching is by exact filename/extension, never by a
    # broad directory basename such as "models".
    for part in path.parts:
        lower = part.lower()
        if lower == ".env" or lower.startswith(".env.") or lower in {
            "id_rsa", "id_ed25519", "credentials.json", "credentials.yaml",
            "secrets.json", "token.json", "settings.local.json",
        }:
            return False
        if lower.endswith((".pem", ".key", ".p12", ".pfx")):
            return False
    return True


def _valid_path(raw: bytes) -> str:
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError("source paths must be valid UTF-8") from exc
    if "\\" in name or "\x00" in name or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name):
        raise SubmissionError(f"unsafe source path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in name.split("/")):
        raise SubmissionError(f"unsafe source path: {name!r}")
    if path.as_posix() != name or (path.parts and ":" in path.parts[0]):
        raise SubmissionError(f"non-canonical source path: {name!r}")
    return name


def _safe_link(path: str, target_bytes: bytes) -> str:
    try:
        target = target_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"symlink target for {path!r} must be valid UTF-8") from exc
    if not target or target.startswith("/") or "\\" in target or "\x00" in target:
        raise SubmissionError(f"unsafe symlink target for {path!r}")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in target):
        raise SubmissionError(f"symlink target contains a control character: {path!r}")
    if ":" in target.split("/", 1)[0]:
        raise SubmissionError(f"non-portable symlink target for {path!r}")
    resolved = list(PurePosixPath(path).parent.parts)
    for component in target.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            if not resolved:
                raise SubmissionError(f"symlink escapes source root: {path!r}")
            resolved.pop()
        else:
            resolved.append(component)
    if not resolved:
        raise SubmissionError(f"symlink resolves to source root: {path!r}")
    return target


def _file_entry(path: str, content: bytes, mode: int) -> dict[str, Any]:
    if content.startswith(_LFS_HEADER):
        raise SubmissionError(f"Git LFS pointer is unsupported: {path}")
    return {
        "path": path,
        "type": "file",
        "mode": 0o100755 if mode & 0o111 else 0o100644,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _manifest_from_commit(root: Path, commit: str) -> tuple[list[dict[str, Any]], dict[str, bytes | str]]:
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", commit)
    manifest: list[dict[str, Any]] = []
    contents: dict[str, bytes | str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, path_raw = record.split(b"\t", 1)
            mode_raw, object_type, object_id = header.split(b" ", 2)
        except ValueError as exc:
            raise SubmissionError("git returned a malformed tree entry") from exc
        path = _valid_path(path_raw)
        rel = PurePosixPath(path)
        if not _included(rel):
            continue
        mode = int(mode_raw, 8)
        if mode == 0o160000 or object_type == b"commit":
            raise SubmissionError(f"Git submodule is unsupported: {path}")
        if object_type != b"blob" or mode not in {0o100644, 0o100755, 0o120000}:
            raise SubmissionError(f"unsupported Git source entry: {path}")
        blob = _git(root, "cat-file", "blob", object_id.decode("ascii"))
        if mode == 0o120000:
            target = _safe_link(path, blob)
            manifest.append({"path": path, "type": "symlink", "mode": mode, "target": target})
            contents[path] = target
        else:
            manifest.append(_file_entry(path, blob, mode))
            contents[path] = blob
    manifest.sort(key=lambda entry: entry["path"])
    _validate_manifest_tree(manifest)
    return manifest, contents


def _worktree_manifest(root: Path) -> tuple[list[dict[str, Any]], dict[str, bytes | str]]:
    raw = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    names = sorted({_valid_path(item) for item in raw.split(b"\0") if item})
    selected_names = [name for name in names if _included(PurePosixPath(name))]
    _reject_submodules(root, set(selected_names))
    _reject_lfs_filters(root, selected_names)
    manifest: list[dict[str, Any]] = []
    contents: dict[str, bytes | str] = {}
    root_resolved = root.resolve()
    for name in names:
        rel = PurePosixPath(name)
        if not _included(rel):
            continue
        source = root.joinpath(*rel.parts)
        try:
            info = source.lstat()
        except FileNotFoundError:
            # Deleted tracked files are absent from the frozen candidate tree.
            continue
        if stat.S_ISLNK(info.st_mode):
            target = _safe_link(name, os.fsencode(os.readlink(source)))
            resolved = (source.parent / target).resolve(strict=False)
            try:
                resolved.relative_to(root_resolved)
            except ValueError as exc:
                raise SubmissionError(f"source symlink escapes worktree: {name}") from exc
            manifest.append({"path": name, "type": "symlink", "mode": 0o120000, "target": target})
            contents[name] = target
        elif stat.S_ISREG(info.st_mode):
            data = source.read_bytes()
            manifest.append(_file_entry(name, data, info.st_mode))
            contents[name] = data
        elif stat.S_ISDIR(info.st_mode):
            # A tracked path changing into a directory is inconsistent with the
            # selected file list and should not be recursively broadened.
            if (source / ".git").exists():
                raise SubmissionError(f"nested Git repository/submodule is unsupported: {name}")
            raise SubmissionError(f"source path changed into a directory: {name}")
        else:
            raise SubmissionError(f"unsupported source file type: {name}")
    manifest.sort(key=lambda entry: entry["path"])
    _validate_manifest_tree(manifest)
    return manifest, contents


def _reject_submodules(root: Path, selected: set[str]) -> None:
    raw = _git(root, "ls-files", "--stage", "-z")
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, path_raw = record.split(b"\t", 1)
            mode = int(header.split(b" ", 1)[0], 8)
            name = _valid_path(path_raw)
        except (ValueError, IndexError) as exc:
            raise SubmissionError("git returned a malformed index entry") from exc
        if mode == 0o160000 and name in selected:
            raise SubmissionError(f"Git submodule is unsupported: {name}")


def _reject_lfs_filters(root: Path, names: list[str]) -> None:
    for start in range(0, len(names), 512):
        batch = names[start : start + 512]
        if not batch:
            continue
        raw = _git(root, "check-attr", "-z", "filter", "--", *batch)
        fields = raw.split(b"\0")
        for offset in range(0, len(fields) - 2, 3):
            path_raw, attribute, value = fields[offset : offset + 3]
            if attribute == b"filter" and value == b"lfs":
                raise SubmissionError(f"Git LFS source is unsupported: {_valid_path(path_raw)}")


def _validate_manifest_tree(manifest: list[dict[str, Any]]) -> None:
    links = {entry["path"] for entry in manifest if entry["type"] == "symlink"}
    paths = {entry["path"] for entry in manifest}
    for name in paths:
        parts = PurePosixPath(name).parts
        for count in range(1, len(parts)):
            ancestor = "/".join(parts[:count])
            if ancestor in links:
                raise SubmissionError(f"source entry {name!r} descends through symlink {ancestor!r}")
    for entry in manifest:
        if entry["type"] != "symlink":
            continue
        resolved = list(PurePosixPath(entry["path"]).parent.parts)
        for component in entry["target"].split("/"):
            if component in {"", "."}:
                continue
            if component == "..":
                if not resolved:
                    raise SubmissionError(f"symlink escapes source root: {entry['path']!r}")
                resolved.pop()
            else:
                resolved.append(component)
            if "/".join(resolved) in links:
                raise SubmissionError(f"symlink target traverses another symlink: {entry['path']!r}")


def _resolve_commit(root: Path, ref: str) -> str:
    if not isinstance(ref, str) or not ref or ref.startswith("-") or "\x00" in ref:
        raise SubmissionError("ref must be a nonempty Git revision")
    commit = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").decode("ascii").strip()
    if len(commit) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in commit):
        raise SubmissionError("resolved ref is not a full Git commit ID")
    return commit


def _first_parent(root: Path, commit: str) -> str:
    parents = _git(root, "show", "-s", "--format=%P", commit).decode("ascii").strip().split()
    return parents[0] if parents else commit


def _write_archive(path: Path, manifest: list[dict[str, Any]], contents: dict[str, bytes | str]) -> None:
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for entry in manifest:
            info = tarfile.TarInfo(entry["path"])
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.pax_headers = {}
            if entry["type"] == "symlink":
                info.type = tarfile.SYMTYPE
                info.mode = 0o777
                info.linkname = str(contents[entry["path"]])
                info.size = 0
                archive.addfile(info)
            else:
                info.type = tarfile.REGTYPE
                info.mode = 0o755 if entry["mode"] == 0o100755 else 0o644
                data = contents[entry["path"]]
                assert isinstance(data, bytes)
                info.size = len(data)
                archive.addfile(info, fileobj=io.BytesIO(data))
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _dirty(root: Path) -> bool:
    raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    fields = raw.split(b"\0")
    index = 0
    while index < len(fields):
        record = fields[index]
        index += 1
        if not record:
            continue
        if len(record) < 4:
            raise SubmissionError("git returned a malformed worktree status record")
        name = _valid_path(record[3:])
        if _included(PurePosixPath(name)):
            return True
        # Porcelain v1 -z emits the old name in a second field for rename and
        # copy records; treat either eligible endpoint as selected source.
        if record[:2].strip() in {b"R", b"C"} and index < len(fields):
            old_name = _valid_path(fields[index])
            index += 1
            if _included(PurePosixPath(old_name)):
                return True
    return False


def freeze_submission(
    source: Path,
    package_dir: Path,
    *,
    epoch_id: str,
    worker_id: str,
    hypothesis_id: str,
    contract_id: str,
    policy_id: str,
    ref: str = "HEAD",
    base_ref: str | None = None,
    include_dirty: bool = False,
) -> dict[str, Any]:
    """Create a content-addressed, frozen source archive and valid submission.

    By default any tracked or non-ignored untracked checkout changes are a hard
    error.  ``include_dirty=True`` explicitly selects the current non-ignored
    worktree sources; ignored data and the anchored private/generated paths
    remain excluded.  A dirty snapshot is never commit-backed/promotable.
    """
    if type(include_dirty) is not bool:
        raise SubmissionError("include_dirty must be boolean")
    for name, value in (("epoch_id", epoch_id), ("worker_id", worker_id), ("hypothesis_id", hypothesis_id)):
        if not isinstance(value, str) or not value or value != value.strip():
            raise SubmissionError(f"{name} must be a nonempty trimmed string")
    for name, value in (("contract_id", contract_id), ("policy_id", policy_id)):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise SubmissionError(f"{name} must be a lowercase 64-digit SHA-256")
    root = Path(source).resolve(strict=True)
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if root != top:
        raise SubmissionError("source must be the Git worktree root")
    resolved_ref = _resolve_commit(root, ref)
    is_dirty = _dirty(root)
    if is_dirty and not include_dirty:
        raise SubmissionError("source worktree is dirty; pass include_dirty=True to freeze selected changes")
    head = _resolve_commit(root, "HEAD")
    if is_dirty and resolved_ref != head:
        raise SubmissionError("dirty inclusion requires ref to resolve to current HEAD")
    if is_dirty:
        manifest, contents = _worktree_manifest(root)
        candidate_commit = None
    else:
        manifest, contents = _manifest_from_commit(root, resolved_ref)
        candidate_commit = resolved_ref
    if base_ref is None:
        base_commit = resolved_ref if is_dirty else _first_parent(root, resolved_ref)
    else:
        base_commit = _resolve_commit(root, base_ref)
        if not _git_succeeds(root, "merge-base", "--is-ancestor", base_commit, resolved_ref):
            raise SubmissionError("base_ref must resolve to an ancestor of ref")
    if not manifest:
        raise SubmissionError("selected source tree contains no source files")

    package_root = Path(package_dir).expanduser().resolve()
    try:
        package_root.relative_to(root)
    except ValueError:
        pass
    else:
        raise SubmissionError("package_dir must not be inside the source worktree")
    if package_root == root:
        raise SubmissionError("package_dir must not be the source worktree")

    snapshot_sha256 = canonical_hash(manifest)
    created_at = datetime.now(timezone.utc).isoformat()
    identity_source = {
        "archive_path": str(package_root / "pending" / "source.tar"),
        "archive_sha256": "0" * 64,
        "snapshot_sha256": snapshot_sha256,
        "git_commit": resolved_ref,
        "dirty": is_dirty,
        "manifest": manifest,
    }
    # The package path and archive digest are transport details.  Submission
    # identity is derived only once archive bytes are finalized below.
    package_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=".freeze-", dir=package_root))
    try:
        archive_path = stage / "source.tar"
        _write_archive(archive_path, manifest, contents)
        archive_digest = sha256_file(archive_path)
        source_record = dict(identity_source)
        source_record["archive_path"] = "pending"
        source_record["archive_sha256"] = archive_digest
        submission: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "submission_id": "0" * 64,
            "epoch_id": epoch_id,
            "worker_id": worker_id,
            "hypothesis_id": hypothesis_id,
            "base_commit": base_commit,
            "candidate_commit": candidate_commit,
            "source": source_record,
            "contract_id": contract_id,
            "policy_id": policy_id,
            "created_at": created_at,
            "idempotency_key": canonical_hash({
                "epoch_id": epoch_id,
                "worker_id": worker_id,
                "hypothesis_id": hypothesis_id,
                "source_snapshot_sha256": snapshot_sha256,
                "contract_id": contract_id,
                "policy_id": policy_id,
            }),
        }
        submission["submission_id"] = canonical_hash(submission_identity(submission))
        final_dir = package_root / submission["submission_id"]
        final_archive = final_dir / "source.tar"
        submission["source"]["archive_path"] = str(final_archive)
        # Local archive location and creation time are deliberately excluded
        # from the source-independent submission identity.
        validated = validate_submission(submission)
        write_json(stage / "submission.json", validated)
        write_json(stage / "source-manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "snapshot_sha256": snapshot_sha256,
            "entries": manifest,
        })
        archive_fd = os.open(archive_path, os.O_RDONLY)
        try:
            os.fchmod(archive_fd, 0o400)
            os.fsync(archive_fd)
        finally:
            os.close(archive_fd)
        if final_dir.exists():
            raise SubmissionError(f"frozen submission already exists: {final_dir}")
        os.rename(stage, final_dir)
        parent_fd = os.open(package_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return validated
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--epoch-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--contract-id", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--base-ref")
    parser.add_argument("--include-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(freeze_submission(
            args.source,
            args.package_dir,
            epoch_id=args.epoch_id,
            worker_id=args.worker_id,
            hypothesis_id=args.hypothesis_id,
            contract_id=args.contract_id,
            policy_id=args.policy_id,
            ref=args.ref,
            base_ref=args.base_ref,
            include_dirty=args.include_dirty,
        ), sort_keys=True))
        return 0
    except (OSError, SubmissionError, ValueError) as exc:
        print(f"submit.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
