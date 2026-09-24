#!/usr/bin/env python3
"""Create owner-only, per-worker research checkouts on the FNN runtime host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Sequence


REMOTE_ALIAS = "fnn"
REMOTE_HOME = "/home/aramirezfamily"
REMOTE_PARENT = f"{REMOTE_HOME}/.local/share"
# Use the already-existing private parent as the unique-root container. This keeps every
# persistent create/cleanup write inside the one worker root.
RESEARCH_ROOT = REMOTE_PARENT
EXPECTED_REMOTE = {
    "hostname": "pop-os.localdomain",
    "uid": 1000,
    "home": REMOTE_HOME,
    "machine_id_sha256_prefix": "d060669c75703880",
}
EXPECTED_GPU_UUID = "GPU-008f1b73-105e-b50b-7469-ddca95c233f6"
PROTECTED_PATHS = (
    "/home/aramirezfamily/josh/ninfer-4090",
    "/home/aramirezfamily/josh/cinference-4090",
)
ARTIFACT = {
    "path": "/home/aramirezfamily/josh/cinference-4090/models/qwen3_8_27b.ablit-wxp.ninfer",
    "sha256": "b7c6693244cf7ab08554fff41c4f3c67695b6ec388c99eed6f12e0e71046fda8",
    "size_bytes": 18_210_704_128,
    "sha256_source": "E0001-LW04 read-only preflight; not rehashed during workspace setup",
}


def _norm(path: str) -> str:
    return posixpath.normpath(path)


def _is_within(path: str, parent: str) -> bool:
    path, parent = _norm(path), _norm(parent)
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def paths_overlap(left: str, right: str) -> bool:
    """Return true when either path is equal to or contains the other."""
    return _is_within(left, right) or _is_within(right, left)


def validate_research_path(path: str, root: str = RESEARCH_ROOT) -> str:
    """Normalize a target and reject protected production paths and out-of-root paths."""
    candidate = _norm(path)
    allowed_root = _norm(root)
    for protected in PROTECTED_PATHS:
        if paths_overlap(candidate, protected):
            raise ValueError(f"target overlaps a protected production path: {candidate}")
    if candidate == allowed_root or not _is_within(candidate, allowed_root):
        raise ValueError(f"target must be a child of the research root: {candidate}")
    return candidate


def validate_child_path(path: str, parent: str) -> str:
    candidate = _norm(path)
    allowed_parent = _norm(parent)
    if candidate == allowed_parent or not _is_within(candidate, allowed_parent):
        raise ValueError(f"target must be inside its assigned directory: {candidate}")
    for protected in PROTECTED_PATHS:
        if paths_overlap(candidate, protected):
            raise ValueError(f"target overlaps a protected production path: {candidate}")
    return candidate


def worker_slug(worker: str) -> str:
    value = worker.lower()
    if (
        not re.fullmatch(r"[a-z0-9](?:[a-z0-9._-]{0,47})", value)
        or ".." in value
        or value.endswith((".", "-"))
    ):
        raise ValueError("worker must be a short path-safe identifier")
    return value


def build_identity(worker: str, revision: str, patch_sha256: str | None) -> str:
    slug = worker_slug(worker)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("revision must be a full lowercase Git object ID")
    if patch_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", patch_sha256):
        raise ValueError("patch SHA-256 must be 64 lowercase hexadecimal characters")
    diff_tag = patch_sha256[:12] if patch_sha256 else "clean"
    identity = f"{slug}-{revision[:12]}-{diff_tag}"
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9._-]{0,47})-[0-9a-f]{12}-(?:clean|[0-9a-f]{12})", identity):
        raise ValueError("generated identity is invalid")
    return identity


def validate_identity(identity: str) -> str:
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9._-]{0,47})-[0-9a-f]{12}-(?:clean|[0-9a-f]{12})", identity):
        raise ValueError("identity must be a generated worker-revision-diff identifier")
    return identity


def _run(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {argv[0]}: {detail}")
    return result.stdout.strip()


def repository_root() -> Path:
    return Path(_run(["git", "rev-parse", "--show-toplevel"])).resolve()


def resolve_revision(repo: Path, revision: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision):
        raise ValueError("revision must be an explicit full Git object ID")
    resolved = _run(["git", "-C", str(repo), "rev-parse", "--verify", f"{revision}^{{commit}}"])
    if resolved.lower() != revision.lower():
        raise ValueError(f"revision resolved to a different commit: {resolved}")
    return resolved.lower()


def patch_info(patch_path: str | None) -> tuple[Path | None, str | None]:
    if patch_path is None:
        return None, None
    path = Path(patch_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("patch must be a regular file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def make_bundle(repo: Path, revision: str, temp_root: Path) -> tuple[Path, str, str]:
    """Export only the chosen commit and its ancestors without editing the source repo."""
    export_repo = temp_root / "export-repo"
    bundle_path = temp_root / "source.bundle"
    _run(["git", "clone", "--shared", "--quiet", "--no-checkout", str(repo), str(export_repo)])
    export_ref = f"refs/tags/ninfer-source-{revision[:12]}"
    _run(["git", "-C", str(export_repo), "update-ref", export_ref, revision])
    _run(["git", "-C", str(export_repo), "bundle", "create", str(bundle_path), export_ref])
    _run(["git", "bundle", "verify", str(bundle_path)])
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    return bundle_path, digest, export_ref


def remote_payload(action: str, **fields: Any) -> dict[str, Any]:
    return {
        "action": action,
        "expected_remote": EXPECTED_REMOTE,
        "expected_gpu_uuid": EXPECTED_GPU_UUID,
        "remote_home": REMOTE_HOME,
        "research_root": RESEARCH_ROOT,
        "protected_paths": list(PROTECTED_PATHS),
        "artifact": ARTIFACT,
        **fields,
    }


def call_remote(payload: dict[str, Any], *, capture_json: bool = True) -> Any:
    remote_code_path = Path(__file__).with_name("fnn_workspace_remote.py")
    remote_code = remote_code_path.read_text(encoding="utf-8")
    serialized = json.dumps(payload, separators=(",", ":"))
    program = remote_code + "\nmain(json.loads(" + repr(serialized) + "))\n"
    remote_command = "python3 -c " + shlex.quote(program)
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", REMOTE_ALIAS, remote_command]
    if not capture_json:
        return subprocess.run(command, check=False).returncode
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"remote {payload['action']} failed ({result.returncode}): {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("remote helper returned invalid metadata") from exc


def scp_to_remote(source: Path, destination: str) -> None:
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(source), f"{REMOTE_ALIAS}:{destination}"],
        check=True,
    )


def derive_paths(identity: str) -> dict[str, str]:
    identity = validate_identity(identity)
    root = validate_research_path(f"{RESEARCH_ROOT}/ninfer-{identity}")
    return {
        "root": root,
        "checkout": f"{root}/source",
        "build": f"{root}/build",
        "results": f"{root}/results",
        "input": f"{root}/input",
        "gpu_lock": RESEARCH_ROOT,
    }


def plan_command(args: argparse.Namespace) -> None:
    repo = repository_root()
    revision = resolve_revision(repo, args.revision)
    patch_path, patch_sha256 = patch_info(args.patch)
    identity = build_identity(args.worker, revision, patch_sha256)
    paths = derive_paths(identity)
    result = call_remote(remote_payload(
        "plan",
        worker=worker_slug(args.worker),
        identity=identity,
        revision=revision,
        patch_sha256=patch_sha256,
        patch_file_name=patch_path.name if patch_path else None,
        paths=paths,
    ))
    print(json.dumps(result, indent=2, sort_keys=True))


def create_command(args: argparse.Namespace) -> None:
    repo = repository_root()
    revision = resolve_revision(repo, args.revision)
    patch_path, patch_sha256 = patch_info(args.patch)
    identity = build_identity(args.worker, revision, patch_sha256)
    paths = derive_paths(identity)
    branch = args.branch
    if branch:
        _run(["git", "check-ref-format", "--branch", branch], cwd=repo)

    plan = call_remote(remote_payload(
        "plan",
        worker=worker_slug(args.worker),
        identity=identity,
        revision=revision,
        patch_sha256=patch_sha256,
        patch_file_name=patch_path.name if patch_path else None,
        paths=paths,
    ))
    if plan["paths"]["identity_root_exists"]:
        raise FileExistsError(f"research root already exists: {paths['root']}")

    with tempfile.TemporaryDirectory(prefix="ninfer-fnn-bundle-") as temp_name:
        temp_root = Path(temp_name)
        bundle, bundle_sha256, export_ref = make_bundle(repo, revision, temp_root)
        call_remote(remote_payload("prepare", identity=identity, paths=paths, revision=revision))
        remote_bundle = f"{paths['input']}/source.bundle"
        scp_to_remote(bundle, remote_bundle)
        remote_patch: str | None = None
        if patch_path:
            remote_patch = f"{paths['input']}/source.patch"
            scp_to_remote(patch_path, remote_patch)
        result = call_remote(remote_payload(
            "finalize",
            worker=worker_slug(args.worker),
            identity=identity,
            revision=revision,
            patch_sha256=patch_sha256,
            patch_file_name=patch_path.name if patch_path else None,
            patch_remote_path=remote_patch,
            bundle_remote_path=remote_bundle,
            bundle_sha256=bundle_sha256,
            export_ref=export_ref,
            branch=branch,
            paths=paths,
            source_repo_before=plan["selected_checkout"],
        ))
    print(json.dumps(result, indent=2, sort_keys=True))


def lock_plan_command(args: argparse.Namespace) -> None:
    identity = validate_identity(args.identity)
    result = call_remote(remote_payload("lock_plan", identity=identity, paths=derive_paths(identity)))
    print(json.dumps(result, indent=2, sort_keys=True))


def gpu_lock_command(args: argparse.Namespace) -> None:
    identity = validate_identity(args.identity)
    argv = list(args.command)
    if argv and argv[0] == "--":
        argv.pop(0)
    if not argv:
        raise ValueError("provide a command after --")
    result = call_remote(remote_payload("gpu_lock", identity=identity, paths=derive_paths(identity), argv=argv), capture_json=False)
    raise SystemExit(result)


def cleanup_command(args: argparse.Namespace) -> None:
    identity = validate_identity(args.identity)
    result = call_remote(remote_payload(
        "cleanup",
        identity=identity,
        paths=derive_paths(identity),
        apply=bool(args.apply),
    ))
    print(json.dumps(result, indent=2, sort_keys=True))


def record_build_command(args: argparse.Namespace) -> None:
    identity = validate_identity(args.identity)
    paths = derive_paths(identity)
    binary_path = validate_child_path(args.binary, paths["build"])
    result = call_remote(remote_payload("record_build", identity=identity, paths=paths, binary_path=binary_path))
    print(json.dumps(result, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="subcommand", required=True)

    for name, handler in (("plan", plan_command), ("create", create_command)):
        sub = commands.add_parser(name)
        sub.add_argument("--worker", required=True, help="stable worker/epoch key, such as E0002-LW04")
        sub.add_argument("--revision", required=True, help="full Git commit ID to export")
        sub.add_argument("--patch", help="explicit binary patch file to apply in the isolated checkout")
        if name == "create":
            sub.add_argument("--branch", help="optional branch name local to this isolated clone")
        sub.set_defaults(handler=handler)

    sub = commands.add_parser("lock-plan", help="show the shared non-polling GPU lock and worker paths")
    sub.add_argument("--identity", required=True)
    sub.set_defaults(handler=lock_plan_command)

    sub = commands.add_parser("gpu-lock", help="run one command under the shared blocking GPU lock")
    sub.add_argument("--identity", required=True)
    sub.add_argument("command", nargs=argparse.REMAINDER)
    sub.set_defaults(handler=gpu_lock_command)

    sub = commands.add_parser("cleanup", help="inspect or remove only this generated research root")
    sub.add_argument("--identity", required=True)
    sub.add_argument("--apply", action="store_true", help="remove the exact generated root after validation")
    sub.set_defaults(handler=cleanup_command)

    sub = commands.add_parser("record-build", help="record an existing isolated build and binary without building")
    sub.add_argument("--identity", required=True)
    sub.add_argument("--binary", required=True, help="absolute binary path inside this identity's build directory")
    sub.set_defaults(handler=record_build_command)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.handler(args)
        return 0
    except (FileExistsError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"fnn_workspace: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
