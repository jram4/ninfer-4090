"""Remote half of fnn_workspace.py; called with a JSON request embedded by the local wrapper."""

from __future__ import annotations

import datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
from typing import Any


def run(argv: list[str], *, timeout: int = 15, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {detail}")
    return result


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(path: str) -> str:
    return os.path.normpath(path)


def within(path: str, parent: str) -> bool:
    path, parent = normalized(path), normalized(parent)
    return path == parent or path.startswith(parent.rstrip(os.sep) + os.sep)


def paths_overlap(left: str, right: str) -> bool:
    return within(left, right) or within(right, left)


def reject_symlink_components(path: str) -> None:
    current = Path("/")
    for part in Path(path).parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"path contains a symlink component: {current}")


def guard_path(path: str, root: str, protected: list[str]) -> str:
    candidate = normalized(path)
    allowed_root = normalized(root)
    for protected_path in protected:
        if paths_overlap(candidate, protected_path):
            raise RuntimeError(f"target overlaps protected production path: {candidate}")
    if candidate == allowed_root or not within(candidate, allowed_root):
        raise RuntimeError(f"target is outside the unique research root: {candidate}")
    reject_symlink_components(candidate)
    resolved = str(Path(candidate).resolve(strict=False))
    if resolved != candidate:
        raise RuntimeError(f"target resolves through a different path: {candidate} -> {resolved}")
    return candidate


def verify_remote_identity(request: dict[str, Any]) -> dict[str, Any]:
    expected = request["expected_remote"]
    hostname = run(["hostname", "-f"]).stdout.strip()
    uid = os.getuid()
    home = pwd.getpwuid(uid).pw_dir
    try:
        # Match `sha256sum /etc/machine-id`, including the file's trailing newline.
        machine_id = Path("/etc/machine-id").read_bytes()
    except OSError as exc:
        raise RuntimeError("remote machine identity is unavailable") from exc
    machine_hash = hashlib.sha256(machine_id).hexdigest()[:16]
    observed = {
        "hostname": hostname,
        "uid": uid,
        "home": home,
        "machine_id_sha256_prefix": machine_hash,
    }
    if observed != expected:
        raise RuntimeError(f"remote identity mismatch: expected {expected}, observed {observed}")
    return observed


def require_private_dir(path: str, uid: int, *, create: bool = False) -> None:
    directory = Path(path)
    reject_symlink_components(str(directory))
    if not directory.exists():
        if not create:
            raise RuntimeError(f"required directory does not exist: {directory}")
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
    info = directory.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError(f"directory is not owner-only and user-owned: {directory}")


def git_observation(path: str) -> dict[str, Any]:
    head = run(["git", "-C", path, "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "-C", path, "branch", "--show-current"]).stdout.strip()
    status = run(["git", "--no-optional-locks", "-C", path, "status", "--porcelain=v1"]).stdout.splitlines()
    return {"path": path, "head": head, "branch": branch, "status_entry_count": len(status)}


def verify_directory_flock(path: str) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Contention proves that the filesystem supports the lock.
                return
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    return
                raise
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RuntimeError(f"shared directory flock is unavailable: {exc}") from exc


def base_metadata(request: dict[str, Any]) -> dict[str, Any]:
    verify_remote_identity(request)
    artifact = request["artifact"]
    reject_symlink_components(artifact["path"])
    artifact_stat = os.lstat(artifact["path"])
    if artifact_stat.st_size != artifact["size_bytes"]:
        raise RuntimeError("artifact size differs from the recorded E0001-LW04 identity")
    if not stat.S_ISREG(artifact_stat.st_mode) or artifact_stat.st_uid != os.getuid():
        raise RuntimeError("artifact is not a user-owned regular file")

    selected_path = "/home/aramirezfamily/josh/ninfer-4090"
    reject_symlink_components(selected_path)
    selected = git_observation(selected_path)
    revision = request.get("revision")
    if revision:
        source_object = run(
            ["git", "-C", selected_path, "cat-file", "-e", f"{revision}^{{commit}}"],
            check=False,
        )
        selected["contains_requested_revision"] = source_object.returncode == 0

    gpu_line = run([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader",
    ]).stdout.strip()
    gpu_fields = [part.strip() for part in gpu_line.split(",")]
    if len(gpu_fields) != 5 or gpu_fields[1] != request["expected_gpu_uuid"]:
        raise RuntimeError(f"GPU identity mismatch: {gpu_line}")

    nvcc_output = run(["/usr/local/cuda-13/bin/nvcc", "--version"]).stdout
    match = re.search(r"release\s+([^,\s]+)", nvcc_output)
    if not match:
        raise RuntimeError("could not identify the CUDA toolkit version")
    compiler_version = run(["/usr/bin/g++-13", "--version"]).stdout.splitlines()[0]
    cmake_version = run(["cmake", "--version"]).stdout.splitlines()[0]
    python_version = run(["python3", "--version"]).stdout.strip()
    git_version = run(["git", "--version"]).stdout.strip()
    parent = request["research_root"]
    require_private_dir(parent, os.getuid())
    verify_directory_flock(parent)
    paths = request.get("paths", {})
    if paths:
        for key in ("root", "checkout", "build", "results", "input"):
            guard_path(paths[key], parent, request["protected_paths"])

    return {
        "remote": verify_remote_identity(request),
        "selected_checkout": selected,
        "toolchain": {
            "git": git_version,
            "cmake": cmake_version,
            "cuda_nvcc_release": match.group(1),
            "cuda_nvcc_binary": "/usr/local/cuda-13/bin/nvcc",
            "cxx": compiler_version,
            "python": python_version,
            "build_state": "not-configured",
        },
        "gpu": {
            "name": gpu_fields[0],
            "uuid": gpu_fields[1],
            "driver_version": gpu_fields[2],
            "memory_total": gpu_fields[3],
            "compute_capability": gpu_fields[4],
        },
        "artifact": {
            **artifact,
            "observed_size_bytes": artifact_stat.st_size,
            "mode": oct(stat.S_IMODE(artifact_stat.st_mode)),
            "owner_uid": artifact_stat.st_uid,
        },
    }


def ensure_identity_root(request: dict[str, Any]) -> tuple[str, dict[str, str]]:
    identity = request["identity"]
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9._-]{0,47})-[0-9a-f]{12}-(?:clean|[0-9a-f]{12})", identity):
        raise RuntimeError("invalid generated identity")
    require_private_dir(request["research_root"], os.getuid())
    paths = request["paths"]
    root = guard_path(paths["root"], request["research_root"], request["protected_paths"])
    expected_root = f"{request['research_root']}/ninfer-{identity}"
    if root != expected_root:
        raise RuntimeError("identity root does not match worker identity")
    for key, leaf in (("checkout", "source"), ("build", "build"), ("results", "results"), ("input", "input")):
        if paths[key] != f"{root}/{leaf}":
            raise RuntimeError(f"unexpected {key} path")
        guard_path(paths[key], request["research_root"], request["protected_paths"])
    return root, paths


def plan(request: dict[str, Any]) -> None:
    root, paths = ensure_identity_root(request)
    metadata = base_metadata(request)
    metadata.update({
        "worker": request["worker"],
        "identity": request["identity"],
        "revision": request["revision"],
        "patch_sha256": request["patch_sha256"],
        "paths": {**paths, "identity_root_exists": Path(root).exists()},
        "writes_performed": False,
    })
    print(json.dumps(metadata, sort_keys=True))


def prepare(request: dict[str, Any]) -> None:
    observed = verify_remote_identity(request)
    require_private_dir(request["research_root"], os.getuid())
    root, paths = ensure_identity_root(request)
    if Path(root).exists():
        raise FileExistsError(f"refusing to reuse an existing research root: {root}")
    os.umask(0o077)
    Path(root).mkdir(mode=0o700)
    for path in (paths["checkout"], paths["build"], paths["results"], paths["input"]):
        Path(path).mkdir(mode=0o700)
        os.chmod(path, 0o700)
    marker = Path(root) / ".ninfer-research-identity"
    marker.write_text(request["identity"] + "\n", encoding="utf-8")
    os.chmod(marker, 0o600)
    for path in (root, paths["checkout"], paths["build"], paths["results"], paths["input"]):
        require_private_dir(path, os.getuid())
    print(json.dumps({"identity": request["identity"], "root": root, "remote": observed, "prepared": True}))


def checked_file(
    path: str,
    expected_name: str,
    expected_sha256: str,
    uid: int,
    input_directory: str,
    protected_paths: list[str],
) -> dict[str, Any]:
    path = guard_path(path, input_directory, protected_paths)
    if path != os.path.join(normalized(input_directory), expected_name):
        raise RuntimeError(f"unexpected input file path: {path}")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError(f"invalid expected SHA-256 for input: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"input cannot be opened safely: {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid:
            raise RuntimeError(f"input is not a user-owned regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        sha256 = digest.hexdigest()
        if sha256 != expected_sha256:
            raise RuntimeError(f"input checksum mismatch: {path}")
        os.fchmod(descriptor, 0o600)
        return {"path": path, "sha256": sha256, "size_bytes": info.st_size}
    finally:
        os.close(descriptor)


def require_identity_marker(root: str, identity: str) -> str:
    marker = str(Path(root) / ".ninfer-research-identity")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(marker, flags)
    except OSError as exc:
        raise RuntimeError(f"research identity marker cannot be opened safely: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError("research identity marker is not a private user-owned regular file")
        marker_contents = os.read(descriptor, 128).decode("utf-8")
    finally:
        os.close(descriptor)
    if marker_contents != identity + "\n":
        raise RuntimeError("research identity marker mismatch")
    return marker


def restrict_tree_permissions(path: str) -> None:
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        os.chmod(current_path, stat.S_IMODE(current_path.lstat().st_mode) & ~0o077)
        for name in directories + files:
            entry = current_path / name
            info = entry.lstat()
            if stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode):
                os.chmod(entry, stat.S_IMODE(info.st_mode) & ~0o077)


def finalize(request: dict[str, Any]) -> None:
    remote = verify_remote_identity(request)
    root, paths = ensure_identity_root(request)
    require_private_dir(root, os.getuid())
    for path in (paths["checkout"], paths["build"], paths["results"], paths["input"]):
        require_private_dir(path, os.getuid())
    require_identity_marker(root, request["identity"])

    os.umask(0o077)
    bundle = checked_file(
        request["bundle_remote_path"],
        "source.bundle",
        request["bundle_sha256"],
        os.getuid(),
        paths["input"],
        request["protected_paths"],
    )
    source_path = paths["checkout"]
    run(["git", "init", "--quiet", source_path])
    run(["git", "-C", source_path, "fetch", "--quiet", request["bundle_remote_path"], request["export_ref"]])
    run(["git", "-C", source_path, "checkout", "--quiet", "--detach", request["revision"]])
    if run(["git", "-C", source_path, "rev-parse", "HEAD"]).stdout.strip() != request["revision"]:
        raise RuntimeError("isolated checkout revision does not match the requested revision")

    patch_meta = None
    if request["patch_remote_path"]:
        patch_meta = checked_file(
            request["patch_remote_path"],
            "source.patch",
            request["patch_sha256"],
            os.getuid(),
            paths["input"],
            request["protected_paths"],
        )
        run(["git", "-C", source_path, "apply", "--check", "--", request["patch_remote_path"]])
        run(["git", "-C", source_path, "apply", "--", request["patch_remote_path"]])
    elif request["patch_sha256"] is not None:
        raise RuntimeError("patch checksum was supplied but no patch file was transferred")

    if request["branch"]:
        run(["git", "-C", source_path, "switch", "--quiet", "--create", request["branch"]])
    restrict_tree_permissions(source_path)
    branch_name = run(["git", "-C", source_path, "branch", "--show-current"]).stdout.strip() or None
    checkout_revision = run(["git", "-C", source_path, "rev-parse", "HEAD"]).stdout.strip()
    source_status_count = len(run(["git", "-C", source_path, "status", "--porcelain=v1"]).stdout.splitlines())

    metadata = base_metadata(request)
    selected_checkout_before = request["source_repo_before"]
    selected_checkout_after = metadata["selected_checkout"]
    if selected_checkout_after != selected_checkout_before:
        raise RuntimeError("selected production checkout metadata changed during setup")
    metadata.update({
        "schema_version": 1,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "worker": request["worker"],
        "identity": request["identity"],
        "source": {
            "revision": request["revision"],
            "checkout_revision": checkout_revision,
            "branch": branch_name,
            "explicit_patch_file": request["patch_file_name"],
            "patch_sha256": request["patch_sha256"],
            "patch_applied": patch_meta is not None,
            "patch_file": patch_meta,
            "bundle": {**bundle, "export_ref": request["export_ref"]},
            "working_tree_status_entry_count": source_status_count,
        },
        "paths": paths,
        "permissions": {
            "root_and_work_areas_mode": "0700",
            "source_tree_group_other_bits_removed": True,
            "provenance_mode": "0600",
        },
        "build": {
            "state": "not-configured",
            "directory": paths["build"],
            "configuration": None,
            "command": None,
        },
        "selected_checkout_before": selected_checkout_before,
        "selected_checkout_after": selected_checkout_after,
        "selected_checkout_unchanged": selected_checkout_before == selected_checkout_after,
        "gpu_serialization": {
            "lock_path": request["research_root"],
            "mechanism": "blocking flock exclusive lock on the preexisting owner-only .local/share directory; no lock file created",
            "workload_run": False,
        },
    })
    results_path = Path(paths["results"])
    manifest = results_path / "provenance.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest, 0o600)
    require_private_dir(paths["results"], os.getuid())
    print(json.dumps({
        "identity": request["identity"],
        "root": root,
        "provenance": str(manifest),
        "remote": remote,
        "metadata": metadata,
        "selected_checkout_after": metadata["selected_checkout"],
    }, sort_keys=True))


def lock_plan(request: dict[str, Any]) -> None:
    verify_remote_identity(request)
    root, paths = ensure_identity_root(request)
    require_private_dir(root, os.getuid())
    require_identity_marker(root, request["identity"])
    print(json.dumps({
        "identity": request["identity"],
        "checkout": paths["checkout"],
        "build": paths["build"],
        "results": paths["results"],
        "lock_path": request["research_root"],
        "lock_mode": "blocking exclusive flock; no polling and no lock file",
        "lock_owner_only": True,
    }))


def gpu_lock(request: dict[str, Any]) -> None:
    verify_remote_identity(request)
    root, paths = ensure_identity_root(request)
    require_private_dir(root, os.getuid())
    for path in (paths["checkout"], paths["build"], paths["results"]):
        require_private_dir(path, os.getuid())
    require_identity_marker(root, request["identity"])
    lock_path = request["research_root"]
    if not shutil.which("flock"):
        raise RuntimeError("flock is unavailable")
    command = request["argv"]
    if not command:
        raise RuntimeError("no command was provided")
    env = {
        "HOME": request["remote_home"],
        "PATH": "/usr/local/cuda-13/bin:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "CUDA_HOME": "/usr/local/cuda-13",
        "NINFER_RESEARCH_ID": request["identity"],
        "NINFER_RESEARCH_ROOT": root,
        "NINFER_RESEARCH_CHECKOUT": paths["checkout"],
        "NINFER_RESEARCH_BUILD_DIR": paths["build"],
        "NINFER_RESEARCH_RESULT_ROOT": paths["results"],
        "NINFER_GPU_SERIALIZATION_LOCK": lock_path,
    }
    os.umask(0o077)
    os.chdir(paths["checkout"])
    # flock blocks in the kernel until acquired; it does not poll and creates no lock file.
    os.execvpe("flock", ["flock", "--exclusive", lock_path, "/usr/bin/env", "--", *command], env)


def cleanup(request: dict[str, Any]) -> None:
    verify_remote_identity(request)
    root, _paths = ensure_identity_root(request)
    if not Path(root).exists():
        print(json.dumps({"identity": request["identity"], "root": root, "exists": False, "removed": False}))
        return
    require_private_dir(root, os.getuid())
    require_identity_marker(root, request["identity"])
    root_resolved = str(Path(root).resolve(strict=True))
    if root_resolved != root or any(paths_overlap(root, path) for path in request["protected_paths"]):
        raise RuntimeError("refusing cleanup of an unsafe path")
    if not request["apply"]:
        print(json.dumps({"identity": request["identity"], "root": root, "exists": True, "removed": False, "would_remove": root}))
        return
    shutil.rmtree(root)
    print(json.dumps({"identity": request["identity"], "root": root, "exists": False, "removed": True}))


def record_build(request: dict[str, Any]) -> None:
    verify_remote_identity(request)
    root, paths = ensure_identity_root(request)
    require_private_dir(root, os.getuid())
    require_private_dir(paths["build"], os.getuid())
    require_private_dir(paths["results"], os.getuid())
    require_identity_marker(root, request["identity"])

    binary_path = guard_path(request["binary_path"], paths["build"], request["protected_paths"])
    binary_stat = os.stat(binary_path)
    if not stat.S_ISREG(binary_stat.st_mode) or binary_stat.st_uid != os.getuid():
        raise RuntimeError("build binary is not a user-owned regular file")
    cache_path = os.path.join(paths["build"], "CMakeCache.txt")
    cache_stat = os.lstat(cache_path)
    if not stat.S_ISREG(cache_stat.st_mode) or cache_stat.st_uid != os.getuid():
        raise RuntimeError("CMakeCache.txt is not a user-owned regular file")

    config_keys = {
        "BUILD_TESTING",
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_GENERATOR",
        "NINFER_BUILD_APPS",
        "NINFER_BUILD_BENCHMARKS",
    }
    config: dict[str, str] = {}
    for line in Path(cache_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key, value = line.split("=", 1)
        name = key.split(":", 1)[0]
        if name in config_keys:
            config[name] = value

    manifest = Path(paths["results"]) / "provenance.json"
    manifest_stat = os.lstat(manifest)
    if not stat.S_ISREG(manifest_stat.st_mode) or manifest_stat.st_uid != os.getuid() or stat.S_IMODE(manifest_stat.st_mode) & 0o077:
        raise RuntimeError("provenance manifest is not an owner-only regular file")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    environment = base_metadata(request)
    metadata["toolchain"] = environment["toolchain"]
    metadata["gpu"] = environment["gpu"]
    metadata["artifact"] = environment["artifact"]
    initial_selected = metadata.get("selected_checkout", {})
    current_selected = environment["selected_checkout"]
    stable_selected_keys = ("path", "head", "branch", "status_entry_count")
    metadata["selected_checkout_unchanged_since_workspace_creation"] = all(
        initial_selected.get(key) == current_selected.get(key) for key in stable_selected_keys
    )
    metadata["build"] = {
        "state": "built",
        "directory": paths["build"],
        "binary": binary_path,
        "binary_sha256": sha256_file(binary_path),
        "binary_size_bytes": binary_stat.st_size,
        "cmake_cache": cache_path,
        "cmake_cache_sha256": sha256_file(cache_path),
        "configuration": config,
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest, 0o600)
    print(json.dumps({"identity": request["identity"], "manifest": str(manifest), "build": metadata["build"]}, sort_keys=True))


def main(request: dict[str, Any]) -> None:
    action = request["action"]
    handlers = {
        "plan": plan,
        "prepare": prepare,
        "finalize": finalize,
        "lock_plan": lock_plan,
        "gpu_lock": gpu_lock,
        "cleanup": cleanup,
        "record_build": record_build,
    }
    handler = handlers.get(action)
    if handler is None:
        raise RuntimeError(f"unsupported action: {action}")
    handler(request)
