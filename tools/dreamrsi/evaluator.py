#!/usr/bin/env python3
"""Single-owner FIFO worker for bounded, isolated FNN NInfer evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
import uuid

try:
    from .lifecycle import (
        GpuLease,
        GpuSampler,
        LifecycleError,
        ManagedProcess,
        ProductionServiceActive,
        assert_ports_free,
        assert_service_units_inactive,
        gpu_snapshot,
        wait_for_health,
    )
    from .submit import SCHEMA_VERSION as SUBMISSION_SCHEMA, _file_sha256, validate_overrides
except ImportError:
    from lifecycle import (  # type: ignore[no-redef]
        GpuLease,
        GpuSampler,
        LifecycleError,
        ManagedProcess,
        ProductionServiceActive,
        assert_ports_free,
        assert_service_units_inactive,
        gpu_snapshot,
        wait_for_health,
    )
    from submit import SCHEMA_VERSION as SUBMISSION_SCHEMA, _file_sha256, validate_overrides


RESEARCH_PORT = 18081
CAPTURE_PORT = 18080
CAPTURE_HOST = "100.119.241.20"
SERVICE_HOST = "127.0.0.1"
MODEL_ID = "qwen/qwen3.8-27b"
EXPECTED_GPU_UUID = "GPU-008f1b73-105e-b50b-7469-ddca95c233f6"
PRODUCTION_UNITS = [
    "cinference-qwen38-4090.service",
    "ninfer-qwen38-4090.service",
    "ninfer-qwen38.service",
]
BASELINE_OVERRIDES: dict[str, Any] = {
    "max_context": 188416,
    "kv_capacity": 188416,
    "max_concurrency": 1,
    "max_pending_requests": 16,
    "pending_timeout_ms": 600000,
    "prefill_chunk": 1024,
    "kv_dtype": "k8v4",
    "spec": "mtp",
    "draft_tokens": 7,
    "lm_head_draft": True,
    "default_max_tokens": 8192,
    "seed": 7008,
    "preserve_thinking": True,
}
QUICK_SEED = 7007
QUICK_PER_BUCKET = 3
CACHE_RECIPE_ID = "mtp7-k8v4-prefix-default-unknown-v1"
SECRET_ENV_MARKERS = ("API_KEY", "SECRET", "TOKEN", "CREDENTIAL", "PASSWORD")
THINKING_SUMMARIZED_DERIVATION = [
    {"path": "$.thinking.display", "from": "omitted", "to": "summarized"},
    {"path": "$.output_config.effort", "from": "high", "to": "xhigh"},
]


def _sanitized_environment() -> dict[str, str]:
    """Keep runtime configuration while preventing credentials reaching child tools."""
    sanitized = {}
    for name, value in os.environ.items():
        upper_name = name.upper()
        if upper_name.startswith(("ANTHROPIC_", "CLAUDE_")):
            continue
        if any(marker in upper_name for marker in SECRET_ENV_MARKERS):
            continue
        sanitized[name] = value
    return sanitized


class EvaluationError(RuntimeError):
    pass


class RequestRejectedError(EvaluationError):
    """A captured request was rejected before it produced inference telemetry."""

    def __init__(self, status: int, response_error: dict[str, str]) -> None:
        self.status = status
        self.response_error = response_error
        super().__init__(f"Serve rejected captured replay with HTTP {status}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise


def _run(command: list[str], *, cwd: Path | None = None,
         timeout: float | None = None, log_path: Path | None = None) -> dict[str, Any]:
    environment = _sanitized_environment()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with log_path.open("ab", buffering=0) as log:
            result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False, env=environment)
        output = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    else:
        result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False,
                                text=True, env=environment)
        output = result.stdout[-8000:]
    return {"command": command, "returncode": result.returncode, "output_tail": output}


def _extract_source(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        marker = destination / ".e0007-snapshot-id"
        if marker.is_file() and marker.read_text(encoding="ascii").strip() == destination.name:
            return
        raise EvaluationError(f"unidentified source snapshot already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise EvaluationError(f"source archive has unsafe path: {member.name!r}")
                if member.issym() or member.islnk():
                    target = PurePosixPath(member.linkname)
                    if target.is_absolute() or ".." in target.parts:
                        raise EvaluationError(f"source archive has unsafe link: {member.name!r}")
            archive.extractall(temporary)
        marker = temporary / ".e0007-snapshot-id"
        marker.write_text(destination.name, encoding="ascii")
        marker.chmod(0o444)
        for path in sorted(temporary.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_symlink():
                continue
            if path.is_dir():
                path.chmod(0o555)
            else:
                path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        temporary.chmod(0o555)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            for path in temporary.rglob("*"):
                try:
                    if not path.is_symlink():
                        path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            try:
                temporary.chmod(0o700)
            except OSError:
                pass
            shutil.rmtree(temporary, ignore_errors=True)


def _manifest_at(queue_root: Path, path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SUBMISSION_SCHEMA:
        raise EvaluationError(f"unsupported submission schema in {path}")
    if manifest.get("role") not in {"baseline", "control", "candidate"}:
        raise EvaluationError(f"invalid role in submission {path}")
    try:
        validate_overrides(manifest.get("overrides", {}))
    except RuntimeError as exc:
        raise EvaluationError(f"invalid allowlisted overrides in {path}: {exc}") from exc
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise EvaluationError(f"missing frozen source identity in {path}")
    archive = (queue_root / source.get("archive", "")).resolve()
    try:
        archive.relative_to(queue_root.resolve())
    except ValueError as exc:
        raise EvaluationError("source archive path escapes queue root") from exc
    if not archive.is_file() or _file_sha256(archive) != source.get("archive_sha256"):
        raise EvaluationError(f"frozen source archive is missing or changed: {archive}")
    return manifest


def _take_next(queue_root: Path) -> tuple[dict[str, Any], Path, Path] | None:
    pending = sorted((queue_root / "pending").glob("*.json"))
    if not pending:
        return None
    path = pending[0]
    manifest = _manifest_at(queue_root, path)
    running_path = queue_root / "running" / path.name
    os.replace(path, running_path)
    archive = queue_root / manifest["source"]["archive"]
    return manifest, running_path, archive


def _expected_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    settings = dict(BASELINE_OVERRIDES)
    overrides = manifest.get("overrides", {})
    if not isinstance(overrides, dict):
        raise EvaluationError("submission overrides are not a JSON object")
    allowed = {
        "max_context", "kv_capacity", "max_concurrency", "max_pending_requests",
        "pending_timeout_ms", "prefill_chunk", "kv_dtype", "spec", "draft_tokens",
        "lm_head_draft", "preserve_thinking", "default_max_tokens", "no_cuda_graph",
        "no_prefix_reuse", "seed",
    }
    unexpected = set(overrides) - allowed
    if unexpected:
        raise EvaluationError(f"submission has unallowlisted runtime overrides: {sorted(unexpected)}")
    settings.update(overrides)
    if settings.get("max_concurrency") != 1:
        raise EvaluationError("research worker only accepts max_concurrency=1")
    return settings


def _server_command(binary: Path, artifact_path: str, request_log: Path,
                    settings: dict[str, Any], port: int) -> list[str]:
    argv = [
        str(binary), artifact_path,
        "--host", SERVICE_HOST,
        "--port", str(port),
        "--model-id", MODEL_ID,
        "--max-context", str(settings["max_context"]),
        "--kv-capacity", str(settings["kv_capacity"]),
        "--max-concurrency", str(settings["max_concurrency"]),
        "--max-pending-requests", str(settings["max_pending_requests"]),
        "--pending-timeout-ms", str(settings["pending_timeout_ms"]),
        "--prefill-chunk", str(settings["prefill_chunk"]),
        "--kv-dtype", str(settings["kv_dtype"]),
        "--default-max-tokens", str(settings["default_max_tokens"]),
        "--request-log-jsonl", str(request_log),
        "--log-level", "info",
        "--seed", str(settings["seed"]),
    ]
    if settings.get("spec"):
        argv.extend(["--spec", str(settings["spec"])])
    if "draft_tokens" in settings:
        argv.extend(["--draft-tokens", str(settings["draft_tokens"])])
    if settings.get("lm_head_draft"):
        argv.append("--lm-head-draft")
    if settings.get("preserve_thinking"):
        argv.append("--preserve-thinking")
    if settings.get("no_cuda_graph"):
        argv.append("--no-cuda-graph")
    if settings.get("no_prefix_reuse"):
        argv.append("--no-prefix-reuse")
    return argv


def _runtime_version(command: list[str]) -> str:
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=10,
                            env=_sanitized_environment())
    return result.stdout.strip()[:3000]


def _verify_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    raw_path = artifact.get("path")
    expected_sha256 = artifact.get("sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise EvaluationError("artifact path must be absolute")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise EvaluationError("artifact manifest has no valid SHA-256")
    path = Path(raw_path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise EvaluationError(f"model artifact is not a regular file: {path}")
    actual_sha256 = _file_sha256(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise EvaluationError("model artifact changed while its SHA-256 was being verified")
    if actual_sha256.lower() != expected_sha256.lower():
        raise EvaluationError("model artifact SHA-256 differs from the queued manifest")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def _source_and_build(root: Path, archive: Path, manifest: dict[str, Any],
                      run_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    snapshot_id = manifest["source"]["snapshot_sha256"]
    source_dir = root / "sources" / snapshot_id
    _extract_source(archive, source_dir)
    build_dir = root / "builds" / snapshot_id
    binary = build_dir / "apps" / "ninfer-serve"
    build_log = root / "build-logs" / f"{snapshot_id}.log"
    if binary.is_file():
        return source_dir, build_dir, {"reused": True, "binary": str(binary),
                                      "binary_sha256": _file_sha256(binary),
                                      "log": str(build_log)}
    build_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    configure = _run([
        "cmake", "-S", str(source_dir), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc",
        "-DCMAKE_CUDA_ARCHITECTURES=89",
        "-DNINFER_BUILD_APPS=ON",
        "-DBUILD_TESTING=OFF",
        "-DNINFER_BUILD_BENCHMARKS=OFF",
    ], timeout=600, log_path=build_log)
    if configure["returncode"] != 0:
        raise EvaluationError(f"CMake configure failed; see {build_log}")
    build = _run(["cmake", "--build", str(build_dir), "-j", "--target", "ninfer-serve"],
                 timeout=3600, log_path=build_log)
    if build["returncode"] != 0 or not binary.is_file():
        raise EvaluationError(f"isolated NInfer build failed; see {build_log}")
    binary.chmod(0o500)
    return source_dir, build_dir, {"reused": False, "binary": str(binary),
                                   "binary_sha256": _file_sha256(binary),
                                   "configure_returncode": configure["returncode"],
                                   "build_returncode": build["returncode"],
                                   "log": str(build_log)}


def _request_body(message: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [{"role": "user", "content": message}],
    }


def _post_probe(base_url: str, message: str, *, max_tokens: int) -> dict[str, Any]:
    body = json.dumps(_request_body(message, max_tokens), separators=(",", ":")).encode("utf-8")
    request = Request(
        f"{base_url}/v1/messages",
        data=body,
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=240) as response:
            status = response.status
            payload = json.loads(response.read())
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"bounded model request failed: {exc}") from exc
    elapsed = time.monotonic() - started
    if status != 200:
        raise EvaluationError(f"bounded model request returned HTTP {status}")
    usage = payload.get("usage", {})
    return {
        "http_status": status,
        "wall_seconds": elapsed,
        "finish_reason": payload.get("stop_reason"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "response_id": payload.get("id"),
        "complete": True,
    }


def _model_probe(base_url: str, message: str, *, max_tokens: int,
                 request_log: Path, sampler: GpuSampler) -> dict[str, Any]:
    before = _request_id_set(request_log)
    sample_index = len(sampler.samples)
    response = _post_probe(base_url, message, max_tokens=max_tokens)
    request_start, terminal = _match_request(request_log, before)
    response["request_start"] = request_start
    response["terminal"] = terminal
    response["vram_peak_bytes"] = _peak_memory_bytes(sampler.samples[sample_index:] + [gpu_snapshot()])
    return response


def _request_events(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    return [row for row in rows if row.get("event") in {
        "request_start", "request_done", "request_error", "request_rejected"
    }]


def _match_request(log_path: Path, request_id_before: set[str], *,
                   wait_seconds: float = 5.0) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    deadline = time.monotonic() + wait_seconds
    start = None
    while time.monotonic() < deadline:
        events = _request_events(log_path)
        starts = [
            row for row in events
            if row.get("event") == "request_start"
            and str(row.get("request", {}).get("request_id")) not in request_id_before
        ]
        if starts:
            start = starts[-1]
            rid = str(start.get("request", {}).get("request_id"))
            terminals = [
                row for row in events
                if row.get("event") in {"request_done", "request_error"}
                and str(row.get("request", {}).get("request_id")) == rid
            ]
            if terminals:
                return start, terminals[-1]
        rejected = [
            row for row in events
            if row.get("event") == "request_rejected"
            and str(row.get("request", {}).get("request_id")) not in request_id_before
        ]
        if rejected:
            return None, rejected[-1]
        time.sleep(0.05)
    if start is not None:
        return start, None
    raise EvaluationError("Serve request telemetry did not record the request")


def _request_id_set(path: Path) -> set[str]:
    return {
        str(row.get("request", {}).get("request_id"))
        for row in _request_events(path)
        if row.get("event") in {"request_start", "request_rejected"}
    }


def _port_is_listening(port: int) -> bool:
    result = subprocess.run(["ss", "-ltnH"], check=False, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=10)
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4:
            try:
                if int(fields[3].rsplit(":", 1)[1]) == port:
                    return True
            except (IndexError, ValueError):
                pass
    return False


def _await_capture_proxy(tools_dir: Path, capture_dir: Path, descriptors: Path,
                         run_dir: Path, root: Path, capture_port: int,
                         capture_window_seconds: int) -> dict[str, Any]:
    script = tools_dir / "capture.py"
    if not script.is_file():
        raise EvaluationError(f"capture proxy is unavailable: {script}")
    capture_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    capture_dir.chmod(0o700)
    descriptor_args = [
        sys.executable, str(script),
        "--listen-host", CAPTURE_HOST,
        "--listen-port", str(capture_port),
        "--upstream", f"http://{SERVICE_HOST}:{RESEARCH_PORT}",
        "--capture-dir", str(capture_dir),
        "--descriptor-jsonl", str(descriptors),
        "--max-requests", "1",
        "--max-seconds", str(capture_window_seconds),
        "--max-output-tokens", "128",
        "--cache-state", "unknown",
    ]
    proxy = ManagedProcess.start(descriptor_args, log_path=run_dir / "capture-proxy.log",
                                 env=_sanitized_environment())
    result: dict[str, Any] = {}
    proxy_cleanup: dict[str, Any] | None = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and proxy.is_running():
            if _port_is_listening(capture_port):
                break
            else:
                time.sleep(0.1)
        else:
            raise EvaluationError("capture proxy did not bind its temporary endpoint")
        result = {
            "event": "research_endpoint_ready",
            "url": f"http://{CAPTURE_HOST}:{capture_port}",
            "upstream": f"http://{SERVICE_HOST}:{RESEARCH_PORT}",
            "max_accepted_requests": 1,
            "max_output_tokens": 128,
            "deadline_seconds": capture_window_seconds,
            "capture_dir": str(capture_dir),
            "descriptor_jsonl": str(descriptors),
        }
        _write_json(root / "capture-ready.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        deadline = time.monotonic() + capture_window_seconds + 5
        rows: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            rows = _read_jsonl(descriptors)
            if rows:
                budgets = [row.get("output_budget") for row in rows]
                if any(not isinstance(value, int) or value < 1 or value > 128 for value in budgets):
                    raise EvaluationError("captured request has no valid bounded output_budget")
                break
            if not proxy.is_running():
                raise EvaluationError("capture proxy stopped before an accepted request was captured")
            time.sleep(0.2)
        else:
            raise EvaluationError("no bounded Claude request arrived before capture deadline")
        result["accepted_descriptors"] = len(rows)
        result["capsule_ids"] = [row.get("capsule_id") for row in rows]
        result["output_budgets"] = budgets
    finally:
        proxy_cleanup = proxy.terminate_group(timeout_seconds=5)
    result["proxy_cleanup"] = proxy_cleanup
    result["listener_stopped"] = _listener_stopped(capture_port)
    if not result["listener_stopped"] or proxy_cleanup.get("process_group_alive"):
        raise EvaluationError("capture proxy did not fully stop after the bounded capture")
    return result


def _peak_memory_bytes(samples: list[dict[str, Any]]) -> int | None:
    values = [
        gpu.get("memory_used_mib", 0)
        for sample in samples
        for gpu in sample.get("gpus", [])
        if isinstance(gpu.get("memory_used_mib"), int)
    ]
    return None if not values else max(values) * 1024 * 1024


def _capsule_map(descriptors_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(descriptors_path):
        capsule_id = row.get("capsule_id")
        if capsule_id:
            result[str(capsule_id)] = row
    return result


def _create_plan(quick_eval: Path, descriptors: Path, run_dir: Path) -> dict[str, Any]:
    descriptor_rows = _read_jsonl(descriptors)
    budgets = {row.get("output_budget") for row in descriptor_rows}
    if len(budgets) != 1:
        raise EvaluationError(f"captured capsules have missing or mixed output budgets: {sorted(map(str, budgets))}")
    output_budget = next(iter(budgets))
    if not isinstance(output_budget, int) or output_budget < 1 or output_budget > 128:
        raise EvaluationError("captured S1 output budget is not within the 128-token bound")
    plan_path = run_dir / "quick-plan.json"
    command = [
        sys.executable, str(quick_eval), "sample",
        "--manifest", str(descriptors),
        "--seed", str(QUICK_SEED),
        "--per-bucket", str(QUICK_PER_BUCKET),
        "--cache-recipe-id", CACHE_RECIPE_ID,
        "--output-budget", str(output_budget),
        "--output", str(plan_path),
    ]
    result = _run(command, timeout=60, log_path=run_dir / "quick-sample.log")
    if result["returncode"] != 0 or not plan_path.is_file():
        raise EvaluationError(f"quick-eval sampler failed; see {run_dir / 'quick-sample.log'}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan.get("selected"):
        raise EvaluationError("quick-eval sampler returned an empty plan")
    return plan


def _validate_derived_capsule_descriptors(capture_dir: Path, descriptors: Path,
                                          expected_parent_request_sha256: str
                                          ) -> dict[str, Any]:
    """Validate the one approved exploratory display transform without exposing its body."""
    if (len(expected_parent_request_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_parent_request_sha256)):
        raise EvaluationError("derived capsule parent request SHA-256 is invalid")
    capture_root = capture_dir.resolve(strict=True)
    descriptor_path = descriptors.resolve(strict=True)
    try:
        descriptor_path.relative_to(capture_root)
    except ValueError as exc:
        raise EvaluationError("derived descriptor manifest must be inside the capture directory") from exc
    if descriptor_path.stat().st_mode & 0o077:
        raise EvaluationError("derived descriptor manifest must be owner-only")
    rows = _read_jsonl(descriptor_path)
    if len(rows) != 1:
        raise EvaluationError("exploratory reuse requires exactly one derived capsule descriptor")
    descriptor = rows[0]
    capsule_value = descriptor.get("capsule_path")
    if not isinstance(capsule_value, str):
        raise EvaluationError("derived capsule descriptor has no capsule path")
    capsule = Path(capsule_value).resolve(strict=True)
    try:
        capsule.relative_to(capture_root)
    except ValueError as exc:
        raise EvaluationError("derived capsule must be inside the capture directory") from exc
    if not capsule.is_dir() or capsule.stat().st_mode & 0o077:
        raise EvaluationError("derived capsule directory must be owner-only")
    try:
        manifest = json.loads((capsule / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("derived capsule manifest could not be read") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise EvaluationError("unsupported derived capsule schema")
    request = manifest.get("request")
    derivation = manifest.get("derivation")
    if not isinstance(request, dict) or not isinstance(derivation, dict):
        raise EvaluationError("derived capsule is missing request or derivation provenance")
    if (request.get("method") != "POST"
            or not isinstance(request.get("target"), str)
            or request["target"].split("?", 1)[0] != "/v1/messages"
            or descriptor.get("capsule_id") != manifest.get("capsule_id")):
        raise EvaluationError("derived capsule descriptor does not identify a messages request")
    body_name = request.get("body_file")
    if not isinstance(body_name, str) or Path(body_name).name != body_name:
        raise EvaluationError("derived capsule request body path is invalid")
    body_path = capsule / body_name
    if body_path.stat().st_mode & 0o077:
        raise EvaluationError("derived capsule request body must be owner-only")
    body = body_path.read_bytes()
    request_sha256 = hashlib.sha256(body).hexdigest()
    if (request_sha256 != request.get("body_sha256")
            or request_sha256 != descriptor.get("request_sha256")
            or request_sha256 != derivation.get("transformed_body_sha256")):
        raise EvaluationError("derived capsule request identity does not match its provenance")
    if (derivation.get("semantic_transform") is not True
            or derivation.get("production_compatible") is not False
            or derivation.get("parent_raw_request_sha256") != expected_parent_request_sha256
            or derivation.get("diff") != THINKING_SUMMARIZED_DERIVATION):
        raise EvaluationError("derived capsule is not the approved summarized-thinking/xhigh transformation")
    try:
        request_body = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("derived capsule request body is not valid JSON") from exc
    if not isinstance(request_body, dict):
        raise EvaluationError("derived capsule request body must be a JSON object")
    thinking = request_body.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("display") != "summarized":
        raise EvaluationError("derived capsule body does not contain summarized thinking display")
    output_config = request_body.get("output_config")
    if not isinstance(output_config, dict) or output_config.get("effort") != "xhigh":
        raise EvaluationError("derived capsule body does not contain the supported xhigh effort")
    output_budget = request_body.get("max_tokens")
    if (type(output_budget) is not int or not 1 <= output_budget <= 128
            or descriptor.get("output_budget") != output_budget):
        raise EvaluationError("derived capsule output budget is missing or exceeds the 128-token bound")
    return {
        "event": "precaptured_derived_capsule_selected",
        "capsule_id": manifest.get("capsule_id"),
        "capsule_path": str(capsule),
        "request_sha256": request_sha256,
        "output_budget": output_budget,
        "parent_capsule_id": derivation.get("parent_capsule_id"),
        "parent_request_sha256": expected_parent_request_sha256,
        "semantic_transform": True,
        "production_compatible": False,
        "diff": [dict(change) for change in THINKING_SUMMARIZED_DERIVATION],
    }


def _run_plan(quick_eval: Path, tools_dir: Path, plan: dict[str, Any],
              descriptors_path: Path, request_log: Path, run_dir: Path,
              sampler: GpuSampler, identity: dict[str, Any]) -> Path:
    replay_script = tools_dir / "replay.py"
    if not replay_script.is_file():
        raise EvaluationError(f"capsule replay tool is unavailable: {replay_script}")
    capsules = _capsule_map(descriptors_path)
    output_path = run_dir / "quick-results.jsonl"
    output_path.unlink(missing_ok=True)
    for selected in plan["selected"]:
        capsule_id = str(selected["capsule_id"])
        capsule = capsules.get(str(capsule_id))
        if capsule is None:
            raise EvaluationError(f"sampled capsule is absent from descriptor manifest: {capsule_id}")
        capsule_path = Path(selected["capsule_path"]).resolve()
        before = _request_id_set(request_log)
        sample_index = len(sampler.samples)
        response_path = run_dir / f"response-{capsule_id}.tmp"
        command = [
            sys.executable, str(replay_script), str(capsule_path),
            "--upstream", f"http://{SERVICE_HOST}:{RESEARCH_PORT}",
            "--response-output", str(response_path),
        ]
        start_wall = time.monotonic()
        replay = _run(command, timeout=300)
        wall_seconds = time.monotonic() - start_wall
        replay_summary: dict[str, Any] = {}
        try:
            parsed_summary = json.loads(replay.get("output_tail", "{}"))
            if isinstance(parsed_summary, dict):
                replay_summary = parsed_summary
        except json.JSONDecodeError:
            replay_summary = {}
        _raise_for_rejected_replay(replay_summary, response_path)
        if replay["returncode"] != 0:
            response_path.unlink(missing_ok=True)
            raise EvaluationError(
                f"capsule replay tool exited with status {replay['returncode']}"
            )
        start, terminal = _match_request(request_log, before)
        response: dict[str, Any] = {}
        if response_path.exists():
            try:
                response = _response_summary(response_path)
            finally:
                response_path.unlink(missing_ok=True)
        usage = response.get("usage", {})
        semantic_transform = selected.get(
            "semantic_transform", plan.get("semantic_transform", False)
        )
        production_compatible = selected.get(
            "production_compatible", plan.get("production_compatible", not semantic_transform)
        )
        parent_request_sha256 = selected.get("parent_request_sha256")
        derived_request_sha256 = selected.get("derived_request_sha256")
        transform_diff = selected.get("transform_diff")
        if semantic_transform and derived_request_sha256 != selected.get("request_sha256"):
            raise EvaluationError("planned derived request hash differs from the replayed capsule")
        row = {
            "schema_version": 1,
            "plan_id": plan.get("plan_id"),
            "cache_recipe_id": plan.get("cache_recipe_id"),
            "capsule_id": str(capsule_id),
            "captured_request_identity": {
                "request_sha256": selected.get("request_sha256"),
                "captured_output_budget": selected.get("capture_output_budget",
                                                       capsule.get("output_budget")),
                "cache_state": capsule.get("metadata", {}).get("cache_state"),
                "semantic_transform": semantic_transform,
                "production_compatible": production_compatible,
                "parent_request_sha256": parent_request_sha256,
                "derived_request_sha256": derived_request_sha256,
                "transform_diff": transform_diff,
            },
            "semantic_transform": semantic_transform,
            "production_compatible": production_compatible,
            "parent_request_sha256": parent_request_sha256,
            "derived_request_sha256": derived_request_sha256,
            "transform_diff": transform_diff,
            "request_start": start,
            "terminal": terminal,
            "client": {
                "complete": (replay["returncode"] == 0 and replay_summary.get("complete") is True
                             and terminal is not None and terminal.get("event") == "request_done"),
                "finish_reason": response.get("stop_reason", response.get("finish_reason",
                                                                            (terminal or {}).get("result", {}).get("finish_reason"))),
                "received_completion_tokens": usage.get("output_tokens"),
                "ttft_seconds": _first_token_seconds(terminal),
                "wall_seconds": wall_seconds,
                "error": (None if replay["returncode"] == 0 and terminal is not None
                          else f"replay exit {replay['returncode']}" if replay["returncode"] != 0
                          else "missing Serve terminal telemetry"),
            },
            "vram_peak_bytes": _peak_memory_bytes(sampler.samples[sample_index:] + [gpu_snapshot()]),
            "identity": identity,
        }
        _append_jsonl(output_path, row)
    return output_path


def _response_summary(path: Path) -> dict[str, Any]:
    """Extract only completion metadata from a response body, never retain response text."""
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return {
                "usage": value.get("usage", {}) if isinstance(value.get("usage"), dict) else {},
                "stop_reason": value.get("stop_reason"),
                "finish_reason": value.get("finish_reason"),
            }
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    usage: dict[str, Any] = {}
    stop_reason = None
    finish_reason = None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        content = line[5:].strip()
        if not content or content == "[DONE]":
            continue
        try:
            event = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "message_start":
            message = event.get("message", {})
            if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                usage.update(message["usage"])
        if event.get("type") == "message_delta":
            delta = event.get("delta", {})
            if isinstance(delta, dict):
                stop_reason = delta.get("stop_reason", stop_reason)
                finish_reason = delta.get("finish_reason", finish_reason)
            if isinstance(event.get("usage"), dict):
                usage.update(event["usage"])
    return {"usage": usage, "stop_reason": stop_reason, "finish_reason": finish_reason}


_SAFE_RESPONSE_ERROR_FIELDS = ("type", "code", "param", "message")
_CREDENTIAL_TEXT_RE = re.compile(r"(?i)(?:sk-ant-[A-Za-z0-9_-]+|bearer\s+\S+)")


def _response_error_summary(path: Path) -> dict[str, str]:
    """Keep only short, noncredential error fields from a rejected HTTP response."""
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    error = value.get("error")
    source = error if isinstance(error, dict) else value
    summary: dict[str, str] = {}
    for field in _SAFE_RESPONSE_ERROR_FIELDS:
        item = source.get(field)
        if isinstance(item, str) and item.strip():
            cleaned = _CREDENTIAL_TEXT_RE.sub("[redacted]", " ".join(item.split()))
            summary[field] = cleaned[:500]
    return summary


def _raise_for_rejected_replay(replay_summary: dict[str, Any],
                               response_path: Path) -> None:
    status = replay_summary.get("status")
    if not isinstance(status, int) or not 400 <= status < 500:
        return
    error = _response_error_summary(response_path) if response_path.is_file() else {}
    # The original capsule retains the restricted raw response; discard the duplicate
    # replay body after extracting the bounded error metadata.
    response_path.unlink(missing_ok=True)
    raise RequestRejectedError(status, error)


def _first_token_seconds(terminal: dict[str, Any] | None) -> float | None:
    if terminal is None:
        return None
    timings = terminal.get("timings_seconds", terminal.get("timing", terminal.get("timings", {})))
    for key in ("ttft", "ttft_seconds", "first_token_seconds", "time_to_first_token_seconds"):
        value = timings.get(key) if isinstance(timings, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _reduce(quick_eval: Path, plan: Path, baseline: Path, control: Path,
            output: Path, log: Path) -> None:
    result = _run([
        sys.executable, str(quick_eval), "reduce",
        "--plan", str(plan),
        "--baseline-results", str(baseline),
        "--candidate-results", str(control),
        "--output", str(output),
    ], timeout=60, log_path=log)
    if result["returncode"] != 0 or not output.is_file():
        raise EvaluationError(f"quick-eval reducer failed; see {log}")


def _server_start_record(path: Path) -> dict[str, Any] | None:
    for row in _read_jsonl(path):
        if row.get("event") == "server_start":
            return row
    return None


def _startup_headroom(server_start: dict[str, Any] | None) -> dict[str, Any]:
    memory = (server_start or {}).get("memory", {})
    available = memory.get("available_after_startup_bytes")
    if not isinstance(available, int):
        return {"threshold_bytes": 1024**3, "available_bytes": None, "meets_1gib": None}
    return {"threshold_bytes": 1024**3, "available_bytes": available,
            "margin_bytes": available - 1024**3, "meets_1gib": available >= 1024**3}


def _attempt_run(*, manifest: dict[str, Any], archive: Path, root: Path, tools_dir: Path,
                 capture_dir: Path, descriptors: Path, quick_eval: Path, probe_text: str,
                 service_port: int, capture_port: int, capture_window_seconds: int,
                 shared_build: bool,
                 precaptured_workload: dict[str, Any] | None = None) -> dict[str, Any]:
    queue_info = manifest.get("queue", {})
    attempt_id = f"{queue_info.get('sequence', 0):012d}-{manifest['submission_id']}"
    run_dir = root / "runs" / attempt_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    run_dir.chmod(0o700)
    request_log = run_dir / "requests.jsonl"
    service_log = run_dir / "serve.log"
    settings = _expected_settings(manifest)
    artifact = manifest["artifact"]
    source_dir, build_dir, build = _source_and_build(root, archive, manifest, run_dir)
    binary = Path(build["binary"])
    if not binary.is_file():
        raise EvaluationError(f"built ninfer-serve is missing: {binary}")
    build["reused_for_pair"] = shared_build

    preflight = {
        "services": assert_service_units_inactive(PRODUCTION_UNITS),
        "listeners_before": assert_ports_free((service_port, capture_port)),
        "gpu_before": gpu_snapshot(),
        "gpu_compute_apps_before": _gpu_compute_apps(),
    }
    if not preflight["gpu_before"].get("available"):
        raise EvaluationError(f"GPU inventory failed: {preflight['gpu_before'].get('error')}")
    gpus = preflight["gpu_before"].get("gpus", [])
    if not any(gpu.get("index") == 0 and gpu.get("uuid") == EXPECTED_GPU_UUID
               for gpu in gpus):
        raise EvaluationError("the expected RTX 4090 is not visible as CUDA device 0")
    if preflight["gpu_compute_apps_before"] is None:
        raise EvaluationError("could not verify that no other compute process uses the GPU")
    if preflight["gpu_compute_apps_before"]:
        raise EvaluationError("refusing to benchmark while another compute process uses the GPU")
    artifact_verification = _verify_artifact(artifact)

    server_command = _server_command(binary, artifact["path"], request_log, settings, service_port)
    environment = _sanitized_environment()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    sampler = GpuSampler()
    sampler.start()
    try:
        service = ManagedProcess.start(server_command, log_path=service_log, cwd=source_dir,
                                       env=environment)
    except BaseException:
        sampler.stop()
        raise
    proxy_cleanup: dict[str, Any] | None = None
    service_cleanup: dict[str, Any] | None = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "submission_id": manifest["submission_id"],
        "role": manifest["role"],
        "queue_sequence": queue_info.get("sequence"),
        "status": "running",
        "started_at_utc": _utc_now(),
        "source": manifest["source"],
        "baseline_id": manifest["baseline_id"],
        "artifact": artifact,
        "artifact_verification": artifact_verification,
        "runtime_id": manifest["runtime_id"],
        "workload": (dict(precaptured_workload) if precaptured_workload is not None else {
            "semantic_transform": False,
            "production_compatible": True,
        }),
        "settings": settings,
        "build": build,
        "preflight": preflight,
        "process": {"server_pid": service.pid, "server_command": server_command},
        "paths": {"run_dir": str(run_dir), "source_dir": str(source_dir),
                  "build_dir": str(build_dir), "request_log": str(request_log),
                  "server_log": str(service_log)},
    }
    try:
        health = wait_for_health(f"http://{SERVICE_HOST}:{service_port}", service,
                                 timeout_seconds=900, poll_seconds=1)
        result["health"] = health
        server_start = _server_start_record(request_log)
        result["startup_headroom"] = _startup_headroom(server_start)
        if server_start is None:
            raise EvaluationError("health passed but server_start telemetry is absent")
        result["server_start"] = server_start
        result["runtime_versions"] = {
            "host": _runtime_version(["hostname"]),
            "kernel": _runtime_version(["uname", "-r"]),
            "cuda": _runtime_version(["/usr/local/cuda-13/bin/nvcc", "--version"]),
            "cmake": _runtime_version(["cmake", "--version"]),
        }

        identity = {
            "source_commit": manifest["source"]["git_commit"],
            "source_snapshot_sha256": manifest["source"]["snapshot_sha256"],
            "artifact_sha256": artifact["sha256"],
            "runtime_id": manifest["runtime_id"],
            "build_binary_sha256": build["binary_sha256"],
            "server_model_id": MODEL_ID,
            "effective_runtime_settings": settings,
            "generation_seed_override": settings["seed"],
            "cache_recipe_id": CACHE_RECIPE_ID,
        }
        if precaptured_workload is not None:
            identity.update({
                "request_sha256": precaptured_workload["request_sha256"],
                "semantic_transform": True,
                "production_compatible": False,
                "parent_request_sha256": precaptured_workload["parent_request_sha256"],
            })

        # These two short requests establish the S0 plumbing path. The captured S1 request has an
        # independently observed output budget and is never shortened or rewritten on replay.
        warmup = _model_probe(f"http://{SERVICE_HOST}:{service_port}", probe_text,
                              max_tokens=1, request_log=request_log, sampler=sampler)
        result["warmup_request"] = warmup
        s0 = _model_probe(f"http://{SERVICE_HOST}:{service_port}", probe_text,
                          max_tokens=4, request_log=request_log, sampler=sampler)
        result["s0"] = s0

        if manifest["role"] == "baseline":
            if precaptured_workload is None:
                capture = _await_capture_proxy(tools_dir, capture_dir, descriptors,
                                               run_dir, root, capture_port,
                                               capture_window_seconds)
            else:
                capture = dict(precaptured_workload)
            result["capture"] = capture
            plan = _create_plan(quick_eval, descriptors, root / "plans")
            plan_path = root / "plans" / "quick-plan.json"
            _write_json(plan_path, plan)
            _write_json(root / "plans" / "baseline-identity.json", identity)
            result["quick_plan"] = str(plan_path)
            baseline_results = _run_plan(quick_eval, tools_dir, plan, descriptors,
                                         request_log, run_dir, sampler, identity)
            result["quick_results"] = str(baseline_results)
        else:
            plan_path = root / "plans" / "quick-plan.json"
            baseline_identity_path = root / "plans" / "baseline-identity.json"
            if not plan_path.is_file() or not baseline_identity_path.is_file():
                raise EvaluationError("control submission has no baseline S1 plan")
            baseline_identity = json.loads(baseline_identity_path.read_text(encoding="utf-8"))
            if identity != baseline_identity:
                raise EvaluationError("control runtime/source/artifact identity differs from baseline")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            control_results = _run_plan(quick_eval, tools_dir, plan, descriptors,
                                        request_log, run_dir, sampler, identity)
            result["quick_plan"] = str(plan_path)
            result["quick_results"] = str(control_results)

        result["request_events"] = _request_events(request_log)
        result["status"] = "completed"
        return result
    except BaseException as exc:
        result["status"] = "failed"
        result["failure"] = _classify_failure(exc)
        raise
    finally:
        if service.is_running():
            service_cleanup = service.terminate_group(timeout_seconds=20)
        else:
            # Always audit and clear descendants even if the leader already exited.
            service_cleanup = service.terminate_group(timeout_seconds=2)
        gpu_samples = sampler.stop()
        gpu_release = _wait_gpu_release(preflight["gpu_before"], service.pid)
        proxy_group_alive = result.get("capture", {}).get("proxy_cleanup", {}).get(
            "process_group_alive", False
        )
        cleanup_clean = (
            not service_cleanup.get("process_group_alive", True)
            and _listener_stopped(service_port)
            and _listener_stopped(capture_port)
            and not proxy_group_alive
            and gpu_release.get("process_query_available") is True
            and gpu_release.get("service_pid_present") is False
            and gpu_release.get("memory_returned") is True
        )
        result["cleanup"] = {
            "service": service_cleanup,
            "listener_stopped": _listener_stopped(service_port),
            "capture_listener_stopped": _listener_stopped(capture_port),
            "gpu_release": gpu_release,
            "gpu_samples": gpu_samples,
            "clean": cleanup_clean,
        }
        if result.get("status") == "completed" and not cleanup_clean:
            result["status"] = "cleanup_failed"
            result["failure"] = {
                "classification": "unclean_process_listener_or_gpu_teardown",
                "message": "server process group, research listener, or GPU memory audit did not clear",
            }
        result["completed_at_utc"] = _utc_now()
        result["paths"]["server_log"] = str(service_log)
        _write_json(run_dir / "result.json", result)


def _listener_stopped(port: int) -> bool:
    listeners = subprocess.run(["ss", "-ltnH"], check=False, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, timeout=10).stdout
    for line in listeners.splitlines():
        fields = line.split()
        if len(fields) >= 4:
            try:
                if int(fields[3].rsplit(":", 1)[1]) == port:
                    return False
            except (IndexError, ValueError):
                pass
    return True


def _gpu_compute_apps() -> list[dict[str, Any]] | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    apps = []
    for line in result.stdout.splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) == 2:
            try:
                apps.append({"pid": int(columns[0]), "used_gpu_memory_mib": int(columns[1])})
            except ValueError:
                continue
    return apps


def _wait_gpu_release(baseline: dict[str, Any], server_pid: int,
                      timeout_seconds: float = 20.0) -> dict[str, Any]:
    baseline_gpus = baseline.get("gpus", [])
    baseline_mib = baseline_gpus[0].get("memory_used_mib") if baseline_gpus else None
    deadline = time.monotonic() + timeout_seconds
    final_snapshot = gpu_snapshot()
    apps = _gpu_compute_apps()
    while time.monotonic() < deadline:
        final_snapshot = gpu_snapshot()
        apps = _gpu_compute_apps()
        final_gpus = final_snapshot.get("gpus", [])
        used_mib = final_gpus[0].get("memory_used_mib") if final_gpus else None
        service_pid_present = apps is not None and any(app["pid"] == server_pid for app in apps)
        memory_returned = (isinstance(used_mib, int) and isinstance(baseline_mib, int)
                           and used_mib <= baseline_mib + 128)
        if memory_returned and apps is not None and not service_pid_present:
            break
        time.sleep(0.5)
    final_gpus = final_snapshot.get("gpus", [])
    used_mib = final_gpus[0].get("memory_used_mib") if final_gpus else None
    return {
        "snapshot": final_snapshot,
        "compute_apps": apps,
        "baseline_memory_used_mib": baseline_mib,
        "final_memory_used_mib": used_mib,
        "memory_delta_mib": (used_mib - baseline_mib
                              if isinstance(used_mib, int) and isinstance(baseline_mib, int)
                              else None),
        "service_pid_present": (any(app["pid"] == server_pid for app in apps)
                                if apps is not None else None),
        "process_query_available": apps is not None,
        "memory_returned": (isinstance(used_mib, int) and isinstance(baseline_mib, int)
                             and used_mib <= baseline_mib + 128),
    }


def _process_queue(*, queue_root: Path, root: Path, lock_path: Path,
                   tools_dir: Path, capture_dir: Path, descriptors: Path,
                   quick_eval: Path, probe_text: str, service_port: int,
                   capture_port: int, capture_window_seconds: int, max_items: int,
                   require_baseline_control: bool,
                   reuse_derived_capsule: bool = False,
                   derived_parent_request_sha256: str | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    queue_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with GpuLease(lock_path):
        precaptured_workload = None
        if reuse_derived_capsule:
            if not require_baseline_control or max_items != 2:
                raise EvaluationError("derived capsule reuse requires exactly one baseline/control pair")
            if derived_parent_request_sha256 is None:
                raise EvaluationError("derived capsule reuse requires its parent request SHA-256")
            precaptured_workload = _validate_derived_capsule_descriptors(
                capture_dir, descriptors, derived_parent_request_sha256
            )
        elif derived_parent_request_sha256 is not None:
            raise EvaluationError("derived parent request SHA-256 requires derived capsule reuse")
        if require_baseline_control:
            pending = sorted((queue_root / "pending").glob("*.json"))[:2]
            if len(pending) != 2:
                raise EvaluationError("baseline/control run requires exactly two pending queue items")
            first = _manifest_at(queue_root, pending[0])
            second = _manifest_at(queue_root, pending[1])
            if first["role"] != "baseline" or second["role"] != "control":
                raise EvaluationError("FIFO head must be baseline followed by control")
            for field in ("baseline_id", "artifact", "runtime_id", "overrides"):
                if first.get(field) != second.get(field):
                    raise EvaluationError(f"baseline/control {field} mismatch")
            if first["source"]["snapshot_sha256"] != second["source"]["snapshot_sha256"]:
                raise EvaluationError("baseline/control frozen source snapshots differ")
            if first["source"]["git_commit"] != second["source"]["git_commit"]:
                raise EvaluationError("baseline/control Git commits differ")

        shared_build = False
        for _ in range(max_items):
            next_item = _take_next(queue_root)
            if next_item is None:
                break
            manifest, running_path, archive = next_item
            try:
                result = _attempt_run(
                    manifest=manifest,
                    archive=archive,
                    root=root,
                    tools_dir=tools_dir,
                    capture_dir=capture_dir,
                    descriptors=descriptors,
                    quick_eval=quick_eval,
                    probe_text=probe_text,
                    service_port=service_port,
                    capture_port=capture_port,
                    capture_window_seconds=capture_window_seconds,
                    shared_build=shared_build,
                    precaptured_workload=precaptured_workload,
                )
                shared_build = True
                _write_json(root / "results" / f"{manifest['role']}.json", result)
                if result.get("status") != "completed":
                    target = queue_root / "failed" / running_path.name
                    os.replace(running_path, target)
                    results.append(result)
                    break
                target = queue_root / "completed" / running_path.name
                os.replace(running_path, target)
                results.append(result)
            except BaseException as exc:
                result = {
                    "schema_version": 1,
                    "submission_id": manifest.get("submission_id"),
                    "role": manifest.get("role"),
                    "queue_sequence": manifest.get("queue", {}).get("sequence"),
                    "status": "failed",
                    "started_at_utc": _utc_now(),
                    "failure": _classify_failure(exc),
                    "source": manifest.get("source"),
                    "artifact": manifest.get("artifact"),
                    "runtime_id": manifest.get("runtime_id"),
                }
                _write_json(root / "results" / f"{manifest.get('role', 'unknown')}.json", result)
                target = queue_root / "failed" / running_path.name
                os.replace(running_path, target)
                results.append(result)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                # A failed baseline makes the paired control meaningless; leave it queued.
                break
        if len(results) == 2 and all(item.get("status") == "completed" for item in results):
            plan = root / "plans" / "quick-plan.json"
            _reduce(
                quick_eval,
                plan,
                Path(results[0]["quick_results"]),
                Path(results[1]["quick_results"]),
                root / "results" / "baseline-control-quick-eval.json",
                root / "results" / "quick-reduce.log",
            )
        return results


def _classify_failure(exc: BaseException, diagnostic: str = "") -> dict[str, Any]:
    message = str(exc)
    lowered = f"{message}\n{diagnostic}".lower()
    if isinstance(exc, RequestRejectedError):
        return {
            "classification": "request_rejected",
            "type": type(exc).__name__,
            "message": message,
            "http_status": exc.status,
            "response_error": exc.response_error,
        }
    if isinstance(exc, ProductionServiceActive):
        classification = "production_service_active"
    elif "memory" in lowered and any(token in lowered for token in ("cuda", "oom", "alloc")):
        classification = "startup_device_memory_failure"
    elif "health timed out" in lowered or "before health" in lowered:
        classification = "server_startup_or_health_failure"
    elif "quick-eval" in lowered or "capsule" in lowered:
        classification = "quick_eval_input_or_replay_failure"
    elif "listener" in lowered or "port" in lowered:
        classification = "research_port_conflict"
    else:
        classification = "worker_failure"
    return {"classification": classification, "type": type(exc).__name__, "message": message}


def _consume_probe_text(path: Path, research_root: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(research_root.resolve())
    except ValueError as exc:
        raise EvaluationError("probe text file must be inside the owner-only research root") from exc
    mode = resolved.stat().st_mode & 0o777
    if mode & 0o077:
        raise EvaluationError("probe text file must be owner-only (mode 0600 or stricter)")
    text_value = resolved.read_text(encoding="utf-8").strip()
    resolved.unlink()
    if not text_value or len(text_value) > 4096:
        raise EvaluationError("probe text must contain 1..4096 UTF-8 characters")
    return text_value


def _parse_capture_window(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("capture window must be an integer number of seconds") from exc
    if not 60 <= seconds <= 900:
        raise argparse.ArgumentTypeError("capture window must be between 60 and 900 seconds")
    return seconds


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker", help="drain one bounded FIFO run")
    worker.add_argument("--queue-root", type=Path, required=True)
    worker.add_argument("--research-root", type=Path, required=True)
    worker.add_argument("--lock-path", type=Path, required=True)
    worker.add_argument("--tools-dir", type=Path, required=True)
    worker.add_argument("--capture-dir", type=Path, required=True)
    worker.add_argument("--descriptors", type=Path, required=True)
    worker.add_argument("--quick-eval", type=Path, required=True)
    worker.add_argument("--probe-text-file", type=Path, required=True,
                        help="owner-only one-use probe input inside research-root; deleted on read")
    worker.add_argument("--service-port", type=int, default=RESEARCH_PORT)
    worker.add_argument("--capture-port", type=int, default=CAPTURE_PORT)
    worker.add_argument("--capture-window-seconds", type=_parse_capture_window, default=900,
                        help="bounded capture window (60..900 seconds; default 900)")
    worker.add_argument("--max-items", type=int, default=2)
    worker.add_argument("--require-baseline-control", action="store_true")
    worker.add_argument("--reuse-derived-capsule", action="store_true",
                        help="reuse one validated summarized-thinking capsule without Claude capture")
    worker.add_argument("--derived-parent-request-sha256",
                        help="original request identity required with --reuse-derived-capsule")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "worker":
            root = args.research_root.resolve()
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            root.chmod(0o700)
            tools_dir = args.tools_dir.resolve()
            probe_text = _consume_probe_text(args.probe_text_file, root)
            results = _process_queue(
                queue_root=args.queue_root.resolve(),
                root=root,
                lock_path=args.lock_path,
                tools_dir=tools_dir,
                capture_dir=args.capture_dir.resolve(),
                descriptors=args.descriptors.resolve(),
                quick_eval=args.quick_eval.resolve(),
                probe_text=probe_text,
                service_port=args.service_port,
                capture_port=args.capture_port,
                capture_window_seconds=args.capture_window_seconds,
                max_items=args.max_items,
                require_baseline_control=args.require_baseline_control,
                reuse_derived_capsule=args.reuse_derived_capsule,
                derived_parent_request_sha256=args.derived_parent_request_sha256,
            )
        else:
            raise EvaluationError(f"unknown command: {args.command}")
        print(json.dumps({"processed": len(results), "results": results}, sort_keys=True))
        return 0 if all(item.get("status") == "completed" for item in results) else 1
    except KeyboardInterrupt:
        print("evaluator.py: interrupted after teardown", file=sys.stderr)
        return 130
    except (OSError, ValueError, EvaluationError, LifecycleError, ProductionServiceActive,
            subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"evaluator.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
