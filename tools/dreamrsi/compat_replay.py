#!/usr/bin/env python3
"""Run one bounded capsule compatibility probe under the FNN GPU lease."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import uuid

try:
    from .evaluator import (
        CACHE_RECIPE_ID,
        CAPTURE_PORT,
        EXPECTED_GPU_UUID,
        MODEL_ID,
        PRODUCTION_UNITS,
        RESEARCH_PORT,
        SERVICE_HOST,
        EvaluationError,
        RequestRejectedError,
        _append_jsonl,
        _classify_failure,
        _expected_settings,
        _file_sha256,
        _gpu_compute_apps,
        _listener_stopped,
        _peak_memory_bytes,
        _read_jsonl,
        _request_events,
        _run_plan,
        _sanitized_environment,
        _server_command,
        _server_start_record,
        _source_and_build,
        _verify_artifact,
        _wait_gpu_release,
        _write_json,
        _utc_now,
    )
    from .lifecycle import (
        GpuLease,
        GpuSampler,
        ManagedProcess,
        assert_ports_free,
        assert_service_units_inactive,
        gpu_snapshot,
        wait_for_health,
    )
    from .submit import SCHEMA_VERSION as SUBMISSION_SCHEMA
except ImportError:
    from evaluator import (  # type: ignore[no-redef]
        CACHE_RECIPE_ID,
        CAPTURE_PORT,
        EXPECTED_GPU_UUID,
        MODEL_ID,
        PRODUCTION_UNITS,
        RESEARCH_PORT,
        SERVICE_HOST,
        EvaluationError,
        RequestRejectedError,
        _append_jsonl,
        _classify_failure,
        _expected_settings,
        _file_sha256,
        _gpu_compute_apps,
        _listener_stopped,
        _peak_memory_bytes,
        _read_jsonl,
        _request_events,
        _run_plan,
        _sanitized_environment,
        _server_command,
        _server_start_record,
        _source_and_build,
        _verify_artifact,
        _wait_gpu_release,
        _write_json,
        _utc_now,
    )
    from lifecycle import (  # type: ignore[no-redef]
        GpuLease,
        GpuSampler,
        ManagedProcess,
        assert_ports_free,
        assert_service_units_inactive,
        gpu_snapshot,
        wait_for_health,
    )
    from submit import SCHEMA_VERSION as SUBMISSION_SCHEMA  # type: ignore[no-redef]


RUN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$")
MAX_PROBE_OUTPUT_TOKENS = 128
EXPECTED_TRANSFORMS = {
    "preserve-false": {"path": "/preserve_thinking", "from": "absent", "to": False},
    "thinking-summarized": {
        "path": "$.thinking.display", "from": "omitted", "to": "summarized"
    },
    "thinking-summarized-effort-xhigh": [
        {"path": "$.thinking.display", "from": "omitted", "to": "summarized"},
        {"path": "$.output_config.effort", "from": "high", "to": "xhigh"},
    ],
}


def _write_private_module(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_flat_replay_import_package(research_root: Path, tools_dir: Path) -> dict[str, str]:
    """Expose the flat deployed capture module at the package path replay.py imports."""
    source = tools_dir.resolve() / "capture.py"
    if not source.is_file() or source.is_symlink():
        raise EvaluationError(f"flat capture module is unavailable: {source}")
    package_root = research_root / "tools"
    package_dir = package_root / "dreamrsi"
    for directory in (package_root, package_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise EvaluationError("replay import package must not contain symlink directories")
    for init_path in (package_root / "__init__.py", package_dir / "__init__.py"):
        if init_path.is_symlink():
            raise EvaluationError("replay import package initializer must not be a symlink")
        if not init_path.exists():
            _write_private_module(init_path, b"\"\"\"Owner-only replay package.\"\"\"\n")
    destination = package_dir / "capture.py"
    source_bytes = source.read_bytes()
    if destination.is_symlink():
        raise EvaluationError("replay capture module must not be a symlink")
    if not destination.is_file() or _file_sha256(destination) != hashlib.sha256(source_bytes).hexdigest():
        _write_private_module(destination, source_bytes)
    return {
        "module_path": str(destination),
        "module_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def inspect_capsule(capsule_dir: Path, expected_request_sha256: str) -> dict[str, Any]:
    """Validate and summarize a capsule without exposing its prompt or response."""
    capsule = capsule_dir.resolve(strict=True)
    manifest_path = capsule / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("capsule manifest could not be read") from exc
    request = manifest.get("request") if isinstance(manifest, dict) else None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise EvaluationError("unsupported capsule schema")
    if not isinstance(request, dict):
        raise EvaluationError("capsule request is missing")
    if request.get("method") != "POST" or request.get("target", "").split("?", 1)[0] != "/v1/messages":
        raise EvaluationError("compatibility probe requires POST /v1/messages")
    body_name = request.get("body_file")
    if not isinstance(body_name, str) or Path(body_name).name != body_name:
        raise EvaluationError("capsule request body path is invalid")
    body_path = capsule / body_name
    try:
        body = body_path.read_bytes()
    except OSError as exc:
        raise EvaluationError("capsule request body could not be read") from exc
    body_sha256 = hashlib.sha256(body).hexdigest()
    if body_sha256 != request.get("body_sha256") or body_sha256 != expected_request_sha256:
        raise EvaluationError("capsule request SHA-256 does not match its expected identity")
    try:
        body_json = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("capsule request body is not valid JSON") from exc
    if not isinstance(body_json, dict):
        raise EvaluationError("capsule request body must be a JSON object")
    output_budget = body_json.get("max_tokens")
    if type(output_budget) is not int or not 1 <= output_budget <= MAX_PROBE_OUTPUT_TOKENS:
        raise EvaluationError("capsule output budget exceeds the compatibility-probe bound")
    thinking = body_json.get("thinking")
    thinking_summary = None
    if isinstance(thinking, dict):
        thinking_summary = {
            key: thinking[key]
            for key in ("type", "display", "budget_tokens")
            if isinstance(thinking.get(key), (str, int)) and type(thinking[key]) in (str, int)
        }
    raw_derivation = manifest.get("derivation")
    derivation = None
    if isinstance(raw_derivation, dict):
        diff = raw_derivation.get("diff")
        derivation = {
            "semantic_transform": raw_derivation.get("semantic_transform"),
            "production_compatible": raw_derivation.get("production_compatible"),
            "parent_capsule_id": raw_derivation.get("parent_capsule_id"),
            "parent_request_sha256": raw_derivation.get("parent_raw_request_sha256"),
            "derived_request_sha256": raw_derivation.get("transformed_body_sha256"),
            "diff": diff if isinstance(diff, (dict, list)) else None,
        }
    return {
        "capsule_id": manifest.get("capsule_id"),
        "capsule_path": str(capsule),
        "request_sha256": body_sha256,
        "method": request["method"],
        "request_target": request["target"],
        "model": body_json.get("model"),
        "max_tokens": output_budget,
        "thinking": thinking_summary,
        "preserve_thinking": body_json.get("preserve_thinking"),
        "cache_state": manifest.get("cache_state", "unknown"),
        "derivation": derivation,
    }


def _load_provenance(path: Path | None, capsule: dict[str, Any],
                     expected_parent_sha256: str | None,
                     expected_transform: str | None) -> dict[str, Any] | None:
    if path is None:
        value = capsule.get("derivation")
        if value is None:
            if expected_parent_sha256 is not None or expected_transform is not None:
                raise EvaluationError("expected request transform has no derivation manifest")
            return None
    else:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError("derivation manifest could not be read") from exc
    if not isinstance(value, dict):
        raise EvaluationError("derivation manifest must be a JSON object")
    derived_hash = value.get("derived_request_sha256", value.get("transformed_body_sha256"))
    parent_hash = value.get("parent_request_sha256", value.get("parent_raw_request_sha256"))
    if derived_hash != capsule["request_sha256"]:
        raise EvaluationError("derivation manifest does not identify the replayed request")
    if expected_parent_sha256 and parent_hash != expected_parent_sha256:
        raise EvaluationError("derivation manifest parent hash does not match the expected source")
    if expected_transform is not None:
        expected_diff = EXPECTED_TRANSFORMS.get(expected_transform)
        if expected_diff is None:
            raise EvaluationError("unsupported expected request transform")
        if (
            value.get("semantic_transform") is not True
            or value.get("production_compatible") is not False
            or value.get("diff") != expected_diff
        ):
            raise EvaluationError("request derivation differs from the authorized transform")
        _verify_request_transform(capsule, expected_transform)
    elif expected_parent_sha256 and (
        value.get("semantic_transform") is not True
        or value.get("production_compatible") is not False
    ):
        raise EvaluationError("derived request is not marked exploratory and non-production")
    return value


def _verify_request_transform(capsule: dict[str, Any], transform: str) -> None:
    """Compare the derived JSON request with the exact expected edits to its parent body."""
    provenance = capsule.get("derivation")
    if not isinstance(provenance, dict):
        raise EvaluationError("request transform is missing capsule derivation metadata")
    parent_id = provenance.get("parent_capsule_id")
    if not isinstance(parent_id, str) or not parent_id:
        raise EvaluationError("request transform is missing its parent capsule identity")
    capsule_path = Path(capsule["capsule_path"]).resolve(strict=True)
    parent_path = capsule_path.parent / f"{parent_id}.capsule"
    try:
        parent_manifest = json.loads((parent_path / "manifest.json").read_text(encoding="utf-8"))
        derived_manifest = json.loads((capsule_path / "manifest.json").read_text(encoding="utf-8"))
        parent_request = parent_manifest["request"]
        derived_request = derived_manifest["request"]
        parent_body = (parent_path / parent_request["body_file"]).read_bytes()
        derived_body = (capsule_path / derived_request["body_file"]).read_bytes()
        parent_value = json.loads(parent_body)
        derived_value = json.loads(derived_body)
    except (OSError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("parent or derived request body could not be verified") from exc
    parent_sha256 = hashlib.sha256(parent_body).hexdigest()
    expected_parent_sha256 = provenance.get(
        "parent_request_sha256", provenance.get("parent_raw_request_sha256")
    )
    if parent_sha256 != expected_parent_sha256:
        raise EvaluationError("parent request body hash differs from derivation provenance")
    expected_value = copy.deepcopy(parent_value)
    if transform == "preserve-false":
        if "preserve_thinking" in expected_value:
            raise EvaluationError("parent already contains preserve_thinking")
        expected_value["preserve_thinking"] = False
    elif transform in {"thinking-summarized", "thinking-summarized-effort-xhigh"}:
        if (not isinstance(expected_value.get("thinking"), dict)
                or expected_value["thinking"].get("display") != "omitted"):
            raise EvaluationError("parent thinking display differs from the declared transform")
        expected_value["thinking"]["display"] = "summarized"
        if transform == "thinking-summarized-effort-xhigh":
            if (not isinstance(expected_value.get("output_config"), dict)
                    or expected_value["output_config"].get("effort") != "high"):
                raise EvaluationError("parent output effort differs from the declared transform")
            expected_value["output_config"]["effort"] = "xhigh"
    if derived_value != expected_value:
        raise EvaluationError("derived request contains changes beyond its declared transform")


def _source_manifest(path: Path, archive_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("submission manifest could not be read") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SUBMISSION_SCHEMA:
        raise EvaluationError("unsupported submission manifest")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise EvaluationError("submission has no frozen source identity")
    archive = archive_path.resolve(strict=True)
    if _file_sha256(archive) != source.get("archive_sha256"):
        raise EvaluationError("source archive SHA-256 differs from the manifest")
    manifest["source"]["archive"] = str(archive)
    return manifest


def _validate_baseline_and_gpu(*, service_port: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    services = assert_service_units_inactive(PRODUCTION_UNITS)
    assert_ports_free(tuple(sorted({service_port, CAPTURE_PORT})))
    before = gpu_snapshot()
    if not before.get("available"):
        raise EvaluationError(f"GPU inventory failed: {before.get('error')}")
    gpus = before.get("gpus", [])
    if not any(gpu.get("index") == 0 and gpu.get("uuid") == EXPECTED_GPU_UUID for gpu in gpus):
        raise EvaluationError("the expected RTX 4090 is not visible as CUDA device 0")
    apps = _gpu_compute_apps()
    if apps is None:
        raise EvaluationError("could not verify that no other compute process uses the GPU")
    if apps:
        raise EvaluationError("refusing to run while another compute process uses the GPU")
    return before, services


def run_probe(*, manifest_path: Path, archive_path: Path, capsule_path: Path,
              expected_request_sha256: str, tools_dir: Path, research_root: Path,
              lock_path: Path, run_id: str, server_preserve_thinking: bool,
              service_port: int = RESEARCH_PORT, derivation_manifest: Path | None = None,
              expected_parent_request_sha256: str | None = None,
              expected_transform: str | None = None,
              expected_binary_sha256: str | None = None) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise EvaluationError("run id must contain only letters, digits, dot, underscore, or hyphen")
    source_manifest = _source_manifest(manifest_path, archive_path)
    capsule = inspect_capsule(capsule_path, expected_request_sha256)
    provenance = _load_provenance(
        derivation_manifest, capsule, expected_parent_request_sha256, expected_transform
    )

    root = research_root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    replay_import_package = _prepare_flat_replay_import_package(root, tools_dir)
    python_paths = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if str(root) not in python_paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(root), *python_paths])
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    run_dir.chmod(0o700)
    request_log = run_dir / "requests.jsonl"
    service_log = run_dir / "serve.log"
    result_path = root / "results" / f"{run_id}.json"

    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "single-capsule-compatibility-diagnostic",
        "run_id": run_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "compatibility": {
            "production_compatible": False,
            "request_semantic_transform": provenance is not None,
            "request_provenance": provenance,
            "server_preserve_thinking": server_preserve_thinking,
            "server_profile_delta": ([] if server_preserve_thinking else [
                "omitted --preserve-thinking; all other baseline settings retained"
            ]),
        },
        "capsule": capsule,
        "submission": {
            "submission_id": source_manifest.get("submission_id"),
            "source": source_manifest.get("source"),
            "artifact": source_manifest.get("artifact"),
            "runtime_id": source_manifest.get("runtime_id"),
        },
        "paths": {"run_dir": str(run_dir), "request_log": str(request_log),
                  "server_log": str(service_log)},
        "replay_import_package": replay_import_package,
    }

    service: ManagedProcess | None = None
    sampler: GpuSampler | None = None
    gpu_before: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    with GpuLease(lock_path):
        try:
            gpu_before, services = _validate_baseline_and_gpu(service_port=service_port)
            result["preflight"] = {
                "services": services,
                "gpu_before": gpu_before,
                "compute_apps_before": [],
            }
            artifact_verification = _verify_artifact(source_manifest["artifact"])
            source_dir, build_dir, build = _source_and_build(
                root, archive_path.resolve(), source_manifest, run_dir
            )
            if expected_binary_sha256 and build["binary_sha256"] != expected_binary_sha256:
                raise EvaluationError("reused ninfer-serve binary SHA-256 differs from expected identity")
            settings = _expected_settings(source_manifest)
            settings["preserve_thinking"] = bool(server_preserve_thinking)
            server_command = _server_command(
                Path(build["binary"]), source_manifest["artifact"]["path"],
                request_log, settings, service_port,
            )
            result["artifact_verification"] = artifact_verification
            result["source_dir"] = str(source_dir)
            result["build_dir"] = str(build_dir)
            result["build"] = build
            result["effective_runtime_settings"] = settings
            result["server_command"] = server_command
            result["identity"] = {
                "source_commit": source_manifest["source"].get("git_commit"),
                "source_snapshot_sha256": source_manifest["source"].get("snapshot_sha256"),
                "source_archive_sha256": source_manifest["source"].get("archive_sha256"),
                "artifact_sha256": artifact_verification["sha256"],
                "runtime_id": source_manifest.get("runtime_id"),
                "build_binary_sha256": build["binary_sha256"],
                "server_model_id": MODEL_ID,
                "effective_runtime_settings": settings,
                "cache_recipe_id": CACHE_RECIPE_ID,
                "request_sha256": capsule["request_sha256"],
            }

            sampler = GpuSampler()
            sampler.start()
            environment = _sanitized_environment()
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            service = ManagedProcess.start(server_command, log_path=service_log,
                                           cwd=source_dir, env=environment)
            result["server_pid"] = service.pid
            health = wait_for_health(f"http://{SERVICE_HOST}:{service_port}", service,
                                     timeout_seconds=900, poll_seconds=1)
            result["health"] = health
            result["server_start"] = _server_start_record(request_log)
            if result["server_start"] is None:
                raise EvaluationError("healthy server has no startup telemetry")

            descriptor = {
                "schema_version": 1,
                "capsule_id": capsule["capsule_id"],
                "capsule_path": capsule["capsule_path"],
                "request_sha256": capsule["request_sha256"],
                "output_budget": capsule["max_tokens"],
                "metadata": {"cache_state": capsule["cache_state"]},
            }
            descriptor_path = run_dir / "descriptor.jsonl"
            _append_jsonl(descriptor_path, descriptor)
            plan = {
                "plan_id": f"compat-{run_id}",
                "cache_recipe_id": CACHE_RECIPE_ID,
                "selected": [{
                    "capsule_id": capsule["capsule_id"],
                    "capsule_path": capsule["capsule_path"],
                    "request_sha256": capsule["request_sha256"],
                    "capture_output_budget": capsule["max_tokens"],
                }],
            }
            _write_json(run_dir / "quick-plan.json", plan)
            try:
                quick_results_path = _run_plan(
                    tools_dir / "quick_eval.py", tools_dir, plan, descriptor_path,
                    request_log, run_dir, sampler, result["identity"],
                )
                result["replay"] = _read_jsonl(quick_results_path)
                result["outcome"] = "accepted"
                result["status"] = "completed"
            except RequestRejectedError as exc:
                result["replay"] = {
                    "http_status": exc.status,
                    "response_error": exc.response_error,
                    "complete_response_body_retained": False,
                }
                result["outcome"] = "request_rejected"
                result["status"] = "completed"
            result["request_events"] = _request_events(request_log)
        except BaseException as exc:
            result["status"] = "failed"
            result["failure"] = _classify_failure(exc)
            result["request_events"] = _request_events(request_log)
            if isinstance(exc, KeyboardInterrupt):
                raise
        finally:
            service_cleanup = None
            if service is not None:
                service_cleanup = service.terminate_group(
                    timeout_seconds=20 if service.is_running() else 2
                )
            gpu_samples = sampler.stop() if sampler is not None else []
            if gpu_before is not None and service is not None:
                gpu_release = _wait_gpu_release(gpu_before, service.pid)
            else:
                gpu_release = {"process_query_available": None, "service_pid_present": None,
                               "memory_returned": None, "snapshot": gpu_snapshot()}
            listener_stopped = _listener_stopped(service_port)
            cleanup = {
                "service": service_cleanup,
                "listener_stopped": listener_stopped,
                "gpu_release": gpu_release,
                "gpu_samples": gpu_samples,
                "clean": (
                    service_cleanup is not None
                    and not service_cleanup.get("process_group_alive", True)
                    and listener_stopped
                    and gpu_release.get("process_query_available") is True
                    and gpu_release.get("service_pid_present") is False
                    and gpu_release.get("memory_returned") is True
                ),
            }
            result["cleanup"] = cleanup
            if result.get("status") == "completed" and not cleanup["clean"]:
                result["status"] = "cleanup_failed"
            result["completed_at_utc"] = _utc_now()
            _write_json(result_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--capsule", type=Path, required=True)
    result.add_argument("--expected-request-sha256", required=True)
    result.add_argument("--tools-dir", type=Path, required=True)
    result.add_argument("--research-root", type=Path, required=True)
    result.add_argument("--lock-path", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--server-preserve-thinking", choices=("on", "off"), required=True)
    result.add_argument("--service-port", type=int, default=RESEARCH_PORT)
    result.add_argument("--derivation-manifest", type=Path)
    result.add_argument("--expected-parent-request-sha256")
    result.add_argument("--expected-transform", choices=tuple(EXPECTED_TRANSFORMS))
    result.add_argument("--expected-binary-sha256")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run_probe(
            manifest_path=args.manifest,
            archive_path=args.archive,
            capsule_path=args.capsule,
            expected_request_sha256=args.expected_request_sha256,
            tools_dir=args.tools_dir.resolve(),
            research_root=args.research_root,
            lock_path=args.lock_path,
            run_id=args.run_id,
            server_preserve_thinking=args.server_preserve_thinking == "on",
            service_port=args.service_port,
            derivation_manifest=args.derivation_manifest,
            expected_parent_request_sha256=args.expected_parent_request_sha256,
            expected_transform=args.expected_transform,
            expected_binary_sha256=args.expected_binary_sha256,
        )
        print(json.dumps({
            "result_path": str(args.research_root.resolve() / "results" / f"{args.run_id}.json"),
            "status": result.get("status"),
            "outcome": result.get("outcome"),
            "failure": result.get("failure"),
        }, sort_keys=True))
        return 0 if result.get("status") == "completed" else 1
    except (OSError, ValueError, EvaluationError, json.JSONDecodeError) as exc:
        print(f"compat_replay.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
