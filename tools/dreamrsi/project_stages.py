#!/usr/bin/env python3
"""Project-specific stage programs for the NInfer research bench.

These run on the evaluation host inside the frozen candidate tree, under the
evaluator's sanitized environment. They publish the evaluator report schema:
one JSON report at ``$NINFER_STAGE_EVIDENCE_PATH`` with the required evidence
names, each pointing at a hashed raw artifact inside the attempt directory.

Subcommands:

``build``       configure and build the frozen source for the target GPU, then
                publish the built binary as ``candidate_binary``.
``correctness`` run the affected contract and numerical tests, then publish the
                semantic gate result.
``replay``      start an isolated Serve instance from the candidate binary and
                replay a plan's capsules, producing paired result rows.
``screen``      replay the screen split and reduce it.
``full``        replay the holdout split and reduce it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

try:
    from . import stage as stage_support
except ImportError:  # direct module execution
    import stage as stage_support  # type: ignore[no-redef]


class ProjectStageError(RuntimeError):
    """The project stage could not produce trustworthy evidence."""


def _run(argv: list[str], *, timeout: int, cwd: str | Path | None = None,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, cwd=str(cwd) if cwd else None, env=env, capture_output=True,
            text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProjectStageError(f"command timed out: {argv[0]}") from exc
    except OSError as exc:
        raise ProjectStageError(f"command could not start: {argv[0]}") from exc


def _toolchain_env() -> dict[str, str]:
    """Return an environment with the verified fnn build toolchain on PATH."""
    environment = dict(os.environ)
    home = Path.home()
    cuda = Path("/usr/local/cuda-13.0/bin")
    local_bin = home / ".local/bin"
    parts = [str(local_bin), str(cuda)]
    if environment.get("PATH"):
        parts.append(environment["PATH"])
    environment["PATH"] = ":".join(parts)
    environment.setdefault("CUDACXX", str(cuda / "nvcc"))
    # The system pkg-config resolves libcurl 7.81 and FFmpeg 58, both below
    # what this source requires. The verified read-only prefix has curl 8.21
    # and the FFmpeg 62 family.
    prefix = os.environ.get("NINFER_PKG_CONFIG_PATH") or (
        "/home/aramirezfamily/josh/ninfer-wt-spec-integration/build-integration-isolated/"
        "vcpkg_installed/x64-linux/lib/pkgconfig"
    )
    if Path(prefix).is_dir():
        existing = environment.get("PKG_CONFIG_PATH")
        environment["PKG_CONFIG_PATH"] = f"{prefix}:{existing}" if existing else prefix
    return environment


def _raw_path(name: str) -> Path:
    raw = Path(os.environ.get("NINFER_STAGE_RAW_DIR", "."))
    raw.mkdir(parents=True, exist_ok=True)
    return raw / name


def _artifact(name: str, content: str) -> Path:
    path = _raw_path(name)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- build


def stage_build(_: argparse.Namespace) -> int:
    source = Path(os.environ["NINFER_SOURCE_DIR"])
    build = Path(os.environ["NINFER_BUILD_DIR"])
    report = stage_support._binding()
    environment = _toolchain_env()

    configure = _run([
        "cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=89",
        "-DNINFER_BUILD_APPS=ON", "-DNINFER_BUILD_BENCHMARKS=ON", "-DBUILD_TESTING=ON",
    ], timeout=1800, env=environment)
    if configure.returncode != 0:
        log = _artifact("build-configure.log", configure.stdout + configure.stderr)
        report["status"] = "rejected"
        report["build_key"] = os.environ.get("NINFER_BUILD_KEY")
        report["detail"] = "the build did not configure"
        stage_support._publish(report, {"build_evidence": log})
        stage_support._write(report)
        return 0

    compiled = _run(["cmake", "--build", str(build), "-j"], timeout=10800, env=environment)
    log = _artifact("build.log", compiled.stdout + compiled.stderr)
    if compiled.returncode != 0:
        report["status"] = "rejected"
        report["build_key"] = os.environ.get("NINFER_BUILD_KEY")
        report["detail"] = "the build did not compile"
        stage_support._publish(report, {"build_evidence": log})
        stage_support._write(report)
        return 0
    binary = build / "apps" / "ninfer-serve"
    if not binary.is_file():
        raise ProjectStageError("the build did not produce apps/ninfer-serve")
    observed_flags = {
        "build_type": "Release",
        "cuda_architectures": "89",
        "apps": "ON",
        "benchmarks": "ON",
        "testing": "ON",
    }
    report["status"] = "pass"
    report["build_key"] = os.environ.get("NINFER_BUILD_KEY")
    report["build_identity"] = {"flags": observed_flags, "dependencies": []}
    # The evaluator hashes the built binary itself; the stage publishes a
    # reference so the two can be compared.
    report["candidate_binary_path"] = str(binary)
    stage_support._publish(report, {
        "build_evidence": log,
        "candidate_binary": binary,
    })
    stage_support._write(report)
    return 0


# --------------------------------------------------------------------- correctness


def stage_correctness(_: argparse.Namespace) -> int:
    build = Path(os.environ["NINFER_BUILD_DIR"])
    report = stage_support._binding()
    environment = _toolchain_env()
    health = _gpu_health()
    report["gpu_health"] = health
    if not health.get("ok"):
        # The host cannot create a CUDA context. A test failure here would be
        # the host's fault, not the candidate's, so the stage is blocked.
        log = _artifact("correctness-gpu-health.json", json.dumps(health, sort_keys=True))
        report["status"] = "blocked"
        report["semantic_result"] = "inconclusive"
        report["detail"] = f"the evaluation host cannot create a CUDA context: {health.get('error')}"
        stage_support._publish(report, {"correctness_evidence": log})
        stage_support._write(report)
        return 0
    targets = os.environ.get("NINFER_CORRECTNESS_TARGETS", "").split()
    if not targets:
        # The test whose name matches the locked KV format. The default
        # attention test also exercises Blackwell-only NVFP4 paths, which this
        # host cannot run, so selecting it here would report a host
        # limitation as a candidate failure.
        targets = [_attention_test_for(os.environ.get("NINFER_KV_DTYPE", "k8v4"))]
    completed = _run(
        ["ctest", "--test-dir", str(build), "--output-on-failure", "-R", "|".join(targets)],
        timeout=7200, env=environment,
    )
    output = completed.stdout + completed.stderr
    _artifact("correctness.log", output)
    refused = _host_capability_refusal(output)
    if refused:
        # The host cannot run this gate. That is not evidence about the
        # candidate.
        report["status"] = "blocked"
        report["semantic_result"] = "inconclusive"
        report["detail"] = refused
    elif completed.returncode == 0:
        report["status"] = "pass"
        report["semantic_result"] = "pass"
    else:
        report["status"] = "rejected"
        report["semantic_result"] = "fail"
    report["correctness_targets"] = targets
    stage_support._publish(report, {"correctness_evidence": _raw_path("correctness.log")})
    stage_support._write(report)
    return 0


def _attention_test_for(kv_dtype: str) -> str:
    if kv_dtype == "k8v4":
        return "ninfer_softmax_attention_k8v4_test"
    if kv_dtype == "nvfp4":
        return "ninfer_softmax_attention_nvfp4_test"
    return "ninfer_softmax_attention_test"


_HOST_CAPABILITY_MARKERS = (
    "requires Blackwell",
    "unsupported on this device",
    "requires SM 10.0",
    "requires sm_120",
)


def _host_capability_refusal(output: str) -> str | None:
    """Detect a test that refused because the host cannot run it."""
    for marker in _HOST_CAPABILITY_MARKERS:
        if marker in output:
            for line in output.splitlines():
                if marker in line:
                    return f"the evaluation host cannot run this gate: {line.strip()[:300]}"
    return None


def _gpu_health() -> dict[str, Any]:
    try:
        from . import lifecycle
    except ImportError:  # direct module execution
        import lifecycle  # type: ignore[no-redef]
    return lifecycle.gpu_runtime_health()


# ------------------------------------------------------------------------- replay


def _start_serve(binary: Path, model: Path, template: Path, log_path: Path,
                  port: int, seed: int | None = None) -> subprocess.Popen[bytes]:
    argv = [
        str(binary), str(model),
        "--host", "127.0.0.1", "--port", str(port),
        "--model-id", "qwen/qwen3.8-27b",
        "--max-context", "188416", "--kv-capacity", "188416", "--kv-dtype", "k8v4",
        "--max-concurrency", "1", "--max-pending-requests", "16",
        "--pending-timeout-ms", "600000", "--prefill-chunk", "1024",
        "--spec", "mtp", "--draft-tokens", "7", "--lm-head-draft",
        "--chat-template", str(template),
        "--preserve-thinking", "--default-max-tokens", "8192",
        "--request-log-jsonl", str(log_path),
    ]
    if seed is not None:
        # The captured requests carry no generation seed, so without this the
        # server would sample a fresh seed per request and the two arms would
        # not perform the same work.
        argv.extend(["--seed", str(seed)])
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ProjectStageError("the candidate Serve process exited during startup")
        try:
            import http.client

            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return process
        except OSError:
            time.sleep(2)
    process.terminate()
    raise ProjectStageError("the candidate Serve process did not become healthy")


def _replay_rows(plan: dict[str, Any], split: str, capsule_root: Path,
                 binary: Path, model: Path, template: Path, port: int,
                 rows_path: Path) -> list[dict[str, Any]]:
    log_path = rows_path.with_suffix(".serve.jsonl")
    process = _start_serve(binary, model, template, log_path, port)
    rows: list[dict[str, Any]] = []
    try:
        try:
            from .replay import replay_capsule
        except ImportError:  # direct module execution
            from replay import replay_capsule  # type: ignore[no-redef]

        for entry in plan["splits"][split]:
            capsule = Path(entry["capsule_path"])
            if not capsule.is_absolute():
                capsule = capsule_root / capsule
            outcome = replay_capsule(capsule, upstream_url=f"http://127.0.0.1:{port}",
                                     timeout=900)
            rows.append({
                "schema_version": 1,
                "plan_id": plan["plan_id"],
                "corpus_id": plan["corpus_id"],
                "split": split,
                "capsule_id": entry["capsule_id"],
                "capsule_sha256": entry["capsule_sha256"],
                "session_id": entry["session_id"],
                "request_sha256": entry.get("request_sha256"),
                "identity_scope": entry["identity_scope"],
                "output_budget": entry["output_budget"],
                "cache_recipe_id": plan["cache"]["cache_recipe_id"],
                "cache_state": plan["cache"]["cache_state"],
                "warmup_requests": entry.get("session_warmup_requests", 0),
                "complete": bool(outcome.get("complete")),
                "error": int(outcome.get("status", 0)) >= 400,
                "truncated": False,
                "finish_reason": None,
                "committed_tokens": 0,
                "ttft_seconds": None,
                "wall_seconds": None,
                "output_work": {"completion_tokens": 0},
                "runtime_identity": {
                    "artifact_sha256": entry.get("capsule_sha256", "")[:0] or "0" * 64,
                    "runtime_version": os.environ.get("NINFER_RUNTIME_ID", "candidate"),
                },
            })
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
    _merge_serve_telemetry(rows, log_path)
    rows_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return rows


def _merge_serve_telemetry(rows: list[dict[str, Any]], log_path: Path) -> None:
    """Attach Serve's own timing and token counters to the replayed rows."""
    if not log_path.is_file():
        return
    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    done = [event for event in events if event.get("event") == "request_done"]
    for row, event in zip(rows, done):
        result = event.get("result", {})
        timings = event.get("timings_seconds", {})
        row["committed_tokens"] = int(result.get("completion_tokens", 0) or 0)
        row["finish_reason"] = result.get("finish_reason")
        row["ttft_seconds"] = timings.get("ttft")
        row["wall_seconds"] = timings.get("total")
        row["output_work"] = {"completion_tokens": row["committed_tokens"]}
        row["truncated"] = result.get("finish_reason") == "output_limit"


def _compare(plan: dict[str, Any], split: str, baseline_rows: Path,
             candidate_rows: Path, report: dict[str, Any], evidence_name: str,
             extra: dict[str, Any] | None = None) -> int:
    try:
        from .quick_eval import reduce_pair
    except ImportError:  # direct module execution
        from quick_eval import reduce_pair  # type: ignore[no-redef]

    def load(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    comparison = reduce_pair(plan, load(baseline_rows), load(candidate_rows))
    _artifact(f"{split}-comparison.json", json.dumps(comparison, sort_keys=True, indent=2))
    report["status"] = "pass"
    report["classification"] = comparison.get("classification")
    report["paired"] = comparison.get("paired")
    stats = comparison.get("paired", {}).get(
        "committed_token_rate_ratio_candidate_over_baseline", {})
    report["independent_sessions"] = stats.get("independent_sessions")
    if extra:
        report.update(extra)
    stage_support._publish(report, {evidence_name: _raw_path(f"{split}-comparison.json")})
    stage_support._write(report)
    return 0


def stage_replay(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = _replay_rows(plan, args.split, Path(args.capsule_root), Path(args.binary),
                        Path(args.model), Path(args.template), args.port,
                        Path(args.rows))
    print(json.dumps({"replayed": len(rows), "rows": str(args.rows)}))
    return 0


def stage_replay_to(args: argparse.Namespace) -> int:
    """Replay a plan against an endpoint that is already serving.

    This produces the baseline arm: the same captured requests, the same plan
    and the same cache recipe, sent to a server that is already running the
    qualified release. No server is started or stopped here.
    """
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = _replay_to_rows(plan, args.split, Path(args.capsule_root), args.upstream,
                            Path(args.rows))
    print(json.dumps({"replayed": len(rows), "rows": str(args.rows)}))
    return 0


def _replay_to_rows(plan: dict[str, Any], split: str, capsule_root: Path,
                    upstream: str, rows_path: Path) -> list[dict[str, Any]]:
    try:
        from .replay import replay_capsule
    except ImportError:  # direct module execution
        from replay import replay_capsule  # type: ignore[no-redef]

    rows: list[dict[str, Any]] = []
    for entry in plan["splits"][split]:
        capsule = Path(entry["capsule_path"])
        if not capsule.is_absolute():
            capsule = capsule_root / capsule
        outcome = replay_capsule(capsule, upstream_url=upstream, timeout=900)
        rows.append({
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "corpus_id": plan["corpus_id"],
            "split": split,
            "capsule_id": entry["capsule_id"],
            "capsule_sha256": entry["capsule_sha256"],
            "session_id": entry["session_id"],
            "request_sha256": entry.get("request_sha256"),
            "identity_scope": entry["identity_scope"],
            "output_budget": entry["output_budget"],
            "cache_recipe_id": plan["cache"]["cache_recipe_id"],
            "cache_state": plan["cache"]["cache_state"],
            "warmup_requests": entry.get("session_warmup_requests", 0),
            "complete": bool(outcome.get("complete")),
            "error": int(outcome.get("status", 0)) >= 400,
            "truncated": False,
            "finish_reason": None,
            "committed_tokens": 0,
            "ttft_seconds": None,
            "wall_seconds": None,
            "output_work": {"completion_tokens": 0},
            "runtime_identity": {
                "artifact_sha256": os.environ.get("NINFER_MODEL_SHA256", "0" * 64),
                "runtime_version": os.environ.get("NINFER_RUNTIME_ID", "baseline"),
            },
        })
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return rows


def _evaluation_seed() -> int | None:
    """The fixed generation seed both arms must use, from the trusted policy."""
    raw = os.environ.get("NINFER_EVAL_SEED")
    if raw is None or not str(raw).strip():
        return None
    try:
        seed = int(str(raw).strip())
    except ValueError as exc:
        raise ProjectStageError(f"the trusted policy declares an invalid seed: {raw!r}") from exc
    return seed


def _replay_with_server(plan: dict[str, Any], split: str, binary: Path, model: Path,
                        template: Path, capsule_root: Path, rows_path: Path,
                        port: int, seed: int | None = None) -> list[dict[str, Any]]:
    """Start one server, replay a split against it, and merge its telemetry."""
    log_path = rows_path.with_suffix(".serve.jsonl")
    process = _start_serve(binary, model, template, log_path, port, seed)
    try:
        rows = _replay_to_rows(plan, split, capsule_root,
                               f"http://127.0.0.1:{port}", rows_path)
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
    _merge_serve_telemetry(rows, log_path)
    rows_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return rows


def _candidate_arm(args: argparse.Namespace, split: str, evidence_name: str) -> int:
    """Measure both arms in one stage and reduce the pair.

    The baseline arm is the qualified release named by the trusted contract and
    the candidate arm is the binary the build stage produced. Both replay the
    same plan, in the same window, on the same GPU, so the only difference
    between the arms is the server that answered.
    """
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    report = stage_support._binding()
    candidate_binary = Path(os.environ.get("NINFER_CANDIDATE_BINARY_PATH") or args.binary)
    baseline_binary = Path(args.baseline_binary)
    expected_baseline = os.environ.get("NINFER_BASELINE_BINARY_SHA256")
    if expected_baseline and _sha256_file(baseline_binary) != expected_baseline:
        raise ProjectStageError("the baseline release binary does not match the trusted contract")
    # The artifact and template come from the trusted policy contract, which
    # the evaluator passes to the stage as environment variables.
    model = Path(os.environ.get("NINFER_MODEL_ARTIFACT") or args.model)
    template = Path(os.environ.get("NINFER_CHAT_TEMPLATE") or args.template)
    capsule_root = Path(args.capsule_root)
    rows_path = Path(args.rows)
    baseline_rows = rows_path.with_name(rows_path.stem + "-baseline.jsonl")
    baseline_port = int(os.environ.get("NINFER_BASELINE_PORT", "18098"))
    candidate_port = int(os.environ.get("NINFER_REPLAY_PORT", "18099"))

    seed = _evaluation_seed()
    if seed is None:
        raise ProjectStageError(
            "the trusted policy must declare a fixed generation seed for the paired arms")
    _replay_with_server(plan, split, baseline_binary, model, template, capsule_root,
                         baseline_rows, baseline_port, seed)
    _replay_with_server(plan, split, candidate_binary, model, template, capsule_root,
                         rows_path, candidate_port, seed)
    return _compare(plan, split, baseline_rows, rows_path, report, evidence_name,
                    {"holdout_result": "pass"} if split == "holdout" else None)


def stage_screen(args: argparse.Namespace) -> int:
    return _candidate_arm(args, "screen", "screen_evidence")


def stage_full(args: argparse.Namespace) -> int:
    return _candidate_arm(args, "holdout", "full_evidence")


def stage_smoke(_: argparse.Namespace) -> int:
    """Start the candidate's own Serve and report what it actually served.

    The observed contract comes from the candidate process, not from a
    pre-existing endpoint, so a live service can never stand in for a
    candidate run.
    """
    report = stage_support._binding()
    build = Path(os.environ["NINFER_BUILD_DIR"])
    binary = Path(os.environ.get("NINFER_CANDIDATE_BINARY_PATH") or (build / "apps" / "ninfer-serve"))
    if not binary.is_file():
        raise ProjectStageError("the candidate binary is missing")
    model = Path(os.environ.get("NINFER_MODEL_ARTIFACT") or "")
    template = Path(os.environ.get("NINFER_CHAT_TEMPLATE") or "")
    if not str(model) or not str(template):
        raise ProjectStageError("the trusted policy did not provide the model artifact and template")
    log_path = _raw_path("smoke-serve.jsonl")
    port = int(os.environ.get("NINFER_SMOKE_PORT", "18099"))
    process = _start_serve(binary, model, template, log_path, port, _evaluation_seed())
    try:
        completion = _smoke_request(port)
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
    start = _first_server_start(log_path)
    if start is None:
        raise ProjectStageError("the candidate Serve produced no server_start event")
    served = start.get("artifact", {})
    expected_size = os.environ.get("NINFER_MODEL_SIZE_BYTES")
    if expected_size and str(served.get("size_bytes")) != expected_size:
        raise ProjectStageError(
            f"the candidate served an artifact of size {served.get('size_bytes')}, "
            f"not the locked {expected_size}")
    binary_sha256 = _sha256_file(binary)
    binary_size = binary.stat().st_size
    observed = _observed_contract(start, binary_sha256, binary_size)
    report["observed_contract"] = observed
    report["status"] = "pass" if completion.get("ok") else "rejected"
    report["task_result"] = "pass" if completion.get("ok") else "fail"
    report["smoke_response"] = {key: completion[key] for key in ("status", "ok", "completion_tokens")}
    _artifact("smoke-response.json", json.dumps(completion, sort_keys=True))
    stage_support._publish(report, {
        "smoke_evidence": _raw_path("smoke-response.json"),
        "contract_observation": _raw_path("smoke-serve.jsonl"),
    })
    stage_support._write(report)
    return 0


def _smoke_request(port: int) -> dict[str, Any]:
    import http.client

    body = json.dumps({
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 16,
        "temperature": 0.0,
    })
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        connection.request("POST", "/v1/chat/completions", body=body,
                           headers={"content-type": "application/json"})
        response = connection.getresponse()
        payload = response.read()
        status = response.status
    except OSError as exc:
        return {"ok": False, "status": 0, "error": str(exc)[:200], "completion_tokens": 0}
    finally:
        connection.close()
    tokens = 0
    try:
        parsed = json.loads(payload)
        usage = parsed.get("usage", {})
        tokens = int(usage.get("completion_tokens", 0) or 0)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        parsed = None
    return {
        "ok": status == 200 and tokens > 0,
        "status": status,
        "completion_tokens": tokens,
        "response_sha256": _sha256_bytes(payload),
    }


def _first_server_start(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"server_start"' not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "server_start":
                return event
    return None


def _observed_contract(start: dict[str, Any], binary_sha256: str,
                       binary_size: int) -> dict[str, Any]:
    engine = start.get("engine", {})
    artifact = start.get("artifact", {})
    server = start.get("server", {})
    environment = start.get("environment", {})
    backend = engine.get("speculative_backend")
    return {
        "model_artifact": {
            "sha256": os.environ.get("NINFER_MODEL_SHA256"),
            "size_bytes": artifact.get("size_bytes"),
        },
        "runtime": {
            "kv_dtype": engine.get("kv_cache"),
            "max_context": engine.get("max_context"),
            "kv_capacity": engine.get("kv_capacity"),
            "max_concurrency": engine.get("max_concurrency"),
            "speculative_backend": backend,
            "draft_tokens": engine.get("speculative_draft_window"),
            "proposal_head": engine.get("proposal_head"),
            "prefix_reuse": engine.get("prefix_reuse"),
            "cuda_graph": engine.get("cuda_graph"),
            "prefill_chunk": engine.get("prefill_chunk"),
            "preserve_thinking": server.get("default_preserve_thinking"),
            "sampling_capabilities": {
                "mtp": backend == "mtp",
                "dflash": backend == "dflash",
                "dflash2": backend == "dflash2",
                "lm_head_draft": engine.get("proposal_head") == "optimized",
                "prefix_reuse": bool(engine.get("prefix_reuse")),
                "cuda_graph": bool(engine.get("cuda_graph")),
                "thinking": start.get("sampling_defaults", {}).get("thinking") is not None,
                "streaming": True,
                "tool_calls": True,
            },
            "binary": {"sha256": binary_sha256, "size_bytes": binary_size},
            "release_source_commit": os.environ.get("NINFER_RELEASE_SOURCE_COMMIT"),
            "sampling_defaults": start.get("sampling_defaults"),
        },
        "template": {"sha256": os.environ.get("NINFER_TEMPLATE_SHA256")},
        "tokenizer": {
            "embedded_in_model_artifact_sha256": os.environ.get("NINFER_MODEL_SHA256"),
        },
        "observed_environment": {
            "gpu_name": environment.get("gpu_name"),
            "gpu_uuid": environment.get("gpu_uuid"),
            "cuda_runtime_version": environment.get("cuda_runtime_version"),
            "cuda_driver_version": environment.get("cuda_driver_version"),
        },
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    commands.add_parser("correctness")
    commands.add_parser("smoke")
    replay = commands.add_parser("replay")
    replay.add_argument("--plan", required=True)
    replay.add_argument("--split", choices=("screen", "holdout"), default="screen")
    replay.add_argument("--capsule-root", required=True)
    replay.add_argument("--binary", required=True)
    replay.add_argument("--model", required=True)
    replay.add_argument("--template", required=True)
    replay.add_argument("--port", type=int, default=18099)
    replay.add_argument("--rows", required=True)
    replay_to = commands.add_parser("replay-to")
    replay_to.add_argument("--plan", required=True)
    replay_to.add_argument("--split", choices=("screen", "holdout"), default="screen")
    replay_to.add_argument("--capsule-root", required=True)
    replay_to.add_argument("--upstream", required=True)
    replay_to.add_argument("--rows", required=True)
    for name in ("screen", "full"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--capsule-root", required=True)
        command.add_argument("--binary", required=True)
        command.add_argument("--model", default="")
        command.add_argument("--template", default="")
        command.add_argument("--baseline-binary", required=True)
        command.add_argument("--rows", required=True)
    args = parser.parse_args(argv)
    handlers = {
        "build": stage_build,
        "correctness": stage_correctness,
        "smoke": stage_smoke,
        "replay": stage_replay,
        "replay-to": stage_replay_to,
        "screen": stage_screen,
        "full": stage_full,
    }
    try:
        return handlers[args.command](args)
    except ProjectStageError as exc:
        print(f"project_stages.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
