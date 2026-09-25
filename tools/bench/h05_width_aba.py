#!/usr/bin/env python3
"""Run and summarize the NON-QUALIFYING public-fixture K7-A / K3 / K7-B probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tools/bench/fixtures/h05_public_engineering.json"
EXPECTED_FIXTURE_SHA256 = "0a689c571de0290078730b0378fb2a389abac2b29c90e937eca04cc0bc34686c"
RUN_SCHEMA = "ninfer_h05_width_aba_run.v1"
ANALYSIS_SCHEMA = "ninfer_h05_width_aba_analysis.v1"
SERVER_LOG_ARTIFACT = "ninfer_serve_request_log"
SERVER_LOG_SCHEMA = 21
MODEL_ID = "qwen3.8-27b"
SEEDS = (
    7632647173703958409,
    7968175640111700217,
    912910298659544128,
)
ARMS = (("k7-a", 7), ("k3", 3), ("k7-b", 7))
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.0,
}
ENGINE_CONFIG = {
    "device": 0,
    "max_context": 188416,
    "kv_capacity": 188416,
    "max_concurrency": 1,
    "prefill_chunk": 1024,
    "kv_cache": "k8v4",
    "cuda_graph": True,
    "prefix_reuse": False,
    "speculative_backend": "mtp",
    "proposal_head": "optimized",
    "vision": False,
}
GPU_SAMPLE_FIELDS = (
    "name",
    "uuid",
    "temperature.gpu",
    "clocks.gr",
    "clocks.mem",
    "power.draw",
    "utilization.gpu",
    "memory.used",
)
SERVER_STARTUP_TIMEOUT = 1800.0
REQUEST_TIMEOUT = 24.0 * 60.0 * 60.0
MAX_OUTPUT_TOTAL_DELTA = 0.15
MAX_OUTPUT_PAIR_DELTA = 0.35
MAX_K7_DRIFT = 0.05
MAX_K7_PAIR_DRIFT = 0.20
MAX_K7_TEMP_DRIFT_C = 8.0
MAX_K7_CLOCK_DRIFT = 0.05


class HarnessError(RuntimeError):
    """An identity, protocol, or measurement-contract violation."""


@dataclass(frozen=True)
class Case:
    case_id: str
    messages: list[dict[str, Any]]
    thinking: bool
    max_completion_tokens: int
    tool_choice: str
    tools: list[dict[str, Any]]
    parallel_tool_calls: bool
    quality_terms: tuple[str, ...]
    expected_tool_call: dict[str, Any] | None
    source: str
    source_sha256: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read {description}: {error}") from error


def _repo_file(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise HarnessError(f"public fixture path escapes the repository: {relative}") from error
    return path


def load_cases(fixture_path: Path = FIXTURE_PATH) -> tuple[list[Case], dict[str, Any]]:
    fixture_path = fixture_path.resolve()
    if fixture_path == FIXTURE_PATH.resolve() and sha256_file(fixture_path) != EXPECTED_FIXTURE_SHA256:
        raise HarnessError("H05 public fixture hash changed")
    fixture = _read_json(fixture_path, "H05 fixture manifest")
    if (
        fixture.get("artifact_type") != "ninfer_h05_public_engineering_fixture"
        or fixture.get("schema_version") != 1
        or fixture.get("classification") != "NON-QUALIFYING"
    ):
        raise HarnessError("unsupported or qualifying H05 fixture manifest")

    examples_path = _repo_file(fixture.get("source_manifest", ""))
    if sha256_file(examples_path) != fixture.get("source_manifest_sha256"):
        raise HarnessError("checked-in public examples manifest hash changed")
    examples = _read_json(examples_path, "public examples manifest")
    examples_by_name = {
        item.get("name"): item for item in examples.get("cases", []) if isinstance(item, dict)
    }

    cases: list[Case] = []
    for item in fixture.get("cases", []):
        if not isinstance(item, dict):
            raise HarnessError("H05 cases must be JSON objects")
        case_id = item.get("case_id")
        source_sha = item.get("source_messages_sha256")
        if "source_case" in item:
            source_case = examples_by_name.get(item["source_case"])
            if source_case is None:
                raise HarnessError(f"unknown public source case for {case_id!r}")
            manifest_message_path = (
                examples_path.parent / str(source_case.get("messages", ""))
            ).resolve()
            try:
                manifest_message_relative = str(manifest_message_path.relative_to(REPO_ROOT))
            except ValueError as error:
                raise HarnessError("public source manifest path escapes the repository") from error
            if manifest_message_relative != item.get("source_messages"):
                raise HarnessError(f"public source path changed for {case_id!r}")
            if source_case.get("messages_sha256") != source_sha:
                raise HarnessError(f"public source manifest hash changed for {case_id!r}")
            if bool(source_case.get("thinking")) != bool(item.get("thinking")):
                raise HarnessError(f"thinking mode changed for {case_id!r}")
            if source_case.get("max_new") != item.get("max_completion_tokens"):
                raise HarnessError(f"output limit changed for {case_id!r}")
            required = source_case.get("oracle", {}).get("requirements", [])
            if list(item.get("quality_terms", [])) != required:
                raise HarnessError(f"public quality terms changed for {case_id!r}")
            source_path = _repo_file(item["source_messages"])
            raw = source_path.read_bytes()
            if sha256_bytes(raw) != source_sha:
                raise HarnessError(f"public message fixture hash changed for {case_id!r}")
            messages = json.loads(raw.decode("utf-8"))
            source = item["source_messages"]
        else:
            messages = item.get("messages")
            source = f"{fixture_path.relative_to(REPO_ROOT)}#cases/{case_id}/messages"
            if not isinstance(messages, list) or not messages:
                raise HarnessError(f"inline public messages are invalid for {case_id!r}")
            source_sha = canonical_sha256(messages)

        if not isinstance(case_id, str) or not case_id:
            raise HarnessError("case_id must be a non-empty string")
        if not isinstance(messages, list) or not messages:
            raise HarnessError(f"messages are invalid for {case_id!r}")
        tools = item.get("tools", [])
        expected_tool = item.get("expected_tool_call")
        if not isinstance(tools, list) or (expected_tool is not None and not tools):
            raise HarnessError(f"tool definitions are invalid for {case_id!r}")
        if expected_tool is not None and item.get("tool_choice") != "auto":
            raise HarnessError(f"tool case {case_id!r} must use tool_choice=auto")
        cases.append(
            Case(
                case_id=case_id,
                messages=messages,
                thinking=bool(item.get("thinking")),
                max_completion_tokens=int(item["max_completion_tokens"]),
                tool_choice=str(item["tool_choice"]),
                tools=tools,
                parallel_tool_calls=bool(item["parallel_tool_calls"]),
                quality_terms=tuple(str(term) for term in item.get("quality_terms", [])),
                expected_tool_call=expected_tool,
                source=source,
                source_sha256=str(source_sha),
            )
        )

    expected_case_ids = ("async_gpu_job_runner", "cuda_toolchain_lookup")
    if tuple(case.case_id for case in cases) != expected_case_ids:
        raise HarnessError("H05 fixture case identity or order changed")
    return cases, fixture


def request_payload(case: Case, seed: int, *, warmup: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": case.messages,
        "max_completion_tokens": 8 if warmup else case.max_completion_tokens,
        **SAMPLING,
        "seed": seed,
        "stream": False,
        "enable_thinking": case.thinking,
        "tool_choice": case.tool_choice,
        "tools": case.tools,
        "parallel_tool_calls": case.parallel_tool_calls,
    }
    return payload


def request_schedule(cases: Sequence[Case]) -> list[tuple[Case, int]]:
    by_id = {case.case_id: case for case in cases}
    schedule: list[tuple[Case, int]] = []
    for index, seed in enumerate(SEEDS):
        ordered_ids = (
            ("async_gpu_job_runner", "cuda_toolchain_lookup")
            if index % 2 == 0
            else ("cuda_toolchain_lookup", "async_gpu_job_runner")
        )
        schedule.extend((by_id[case_id], seed) for case_id in ordered_ids)
    return schedule


def _git_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = git("status", "--porcelain", "--untracked-files=all")
        return {
            "revision": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "clean": not bool(status),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        raise HarnessError(f"cannot establish source revision: {error}") from error


def _cmake_cache_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise HarnessError(f"CMake cache not found: {path}")
    selected: dict[str, str] = {}
    relevant = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_GENERATOR",
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.partition(":")[0] in relevant:
            selected[key.partition(":")[0]] = value
    return {"path": str(path), "sha256": sha256_file(path), "selected": selected}


def _probe_decode_research_flag(server: Path) -> bool:
    try:
        result = subprocess.run(
            [str(server), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--decode-research-jsonl" in (result.stdout + result.stderr)


def _median(values: Iterable[float]) -> float | None:
    items = list(values)
    return float(statistics.median(items)) if items else None


def _number(value: str) -> float | int | str:
    cleaned = value.strip()
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def gpu_sample() -> dict[str, Any]:
    captured = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_SAMPLE_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=True,
        )
        rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != len(GPU_SAMPLE_FIELDS):
            return {"captured_at_utc": captured, "available": False, "error": "shape"}
        return {
            "captured_at_utc": captured,
            "available": True,
            **{name: _number(value) for name, value in zip(GPU_SAMPLE_FIELDS, rows[0])},
        }
    except (OSError, subprocess.SubprocessError):
        return {"captured_at_utc": captured, "available": False, "error": "unavailable"}


class JsonlTail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.buffer = b""
        self.pending: list[dict[str, Any]] = []

    def _read_new(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as source:
            source.seek(self.offset)
            chunk = source.read()
            self.offset = source.tell()
        if not chunk:
            return
        self.buffer += chunk
        lines = self.buffer.split(b"\n")
        self.buffer = lines.pop()
        for raw in lines:
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise HarnessError(f"invalid server JSONL event in {self.path}: {error}") from error
            if not isinstance(event, dict):
                raise HarnessError(f"server JSONL event is not an object in {self.path}")
            self.pending.append(event)

    def wait_for(self, predicate: Any, description: str, process: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            self._read_new()
            for index, event in enumerate(self.pending):
                if predicate(event):
                    return self.pending.pop(index)
            if process.poll() is not None:
                raise HarnessError(f"Serve exited while waiting for {description}")
            if time.monotonic() >= deadline:
                raise HarnessError(f"timed out waiting for {description}")
            time.sleep(0.05)

    def next_transaction(
        self, process: subprocess.Popen[bytes], timeout: float = REQUEST_TIMEOUT
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        events: list[dict[str, Any]] = []
        terminals = {"request_done", "request_error", "request_rejected"}
        while True:
            self._read_new()
            if self.pending:
                event = self.pending.pop(0)
                events.append(event)
                if event.get("event") in terminals:
                    return events
                continue
            if process.poll() is not None:
                raise HarnessError("Serve exited before writing a request terminal")
            if time.monotonic() >= deadline:
                raise HarnessError("timed out waiting for a request terminal")
            time.sleep(0.05)


class RunningServer:
    def __init__(self, command: Sequence[str], port: int, log_path: Path) -> None:
        self.command = list(command)
        self.port = port
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.stderr_file: Any = None
        self.tail = JsonlTail(log_path)

    def __enter__(self) -> "RunningServer":
        self.stderr_file = self.log_path.with_suffix(".stderr.log").open("xb")
        self.process = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=self.stderr_file,
            stderr=subprocess.STDOUT,
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.stderr_file is not None:
            self.stderr_file.close()

    def wait_until_ready(self) -> tuple[dict[str, Any], str]:
        assert self.process is not None
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
        while True:
            if self.process.poll() is not None:
                raise HarnessError("Serve exited during startup; inspect the private stderr log")
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2.0)
            try:
                connection.request("GET", "/health", headers={"Connection": "close"})
                response = connection.getresponse()
                body = response.read()
                if response.status == 200 and json.loads(body) == {"status": "ok"}:
                    break
            except (OSError, http.client.HTTPException, json.JSONDecodeError):
                pass
            finally:
                connection.close()
            if time.monotonic() >= deadline:
                raise HarnessError("timed out waiting for Serve health")
            time.sleep(0.2)
        start = self.tail.wait_for(
            lambda event: event.get("event") == "server_start",
            "server_start event",
            self.process,
            60.0,
        )
        instance_id = start.get("server_instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise HarnessError("server_start has no server_instance_id")
        return start, instance_id

    def post(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
        assert self.process is not None
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=REQUEST_TIMEOUT
        )
        response_json: dict[str, Any] | None = None
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=encoded,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(encoded)),
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            status = response.status
            response_body = response.read()
            if status == 200:
                parsed = json.loads(response_body)
                if not isinstance(parsed, dict):
                    raise HarnessError("chat completion response is not a JSON object")
                response_json = parsed
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            raise HarnessError(f"HTTP request failed: {type(error).__name__}") from error
        finally:
            connection.close()

        events = self.tail.next_transaction(self.process)
        return status, response_json, events


def _check_log_event(event: dict[str, Any], event_name: str) -> None:
    identity = (event.get("artifact_type"), event.get("schema_version"), event.get("event"))
    expected = (SERVER_LOG_ARTIFACT, SERVER_LOG_SCHEMA, event_name)
    if identity != expected:
        raise HarnessError(f"unexpected request-log identity {identity!r}")


def validate_server_start(
    event: dict[str, Any],
    *,
    artifact: Path,
    artifact_size: int,
    draft_tokens: int,
    port: int | None = None,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    _check_log_event(event, "server_start")
    if Path(event.get("artifact", {}).get("path", "")).resolve() != artifact.resolve():
        raise HarnessError("Serve loaded a different artifact path")
    if event.get("artifact", {}).get("size_bytes") != artifact_size:
        raise HarnessError("Serve loaded an artifact with a different size")
    if event.get("server", {}).get("public_model_id") != model_id:
        raise HarnessError("Serve public model ID changed")
    server_config = event.get("server", {})
    if server_config.get("host") != "127.0.0.1":
        raise HarnessError("Serve host must remain loopback-only")
    if port is not None and server_config.get("port") != port:
        raise HarnessError("Serve port changed")
    if server_config.get("api_key_configured") is not False:
        raise HarnessError("unexpected server API-key configuration")
    engine = event.get("engine", {})
    expected = {**ENGINE_CONFIG, "speculative_draft_window": draft_tokens}
    actual = {name: engine.get(name) for name in expected}
    if actual != expected:
        raise HarnessError(f"Serve Engine configuration mismatch: {actual!r}")
    if engine.get("prefill_chunk") != ENGINE_CONFIG["prefill_chunk"]:
        raise HarnessError("Serve prefill chunk changed")
    if engine.get("log_stats_interval_ms") != 0:
        raise HarnessError("Serve periodic statistics logging must be disabled")
    environment = event.get("environment", {})
    if "RTX 4090" not in str(environment.get("gpu_name", "")):
        raise HarnessError("Serve is not running on an RTX 4090")
    if (environment.get("compute_capability_major"), environment.get("compute_capability_minor")) != (8, 9):
        raise HarnessError("Serve GPU compute capability is not 8.9")
    overrides = event.get("sampling_defaults", {}).get("server_overrides", {})
    for name, expected_value in SAMPLING.items():
        if overrides.get(name) != expected_value:
            raise HarnessError(f"Serve sampling override changed for {name}")
    if event.get("sampling_defaults", {}).get("greedy") is not False:
        raise HarnessError("Serve is not using positive-temperature sampling")
    return {
        "server_instance_id": event.get("server_instance_id"),
        "artifact": event.get("artifact"),
        "engine": engine,
        "sampling_defaults": event.get("sampling_defaults"),
        "environment": environment,
        "server": event.get("server"),
    }


def validate_request_start(event: dict[str, Any], payload: dict[str, Any]) -> None:
    _check_log_event(event, "request_start")
    request = event.get("request", {})
    if request.get("protocol") != "openai_chat_completions":
        raise HarnessError("request used an unexpected protocol")
    if request.get("model") != payload["model"] or request.get("stream") is not False:
        raise HarnessError("request model or streaming mode drifted")
    if request.get("requested_output_tokens") != payload["max_completion_tokens"]:
        raise HarnessError("request output cap drifted")
    if request.get("requested_output_tokens_source") != "client":
        raise HarnessError("request output cap did not come from the checked-in request")
    if request.get("enable_thinking") is not payload["enable_thinking"]:
        raise HarnessError("request thinking mode drifted")
    if request.get("tool_count") != len(payload["tools"]):
        raise HarnessError("request tool count drifted")
    if request.get("tool_choice") != payload["tool_choice"]:
        raise HarnessError("request tool choice drifted")
    if request.get("sampling") != {**SAMPLING, "seed": payload["seed"]}:
        raise HarnessError("resolved request sampling parameters or seed changed")


def _quality_facts(case: Case, response: dict[str, Any], request_done: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise HarnessError("response does not contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise HarnessError("response choice has no message")
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise HarnessError("response content is not text")
    output = content.casefold()
    term_facts = {term: term.casefold() in output for term in case.quality_terms}

    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise HarnessError("response tool_calls is not an array")
    names: list[str] = []
    valid_expected_call = False
    expected = case.expected_tool_call
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function", {})
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str):
            names.append(name)
        if expected is None or name != expected.get("name"):
            continue
        arguments = function.get("arguments")
        try:
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            continue
        valid_expected_call = decoded == expected.get("arguments")

    parsed = request_done.get("result", {}).get("tool_call_parse", {})
    structured_count = request_done.get("result", {}).get("tool_call_count", 0)
    finish = choice.get("finish_reason")
    if expected is None:
        tool_quality = len(tool_calls) == 0 and structured_count == 0
    else:
        tool_quality = (
            len(tool_calls) == 1
            and names == [expected["name"]]
            and valid_expected_call
            and finish == "tool_calls"
            and structured_count == 1
            and parsed.get("structured_call_count") == 1
            and parsed.get("fallback_reason") == "none"
        )
    complete = (
        finish in {"stop", "tool_calls"}
        and request_done.get("result", {}).get("finish_reason") in {"stop_token", "stop_string"}
    )
    return {
        "http_finish_reason": finish,
        "server_finish_reason": request_done.get("result", {}).get("finish_reason"),
        "tool_call_count": len(tool_calls),
        "tool_names": names,
        "server_tool_call_count": structured_count,
        "tool_fallback_reason": parsed.get("fallback_reason"),
        "expected_tool_behavior": bool(tool_quality),
        "quality_terms": term_facts,
        "quality_passed": all(term_facts.values()) and bool(tool_quality) and bool(complete),
        "completion_not_truncated": bool(complete),
    }


def _validate_done(
    request_start: dict[str, Any],
    request_done: dict[str, Any],
    payload: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    _check_log_event(request_done, "request_done")
    start_id = request_start.get("request", {}).get("request_id")
    done_id = request_done.get("request", {}).get("request_id")
    if start_id != done_id:
        raise HarnessError("request_start and request_done identities differ")
    validate_request_start(request_start, payload)
    request = request_done.get("request", {})
    if request.get("sampling") != {**SAMPLING, "seed": payload["seed"]}:
        raise HarnessError("request_done sampler or seed changed")
    if request.get("requested_output_tokens") != payload["max_completion_tokens"]:
        raise HarnessError("request_done output cap changed")
    if request.get("enable_thinking") is not payload["enable_thinking"]:
        raise HarnessError("request_done thinking mode changed")

    result = request_done.get("result", {})
    timings = request_done.get("timings_seconds", {})
    speculative = request_done.get("speculative", {})
    try:
        prompt_tokens = int(result["prompt_tokens"])
        completion_tokens = int(result["completion_tokens"])
        rounds = int(speculative["rounds"])
        drafted_tokens = int(speculative["drafted_tokens"])
        accepted_tokens = int(speculative["accepted_tokens"])
        decode_seconds = float(timings["decode"])
        usage = response["usage"]
    except (KeyError, TypeError, ValueError) as error:
        raise HarnessError(f"request result is missing required metrics: {error}") from error
    if (
        prompt_tokens <= 0
        or completion_tokens <= 0
        or rounds <= 0
        or decode_seconds <= 0
        or not math.isfinite(decode_seconds)
    ):
        raise HarnessError("request returned nonpositive token, round, or decode-time metrics")
    if drafted_tokens < accepted_tokens:
        raise HarnessError("accepted draft tokens exceed drafted tokens")
    if usage.get("prompt_tokens") != prompt_tokens or usage.get("completion_tokens") != completion_tokens:
        raise HarnessError("HTTP usage differs from request-log token lengths")
    if speculative.get("backend") != "mtp":
        raise HarnessError("request did not use the MTP route")
    if result.get("prefix_cache_hit_tokens") != 0 or result.get("prefix_reuse_path") != "root":
        raise HarnessError("request used an unexpected prefix/cache path")
    accepted_positions = speculative.get("accepted_per_position", [])
    if not isinstance(accepted_positions, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accepted_positions
    ):
        raise HarnessError("request accepted_per_position is invalid")
    return {
        "request_id": done_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens": max(completion_tokens - 1, 0),
        "decode_seconds": decode_seconds,
        "decode_tok_s": (completion_tokens - 1) / decode_seconds,
        "rounds": rounds,
        "tokens_per_round": (completion_tokens - 1) / rounds,
        "ms_per_round": decode_seconds * 1000.0 / rounds,
        "draft_window": speculative.get("draft_window"),
        "drafted_tokens": drafted_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance": accepted_tokens / drafted_tokens if drafted_tokens else None,
        "fallback_steps": int(speculative.get("fallback_steps", 0)),
        "accepted_per_position": accepted_positions,
    }


def _server_command(
    server: Path,
    artifact: Path,
    request_log: Path,
    decode_log: Path | None,
    port: int,
    draft_tokens: int,
) -> list[str]:
    command = [
        str(server),
        str(artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        MODEL_ID,
        "--max-context",
        str(ENGINE_CONFIG["max_context"]),
        "--kv-capacity",
        str(ENGINE_CONFIG["kv_capacity"]),
        "--max-concurrency",
        str(ENGINE_CONFIG["max_concurrency"]),
        "--prefill-chunk",
        str(ENGINE_CONFIG["prefill_chunk"]),
        "--log-stats-interval-ms",
        "0",
        "--device",
        str(ENGINE_CONFIG["device"]),
        "--request-log-jsonl",
        str(request_log),
        "--kv-dtype",
        str(ENGINE_CONFIG["kv_cache"]),
        "--no-prefix-reuse",
        "--no-thinking",
        "--spec",
        "mtp",
        "--draft-tokens",
        str(draft_tokens),
        "--lm-head-draft",
        "--temperature",
        str(SAMPLING["temperature"]),
        "--top-p",
        str(SAMPLING["top_p"]),
        "--top-k",
        str(SAMPLING["top_k"]),
        "--min-p",
        str(SAMPLING["min_p"]),
        "--presence-penalty",
        str(SAMPLING["presence_penalty"]),
        "--frequency-penalty",
        str(SAMPLING["frequency_penalty"]),
    ]
    if decode_log is not None:
        command.extend(["--decode-research-jsonl", str(decode_log)])
    return command


def _transaction_parts(events: Sequence[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    starts = [event for event in events if event.get("event") == "request_start"]
    terminals = [
        event
        for event in events
        if event.get("event") in {"request_done", "request_error", "request_rejected"}
    ]
    return (starts[0] if len(starts) == 1 else None, terminals[0] if len(terminals) == 1 else {})


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _arm_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "ok"]
    decode_tokens = sum(int(row["decode_tokens"]) for row in completed)
    decode_seconds = math.fsum(float(row["decode_seconds"]) for row in completed)
    rounds = sum(int(row["rounds"]) for row in completed)
    drafted = sum(int(row["drafted_tokens"]) for row in completed)
    accepted = sum(int(row["accepted_tokens"]) for row in completed)
    position_counts: list[int] = []
    for row in completed:
        for index, count in enumerate(row.get("accepted_per_position", [])):
            while len(position_counts) <= index:
                position_counts.append(0)
            position_counts[index] += int(count)
    gpu_after = [row.get("gpu_after", {}) for row in rows]
    gpu_after = [sample for sample in gpu_after if sample.get("available")]
    complete_gpu_after = [
        sample
        for sample in gpu_after
        if isinstance(sample.get("temperature.gpu"), (int, float))
        and isinstance(sample.get("clocks.gr"), (int, float))
        and isinstance(sample.get("uuid"), str)
        and bool(sample["uuid"])
    ]
    return {
        "request_count": len(rows),
        "successful_request_count": len(completed),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in completed),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in completed),
        "committed_decode_tokens": decode_tokens,
        "decode_seconds": decode_seconds,
        "decode_rounds": rounds,
        "drafted_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "draft_acceptance": accepted / drafted if drafted else None,
        "committed_decode_tok_s": decode_tokens / decode_seconds if decode_seconds else None,
        "committed_tokens_per_round": decode_tokens / rounds if rounds else None,
        "ms_per_round": decode_seconds * 1000.0 / rounds if rounds else None,
        "accepted_per_position_counts_aggregate_only": position_counts,
        "gpu_after_sample_count": len(gpu_after),
        "gpu_after_thermal_sample_count": len(complete_gpu_after),
        "gpu_after_thermal_samples_complete": len(complete_gpu_after) == len(rows),
        "gpu_temperature_c_median_after": _median(
            float(sample["temperature.gpu"]) for sample in complete_gpu_after
        ),
        "gpu_graphics_clock_mhz_median_after": _median(
            float(sample["clocks.gr"]) for sample in complete_gpu_after
        ),
        "gpu_memory_used_mib_peak": max(
            (float(sample["memory.used"]) for sample in gpu_after
             if isinstance(sample.get("memory.used"), (int, float))),
            default=None,
        ),
    }


def _ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left <= 0 or right <= 0:
        return None
    return left / right


def _geomean(values: Iterable[float]) -> float | None:
    items = [value for value in values if value > 0 and math.isfinite(value)]
    if not items:
        return None
    return math.exp(math.fsum(math.log(value) for value in items) / len(items))


def _h03_analyzer() -> Any:
    path = REPO_ROOT / "tools/analyze_decode_research.py"
    spec = importlib.util.spec_from_file_location("ninfer_decode_research", path)
    if spec is None or spec.loader is None:
        raise HarnessError("cannot load the reviewed H03 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_h03(path: Path, draft_tokens: int, expected_request_count: int) -> dict[str, Any]:
    analyzer = _h03_analyzer()
    events = analyzer.load_events(path)
    sessions = {event["engine_session_id"] for event in events}
    if len(sessions) != 1:
        raise HarnessError(f"H03 stream {path} contains multiple Engine sessions")
    if any(
        event["backend"] != "mtp"
        or event["draft_window"] != draft_tokens
        or event["batch_size"] != 1
        or event["cuda_graph"] is not True
        for event in events
    ):
        raise HarnessError(f"H03 stream {path} has a width, batch, or graph mismatch")
    request_ids = sorted({event["request_id"] for event in events})
    if len(request_ids) != expected_request_count:
        raise HarnessError(
            f"H03 stream has {len(request_ids)} requests; expected {expected_request_count}"
        )
    grouped = {
        request_id: [event for event in events if event["request_id"] == request_id]
        for request_id in request_ids
    }
    out_by_request: list[dict[str, Any]] = []
    for request_id in request_ids:
        rows = grouped[request_id]
        elapsed_ns = sum(row["round_elapsed_ns"] for row in rows)
        committed = sum(row["committed_count"] for row in rows)
        out_by_request.append(
            {
                "engine_request_id": request_id,
                "rounds": len(rows),
                "committed_decode_tokens": committed,
                "decode_round_seconds": elapsed_ns / 1e9,
                "committed_decode_tok_s": committed * 1e9 / elapsed_ns,
                "committed_tokens_per_round": committed / len(rows),
                "ms_per_round": elapsed_ns / len(rows) / 1e6,
            }
        )
    measured_ids = request_ids[1:]
    measured = [event for event in events if event["request_id"] in measured_ids]
    engine_rounds = {
        (event["engine_session_id"], event["engine_round_id"]): event
        for event in measured
    }
    elapsed_ns = sum(row["round_elapsed_ns"] for row in engine_rounds.values())
    committed = sum(row["committed_count"] for row in measured)
    total_rounds = len(engine_rounds)
    max_extent = max(row["draft_extent"] for row in measured)
    positions = []
    for position in range(1, max_extent + 1):
        eligible = sum(row["draft_extent"] >= position for row in measured)
        accepted = sum(row["accepted_prefix"] >= position for row in measured)
        committed_accepted = sum(
            row["committed_draft_prefix"] >= position for row in measured
        )
        positions.append(
            {
                "position": position,
                "eligible_request_rounds": eligible,
                "accepted_request_rounds": accepted,
                "survival": accepted / eligible if eligible else None,
                "committed_accepted_request_rounds": committed_accepted,
                "committed_survival": committed_accepted / eligible if eligible else None,
                "committed_accepted_per_round": (
                    committed_accepted / total_rounds if total_rounds else None
                ),
            }
        )
    return {
        "source": str(path.resolve()),
        "schema": analyzer.SCHEMA,
        "request_count_including_warmup": len(request_ids),
        "request_summaries_in_order_including_warmup": out_by_request,
        "measurement_rounds": total_rounds,
        "committed_decode_tokens": committed,
        "round_elapsed_seconds": elapsed_ns / 1e9,
        "committed_decode_tok_s": committed * 1e9 / elapsed_ns if elapsed_ns else None,
        "committed_tokens_per_round": committed / total_rounds if total_rounds else None,
        "ms_per_round": elapsed_ns / total_rounds / 1e6 if total_rounds else None,
        "draft_extent_histogram": {
            str(extent): sum(row["draft_extent"] == extent for row in measured)
            for extent in sorted({row["draft_extent"] for row in measured})
        },
        "position_survival": positions,
    }


def analyze_rows(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    expected_pairs: Sequence[str],
    h03_by_arm: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    arm_metrics = {arm: _arm_summary(rows) for arm, rows in rows_by_arm.items()}
    if tuple(rows_by_arm) != tuple(name for name, _ in ARMS):
        reasons.append("arm_order_or_identity_mismatch")

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, rows in rows_by_arm.items():
        if not arm_metrics.get(arm, {}).get("gpu_after_thermal_samples_complete"):
            reasons.append(f"missing_gpu_thermal_sample:{arm}")
        for row in rows:
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str) or pair_id in by_pair.setdefault(arm, {}):
                reasons.append(f"duplicate_or_missing_pair_id:{arm}")
                continue
            by_pair[arm][pair_id] = row
        if set(by_pair[arm]) != set(expected_pairs):
            reasons.append(f"request_set_mismatch:{arm}")

    for pair_id in expected_pairs:
        pair_rows = [by_pair.get(arm, {}).get(pair_id) for arm, _ in ARMS]
        if any(row is None for row in pair_rows):
            continue
        if len({row.get("request_body_sha256") for row in pair_rows}) != 1:
            reasons.append(f"request_body_hash_drift:{pair_id}")
        if len({row.get("prompt_tokens") for row in pair_rows}) != 1:
            reasons.append(f"prompt_length_drift:{pair_id}")
        if any(row.get("status") != "ok" for row in pair_rows):
            reasons.append(f"request_failure:{pair_id}")
        if any(not row.get("quality_passed") for row in pair_rows):
            reasons.append(f"quality_or_tool_behavior_failed:{pair_id}")
        if any(not row.get("configuration_verified") for row in pair_rows):
            reasons.append(f"request_configuration_unverified:{pair_id}")

    k3_tot = arm_metrics.get("k3", {}).get("completion_tokens", 0)
    length_deltas: dict[str, float | None] = {}
    if k3_tot:
        for arm in ("k7-a", "k7-b"):
            amount = arm_metrics.get(arm, {}).get("completion_tokens", 0)
            delta = amount / k3_tot - 1.0
            length_deltas[arm] = delta
            if abs(delta) > MAX_OUTPUT_TOTAL_DELTA:
                reasons.append(f"output_length_total_delta_exceeds_15pct:{arm}")
    else:
        length_deltas = {"k7-a": None, "k7-b": None}
        reasons.append("no_successful_k3_output_tokens")

    pair_length_deltas: dict[str, dict[str, float | None]] = {"k7-a": {}, "k7-b": {}}
    for arm in ("k7-a", "k7-b"):
        for pair_id in expected_pairs:
            baseline = by_pair.get("k3", {}).get(pair_id)
            candidate = by_pair.get(arm, {}).get(pair_id)
            if (
                baseline is None
                or candidate is None
                or not baseline.get("completion_tokens")
                or not candidate.get("completion_tokens")
            ):
                pair_length_deltas[arm][pair_id] = None
                continue
            delta = candidate["completion_tokens"] / baseline["completion_tokens"] - 1.0
            pair_length_deltas[arm][pair_id] = delta
            if abs(delta) > MAX_OUTPUT_PAIR_DELTA:
                reasons.append(f"output_length_pair_delta_exceeds_35pct:{arm}:{pair_id}")

    k7_aba_pair_lengths: dict[str, float | None] = {}
    for pair_id in expected_pairs:
        row_a = by_pair.get("k7-a", {}).get(pair_id)
        row_b = by_pair.get("k7-b", {}).get(pair_id)
        if (
            row_a is None
            or row_b is None
            or not row_a.get("completion_tokens")
            or not row_b.get("completion_tokens")
        ):
            k7_aba_pair_lengths[pair_id] = None
            continue
        delta = row_b["completion_tokens"] / row_a["completion_tokens"] - 1.0
        k7_aba_pair_lengths[pair_id] = delta
        if abs(delta) > MAX_OUTPUT_PAIR_DELTA:
            reasons.append(f"k7_aba_output_length_pair_delta_exceeds_35pct:{pair_id}")

    a = arm_metrics.get("k7-a", {})
    b = arm_metrics.get("k7-b", {})
    k3 = arm_metrics.get("k3", {})
    paired_tps_ratios: list[float] = []
    paired_round_ratios: list[float] = []
    paired_output_ratios: list[float] = []
    per_pair_k7_drift: dict[str, dict[str, float | None]] = {}
    for pair_id in expected_pairs:
        row_a = by_pair.get("k7-a", {}).get(pair_id)
        row_b = by_pair.get("k7-b", {}).get(pair_id)
        if row_a is None or row_b is None:
            continue
        pair_drift: dict[str, float | None] = {}
        for metric_name, destination, numerator, denominator in (
            (
                "decode_tok_s_ratio_b_over_a",
                paired_tps_ratios,
                row_b.get("decode_tok_s"),
                row_a.get("decode_tok_s"),
            ),
            (
                "ms_per_round_ratio_b_over_a",
                paired_round_ratios,
                row_b.get("ms_per_round"),
                row_a.get("ms_per_round"),
            ),
            (
                "completion_tokens_ratio_b_over_a",
                paired_output_ratios,
                row_b.get("completion_tokens"),
                row_a.get("completion_tokens"),
            ),
        ):
            ratio = _ratio(numerator, denominator)
            pair_drift[metric_name] = ratio
            if ratio is not None:
                destination.append(ratio)
        per_pair_k7_drift[pair_id] = pair_drift
    tps_drift = _geomean(paired_tps_ratios)
    round_drift = _geomean(paired_round_ratios)
    output_drift = _geomean(paired_output_ratios)
    max_pair_tps_drift = max(
        (abs(value - 1.0) for value in paired_tps_ratios), default=None
    )
    max_pair_round_drift = max(
        (abs(value - 1.0) for value in paired_round_ratios), default=None
    )
    temp_a = a.get("gpu_temperature_c_median_after")
    temp_b = b.get("gpu_temperature_c_median_after")
    clock_a = a.get("gpu_graphics_clock_mhz_median_after")
    clock_b = b.get("gpu_graphics_clock_mhz_median_after")
    temp_delta = temp_b - temp_a if temp_a is not None and temp_b is not None else None
    clock_ratio = _ratio(clock_b, clock_a)
    stable_k7 = (
        tps_drift is not None
        and round_drift is not None
        and abs(tps_drift - 1.0) <= MAX_K7_DRIFT
        and abs(round_drift - 1.0) <= MAX_K7_DRIFT
        and max_pair_tps_drift is not None
        and max_pair_tps_drift <= MAX_K7_PAIR_DRIFT
        and max_pair_round_drift is not None
        and max_pair_round_drift <= MAX_K7_PAIR_DRIFT
        and temp_delta is not None
        and abs(temp_delta) <= MAX_K7_TEMP_DRIFT_C
        and clock_ratio is not None
        and abs(clock_ratio - 1.0) <= MAX_K7_CLOCK_DRIFT
    )
    if not stable_k7:
        reasons.append("k7_aba_or_thermal_drift_exceeds_gate")

    h03_by_arm = h03_by_arm or {}
    h03_status = "present" if h03_by_arm else "request_aggregate_fallback"
    for arm, h03 in h03_by_arm.items():
        summary = arm_metrics.get(arm, {})
        request_summaries = h03.get("request_summaries_in_order_including_warmup", [])
        if len(request_summaries) != len(rows_by_arm.get(arm, [])) + 1:
            reasons.append(f"h03_request_mapping_mismatch:{arm}")
        for index, row in enumerate(rows_by_arm.get(arm, []), start=1):
            if index >= len(request_summaries):
                break
            if request_summaries[index].get("committed_decode_tokens") != row.get("decode_tokens"):
                reasons.append(f"h03_committed_token_mismatch:{arm}:{row.get('pair_id')}")
        if summary.get("decode_rounds") != h03.get("measurement_rounds"):
            reasons.append(f"h03_round_count_mismatch:{arm}")

    valid = not reasons
    k7_rates = [value for value in (a.get("committed_decode_tok_s"), b.get("committed_decode_tok_s")) if value]
    k7_yields = [value for value in (a.get("committed_tokens_per_round"), b.get("committed_tokens_per_round")) if value]
    k7_round_ms = [value for value in (a.get("ms_per_round"), b.get("ms_per_round")) if value]
    k7_rate = _geomean(k7_rates)
    k7_yield = _geomean(k7_yields)
    k7_ms = _geomean(k7_round_ms)
    rate_ratio = _ratio(k7_rate, k3.get("committed_decode_tok_s"))
    yield_ratio = _ratio(k7_yield, k3.get("committed_tokens_per_round"))
    round_ratio = _ratio(k7_ms, k3.get("ms_per_round"))
    if not valid or rate_ratio is None or yield_ratio is None or round_ratio is None:
        branch = "inconclusive"
    elif yield_ratio >= 1.05 and rate_ratio >= 1.05:
        branch = "acceptance-axis"
    elif round_ratio >= 1.05 and yield_ratio < 1.05 and rate_ratio <= 1.05:
        branch = "latency-axis"
    else:
        branch = "inconclusive"

    return {
        "schema": ANALYSIS_SCHEMA,
        "classification": "NON-QUALIFYING",
        "valid_for_exploratory_axis_selection": valid,
        "invalid_reasons": sorted(set(reasons)),
        "h03_evidence": {
            "status": h03_status,
            "note": (
                "Per-round extent-conditioned survival is available from H03."
                if h03_by_arm
                else "Serve did not provide H03 per-round events; request-level aggregates support coarse yield/cost triage only. Exact extent-conditioned position survival is unavailable."
            ),
            "arms": h03_by_arm,
        },
        "arm_metrics": arm_metrics,
        "paired_k7_a_to_k7_b": {
            "geometric_mean_paired_decode_tok_s_ratio_b_over_a": tps_drift,
            "geometric_mean_paired_ms_per_round_ratio_b_over_a": round_drift,
            "geometric_mean_paired_completion_tokens_ratio_b_over_a": output_drift,
            "paired_request_count": len(paired_tps_ratios),
            "max_paired_decode_tok_s_absolute_delta": max_pair_tps_drift,
            "max_paired_ms_per_round_absolute_delta": max_pair_round_drift,
            "maximum_paired_drift": MAX_K7_PAIR_DRIFT,
            "by_case_seed": per_pair_k7_drift,
            "median_after_temperature_delta_c": temp_delta,
            "graphics_clock_ratio_b_over_a": clock_ratio,
            "stable_under_predeclared_gate": stable_k7,
        },
        "output_length_controls": {
            "maximum_total_arm_delta": MAX_OUTPUT_TOTAL_DELTA,
            "maximum_paired_request_delta": MAX_OUTPUT_PAIR_DELTA,
            "k7_a_and_b_vs_k3_total_deltas": length_deltas,
            "paired_request_deltas": pair_length_deltas,
            "k7_b_vs_k7_a_paired_deltas": k7_aba_pair_lengths,
        },
        "k7_vs_k3": {
            "geometric_mean_decode_tok_s_ratio": rate_ratio,
            "geometric_mean_committed_tokens_per_round_ratio": yield_ratio,
            "geometric_mean_ms_per_round_ratio": round_ratio,
            "branch": branch,
            "criteria": {
                "acceptance_axis": "Only when all gates pass, K7 committed tokens/round and decode tok/s each exceed K3 by at least 5%.",
                "latency_axis": "Only when all gates pass, K7 ms/round is at least 5% higher, K7 yield is below +5%, and K7 decode tok/s is no more than +5%.",
                "otherwise": "inconclusive; do not promote a width or performance claim",
            },
        },
    }


def _expected_pair_ids(cases: Sequence[Case]) -> list[str]:
    return [f"{case.case_id}:{seed}" for case, seed in request_schedule(cases)]


def _write_request_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def _run_attempt(
    server: RunningServer,
    case: Case,
    seed: int,
    *,
    arm: str,
    draft_tokens: int,
    sequence: int,
    last_request_id: int,
    expected_gpu_uuid: str,
    warmup: bool = False,
) -> tuple[dict[str, Any], int]:
    payload = request_payload(case, seed, warmup=warmup)
    payload_hash = canonical_sha256(payload)
    pair_id = "warmup" if warmup else f"{case.case_id}:{seed}"
    row: dict[str, Any] = {
        "classification": "NON-QUALIFYING",
        "arm": arm,
        "sequence": sequence,
        "pair_id": pair_id,
        "case_id": case.case_id,
        "seed": seed,
        "request_body_sha256": payload_hash,
        "requested_output_tokens": payload["max_completion_tokens"],
        "status": "failed",
    }
    before = gpu_sample()
    row["gpu_before"] = before
    try:
        status, response, events = server.post(payload)
    except HarnessError as error:
        row["gpu_after"] = gpu_sample()
        row["failure"] = f"request_transport_or_log_failure:{error}"
        row["configuration_verified"] = False
        return row, last_request_id
    after = gpu_sample()
    row["gpu_after"] = after
    row["http_status"] = status
    try:
        for sample in (before, after):
            if sample.get("available") and sample.get("uuid") != expected_gpu_uuid:
                raise HarnessError("nvidia-smi sampled a different GPU than server_start")
    except HarnessError as error:
        row["failure"] = f"request_configuration_failure:{error}"
        row["configuration_verified"] = False
        return row, last_request_id
    start, terminal = _transaction_parts(events)
    if not terminal:
        row["failure"] = "missing_server_terminal"
        return row, last_request_id
    try:
        _check_log_event(terminal, str(terminal.get("event")))
        request = terminal.get("request", {})
        request_id = request.get("request_id")
        if isinstance(request_id, int) and request_id != last_request_id + 1:
            raise HarnessError(f"nonsequential server request id {request_id}")
        if isinstance(request_id, int):
            last_request_id = request_id
        row["request_id"] = request_id

        if status != 200 or terminal.get("event") != "request_done" or start is None or response is None:
            row["failure"] = f"http_{status}_{terminal.get('event', 'no_event')}"
            row["configuration_verified"] = False
            return row, last_request_id
        _check_log_event(start, "request_start")
        validate_request_start(start, payload)
        if start.get("request", {}).get("request_id") != request_id:
            raise HarnessError("request_start and terminal request IDs differ")
        metrics = _validate_done(start, terminal, payload, response)
        if metrics["draft_window"] != draft_tokens:
            raise HarnessError("request speculative width differs from the active arm")
        quality = _quality_facts(case, response, terminal)
        row.update(metrics)
        row.update(quality)
        row["configuration_verified"] = True
        row["status"] = "ok"
    except HarnessError as error:
        row["failure"] = f"request_validation_failure:{error}"
        row["configuration_verified"] = False
    return row, last_request_id


def _private_empty_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise HarnessError("output directory must not be a symlink")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        path.chmod(0o700)
        if path.stat().st_mode & 0o077:
            raise HarnessError("output directory is not owner-only")
    if any(path.iterdir()):
        raise HarnessError(f"output directory must be empty: {path}")
    return path


def _run(args: argparse.Namespace) -> int:
    server = args.serve.expanduser().resolve()
    artifact = args.artifact.expanduser().resolve()
    cache = args.cmake_cache.expanduser().resolve()
    if not server.is_file() or not artifact.is_file():
        raise HarnessError("explicit Serve binary and artifact paths must exist")
    expected_hash = args.artifact_sha256.lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise HarnessError("--artifact-sha256 must be exactly 64 hexadecimal characters")
    source = _git_identity()
    if not source["clean"]:
        raise HarnessError("source worktree must be clean for a reproducible run")
    if sha256_file(artifact) != expected_hash:
        raise HarnessError("artifact SHA-256 differs from --artifact-sha256")
    cases, fixture = load_cases()
    fixture_hash = sha256_file(FIXTURE_PATH)
    schedule = request_schedule(cases)
    decode_supported = _probe_decode_research_flag(server) if args.decode_events != "off" else False
    if args.decode_events == "required" and not decode_supported:
        raise HarnessError("Serve does not expose --decode-research-jsonl")
    if not cache.is_file():
        raise HarnessError(f"explicit CMake cache not found: {cache}")
    binary_identity = {
        "path": str(server),
        "sha256": sha256_file(server),
        "cmake_cache": _cmake_cache_identity(cache),
    }
    output = _private_empty_dir(args.output)
    if os.name == "posix":
        os.umask(0o077)

    run = {
        "schema": RUN_SCHEMA,
        "classification": "NON-QUALIFYING",
        "hypothesis": "E0002-H05",
        "status": "running",
        "source": {**source, "repository": str(REPO_ROOT)},
        "build": binary_identity,
        "artifact": {
            "path": str(artifact),
            "sha256": expected_hash,
            "size_bytes": artifact.stat().st_size,
        },
        "fixture": {
            "path": str(FIXTURE_PATH),
            "sha256": fixture_hash,
            "source_manifest_sha256": fixture.get("source_manifest_sha256"),
            "cases": [
                {
                    "case_id": case.case_id,
                    "source": case.source,
                    "source_sha256": case.source_sha256,
                    "messages_sha256": canonical_sha256(case.messages),
                    "thinking": case.thinking,
                    "max_completion_tokens": case.max_completion_tokens,
                    "tool_choice": case.tool_choice,
                    "tools_sha256": canonical_sha256(case.tools),
                    "expected_tool_call": (
                        {
                            "name": case.expected_tool_call["name"],
                            "arguments_sha256": canonical_sha256(
                                case.expected_tool_call["arguments"]
                            ),
                        }
                        if case.expected_tool_call is not None
                        else None
                    ),
                    "quality_terms": list(case.quality_terms),
                }
                for case in cases
            ],
        },
        "sampling": {**SAMPLING, "positive_temperature": True, "seed_stream": list(SEEDS)},
        "engine_config": ENGINE_CONFIG,
        "request_order": [
            {
                "sequence": index,
                "case_id": case.case_id,
                "seed": seed,
                "request_body_sha256": canonical_sha256(request_payload(case, seed)),
            }
            for index, (case, seed) in enumerate(schedule, start=1)
        ],
        "warmup": {
            "count_per_arm": 1,
            "case_id": cases[0].case_id,
            "seed": SEEDS[0],
            "max_completion_tokens": 8,
            "request_body_sha256": canonical_sha256(
                request_payload(cases[0], SEEDS[0], warmup=True)
            ),
            "measured": False,
        },
        "arm_order": [{"name": name, "draft_tokens": width} for name, width in ARMS],
        "h03_capture": {
            "status": "available" if decode_supported else "not_exposed_by_serve_cli",
            "mode": args.decode_events,
            "fallback": "request-level aggregates; exact extent-conditioned survival unavailable",
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "output_dir": str(output),
    }
    _atomic_json(output / "run.json", run)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    h03_by_arm: dict[str, dict[str, Any]] = {}
    expected_size = artifact.stat().st_size
    expected_gpu_uuid: str | None = None

    for arm_index, (arm, draft_tokens) in enumerate(ARMS):
        if sha256_file(server) != binary_identity["sha256"]:
            raise HarnessError("Serve binary changed between arms")
        if _cmake_cache_identity(cache)["sha256"] != binary_identity["cmake_cache"]["sha256"]:
            raise HarnessError("CMake cache changed between arms")
        if _git_identity() != source:
            raise HarnessError("source revision or worktree state changed between arms")
        if sha256_file(artifact) != expected_hash:
            raise HarnessError("artifact content changed between arms")
        arm_dir = output / arm
        arm_dir.mkdir(mode=0o700)
        request_log = arm_dir / "requests.jsonl"
        decode_log = arm_dir / "decode_research.jsonl" if decode_supported else None
        command = _server_command(server, artifact, request_log, decode_log, args.port, draft_tokens)
        run_path = arm_dir / "command.json"
        _atomic_json(
            run_path,
            {
                "arm": arm,
                "command": command,
                "server_binary_sha256": binary_identity["sha256"],
                "request_log": str(request_log),
                "decode_research_log": str(decode_log) if decode_log else None,
            },
        )
        rows: list[dict[str, Any]] = []
        with RunningServer(command, args.port, request_log) as running:
            server_start, _instance_id = running.wait_until_ready()
            start_summary = validate_server_start(
                server_start,
                artifact=artifact,
                artifact_size=expected_size,
                draft_tokens=draft_tokens,
                port=args.port,
            )
            gpu_uuid = start_summary["environment"].get("gpu_uuid")
            if not isinstance(gpu_uuid, str) or not gpu_uuid:
                raise HarnessError("server_start has no GPU UUID")
            if expected_gpu_uuid is None:
                expected_gpu_uuid = gpu_uuid
            elif gpu_uuid != expected_gpu_uuid:
                raise HarnessError("server arm changed physical GPU UUID")
            run.setdefault("arm_server_identities", {})[arm] = start_summary
            _atomic_json(arm_dir / "server_start.json", start_summary)
            last_request_id = 0
            warmup_row, last_request_id = _run_attempt(
                running,
                cases[0],
                SEEDS[0],
                arm=arm,
                draft_tokens=draft_tokens,
                sequence=0,
                last_request_id=last_request_id,
                expected_gpu_uuid=gpu_uuid,
                warmup=True,
            )
            if warmup_row.get("status") != "ok":
                warmup_row["classification"] = "NON-QUALIFYING"
                _write_request_row(arm_dir / "warmup.jsonl", warmup_row)
                raise HarnessError(f"{arm} warmup request failed")
            _write_request_row(arm_dir / "warmup.jsonl", warmup_row)
            for sequence, (case, seed) in enumerate(schedule, start=1):
                row, last_request_id = _run_attempt(
                    running,
                    case,
                    seed,
                    arm=arm,
                    draft_tokens=draft_tokens,
                    sequence=sequence,
                    last_request_id=last_request_id,
                    expected_gpu_uuid=gpu_uuid,
                )
                row["classification"] = "NON-QUALIFYING"
                rows.append(row)
                _write_request_row(arm_dir / "measured.jsonl", row)
                print(
                    f"{arm} {sequence}/{len(schedule)} {case.case_id} seed={seed} "
                    f"{row.get('status')} {row.get('quality_passed', False)}",
                    flush=True,
                )
                if row.get("status") != "ok":
                    raise HarnessError(
                        f"{arm} request {sequence} failed; see measured.jsonl for its body-free record"
                    )
        rows_by_arm[arm] = rows
        if decode_log is not None:
            if not decode_log.is_file():
                raise HarnessError(f"H03 capture was requested but no file was written for {arm}")
            h03_by_arm[arm] = summarize_h03(decode_log, draft_tokens, len(schedule) + 1)
        run["status"] = "running"
        run["completed_arms"] = arm_index + 1
        _atomic_json(output / "run.json", run)

    analysis = analyze_rows(
        rows_by_arm,
        expected_pairs=_expected_pair_ids(cases),
        h03_by_arm=h03_by_arm,
    )
    _atomic_json(output / "analysis.json", analysis)
    run["status"] = "complete"
    run["completed_arms"] = len(ARMS)
    run["analysis"] = "analysis.json"
    _atomic_json(output / "run.json", run)
    print(
        f"H05 {analysis['k7_vs_k3']['branch']}: "
        f"valid={analysis['valid_for_exploratory_axis_selection']} "
        f"h03={analysis['h03_evidence']['status']} output={output}",
        flush=True,
    )
    return 0 if analysis["valid_for_exploratory_axis_selection"] else 2


def _load_measured(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise HarnessError(f"invalid result JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict) or row.get("classification") != "NON-QUALIFYING":
                raise HarnessError(f"invalid or qualifying result row at {path}:{line_number}")
            rows.append(row)
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the fixed public-fixture ABA probe")
    run.add_argument("--serve", required=True, type=Path, help="explicit ninfer-serve binary")
    run.add_argument("--artifact", required=True, type=Path, help="explicit .ninfer artifact")
    run.add_argument("--artifact-sha256", required=True, help="expected artifact SHA-256")
    run.add_argument("--cmake-cache", required=True, type=Path, help="CMakeCache.txt for this binary")
    run.add_argument("--output", required=True, type=Path, help="new owner-only output directory")
    run.add_argument("--port", type=int, default=18082)
    run.add_argument("--decode-events", choices=("auto", "required", "off"), default="auto")
    return parser.parse_args(argv)


def _record_run_failure(output: Path, error: BaseException) -> None:
    run_path = output.expanduser().resolve() / "run.json"
    if not run_path.is_file():
        return
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["status"] = "failed"
        run["failure"] = str(error)
        _atomic_json(run_path, run)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "run":
        try:
            return _run(args)
        except (HarnessError, OSError, ValueError, subprocess.SubprocessError) as error:
            _record_run_failure(args.output, error)
            print(f"H05 error: {error}", file=sys.stderr)
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
