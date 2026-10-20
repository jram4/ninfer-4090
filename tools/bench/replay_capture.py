#!/usr/bin/env python3
"""Replay an ordered HTTP JSON capture against an already-running NInfer Serve."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit


_ALLOWED_CAPTURE_HEADERS = {
    "accept",
    "anthropic-beta",
    "anthropic-version",
    "content-type",
}
_METHOD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9!#$%&'*+.^_|~-]*$")
_REQUEST_EVENTS = {"request_start", "request_rejected"}
_TERMINAL_EVENTS = {"request_done", "request_error"}


@dataclass(frozen=True)
class Endpoint:
    scheme: str
    host: str
    port: int | None
    path_prefix: str


@dataclass
class ResponseMetrics:
    started_ns: int
    first_byte_ns: int | None = None
    first_token_ns: int | None = None
    output_tokens: int | None = None
    decode_tok_per_s: float | None = None
    decode_rate_source: str | None = None

    def note_first_byte(self, now_ns: int) -> None:
        if self.first_byte_ns is None:
            self.first_byte_ns = now_ns

    def note_token(self, now_ns: int) -> None:
        if self.first_token_ns is None:
            self.first_token_ns = now_ns

    def observe_json(self, value: Any) -> None:
        if not isinstance(value, dict):
            return

        usages: list[dict[str, Any]] = []
        direct_usage = value.get("usage")
        if isinstance(direct_usage, dict):
            usages.append(direct_usage)
        response = value.get("response")
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            usages.append(response["usage"])
        message = value.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            usages.append(message["usage"])

        for usage in usages:
            for key in ("output_tokens", "completion_tokens"):
                count = _nonnegative_int(usage.get(key))
                if count is not None:
                    self.output_tokens = count
                    break

        timings = value.get("timings")
        metrics = value.get("metrics")
        for candidate in (timings, metrics, value.get("timings_seconds")):
            if not isinstance(candidate, dict):
                continue
            predicted_n = _nonnegative_int(candidate.get("predicted_n"))
            if predicted_n is not None and self.output_tokens is None:
                self.output_tokens = predicted_n
            for key in (
                "predicted_per_second",
                "decode_tok_per_s",
                "decode_tokens_per_second",
                "decode_tokens_per_s",
            ):
                rate = _nonnegative_float(candidate.get(key))
                if rate is not None:
                    self.decode_tok_per_s = rate
                    self.decode_rate_source = "response_metrics"
                    break
            if self.decode_tok_per_s is not None:
                break

        if self.decode_tok_per_s is None:
            for candidate in (timings, metrics, value.get("timings_seconds")):
                if not isinstance(candidate, dict):
                    continue
                decode_ms = _nonnegative_float(candidate.get("predicted_ms"))
                predicted_n = _nonnegative_int(candidate.get("predicted_n"))
                if decode_ms is not None and decode_ms > 0 and predicted_n is not None:
                    self.decode_tok_per_s = max(0, predicted_n - 1) * 1000.0 / decode_ms
                    self.decode_rate_source = "response_metrics"
                    break

    def observe_sse_data(self, data: str, event_name: str, now_ns: int) -> None:
        if not data or data == "[DONE]":
            return
        try:
            value = json.loads(data)
        except (json.JSONDecodeError, UnicodeError):
            return
        self.observe_json(value)
        if _has_output_delta(value, event_name):
            self.note_token(now_ns)


class SSEParser:
    """Small incremental SSE reader that retains metrics only, never generated text."""

    def __init__(self, metrics: ResponseMetrics) -> None:
        self.metrics = metrics
        self.buffer = bytearray()
        self.data_lines: list[str] = []
        self.event_name = ""
        self.first_line = True

    def feed(self, chunk: bytes, now_ns: int) -> None:
        self.buffer.extend(chunk)
        while True:
            newline = self.buffer.find(b"\n")
            if newline < 0:
                return
            raw_line = bytes(self.buffer[:newline])
            del self.buffer[: newline + 1]
            self._line(raw_line, now_ns)

    def finish(self, now_ns: int) -> None:
        if self.buffer:
            self._line(bytes(self.buffer), now_ns)
            self.buffer.clear()
        self._dispatch(now_ns)

    def _line(self, raw_line: bytes, now_ns: int) -> None:
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        if self.first_line:
            self.first_line = False
            if raw_line.startswith(b"\xef\xbb\xbf"):
                raw_line = raw_line[3:]
        if not raw_line:
            self._dispatch(now_ns)
            return
        if raw_line.startswith(b":"):
            return
        field, separator, value = raw_line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        name = field.decode("ascii", errors="ignore")
        if name == "data":
            self.data_lines.append(value.decode("utf-8", errors="replace"))
        elif name == "event":
            self.event_name = value.decode("utf-8", errors="replace")

    def _dispatch(self, now_ns: int) -> None:
        if self.data_lines:
            self.metrics.observe_sse_data("\n".join(self.data_lines), self.event_name, now_ns)
        self.data_lines.clear()
        self.event_name = ""


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _has_output_delta(value: Any, event_name: str) -> bool:
    if not isinstance(value, dict):
        return False

    delta = value.get("delta")
    if isinstance(delta, dict):
        for key in ("text", "thinking", "partial_json", "reasoning_content", "content"):
            part = delta.get(key)
            if isinstance(part, str) and part:
                return True

    if isinstance(delta, str) and delta:
        event_type = value.get("type")
        if isinstance(event_type, str) and event_type.endswith(".delta"):
            return True
        if event_name.endswith(".delta"):
            return True

    choices = value.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_delta = choice.get("delta")
            if not isinstance(choice_delta, dict):
                continue
            for key in ("content", "reasoning_content"):
                part = choice_delta.get(key)
                if isinstance(part, str) and part:
                    return True
            tool_calls = choice_delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if isinstance(function, dict) and any(
                        isinstance(function.get(key), str) and function[key]
                        for key in ("name", "arguments")
                    ):
                        return True

    event_type = value.get("type", event_name)
    if isinstance(event_type, str) and event_type in {
        "content_block_delta",
        "response.output_text.delta",
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
        "response.function_call_arguments.delta",
    }:
        return True
    return False


def _parse_endpoint(base_url: str) -> Endpoint:
    try:
        parts = urlsplit(base_url)
        port = parts.port
    except ValueError:
        raise ValueError("invalid base URL") from None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("base URL must be an http(s) origin or path prefix without credentials/query")
    return Endpoint(parts.scheme, parts.hostname, port, parts.path.rstrip("/"))


def _request_target(endpoint: Endpoint, captured_path: str) -> str:
    if (
        not isinstance(captured_path, str)
        or not captured_path.startswith("/")
        or "\r" in captured_path
        or "\n" in captured_path
    ):
        raise ValueError("manifest path must be an absolute HTTP path")
    parts = urlsplit(captured_path)
    if parts.scheme or parts.netloc or parts.fragment:
        raise ValueError("manifest path must contain only a path and optional query")
    return endpoint.path_prefix + parts.path + (f"?{parts.query}" if parts.query else "")


def _safe_capture_headers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("manifest headers must be an object")
    headers: dict[str, str] = {}
    for name, header_value in value.items():
        if not isinstance(name, str) or not isinstance(header_value, str):
            raise ValueError("manifest header names and values must be strings")
        normalized = name.strip().lower()
        if normalized not in _ALLOWED_CAPTURE_HEADERS:
            continue
        if "\r" in header_value or "\n" in header_value:
            raise ValueError("manifest contains an invalid header value")
        headers[normalized] = header_value
    return headers


def _validate_manifest_row(value: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"manifest line {line_number} must be a JSON object")
    seq = value.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise ValueError(f"manifest line {line_number} has an invalid seq")
    method = value.get("method")
    if not isinstance(method, str) or not _METHOD_RE.fullmatch(method):
        raise ValueError(f"manifest line {line_number} has an invalid method")
    captured_path = value.get("path")
    if not isinstance(captured_path, str) or not captured_path.startswith("/"):
        raise ValueError(f"manifest line {line_number} has an invalid path")
    if "\r" in captured_path or "\n" in captured_path or "#" in captured_path:
        raise ValueError(f"manifest line {line_number} has an invalid path")
    body_file = value.get("body_file")
    if not isinstance(body_file, str) or not body_file:
        raise ValueError(f"manifest line {line_number} has no body_file")
    if Path(body_file).is_absolute():
        raise ValueError(f"manifest line {line_number} body_file must be relative")
    request_bytes = value.get("request_bytes")
    if isinstance(request_bytes, bool) or not isinstance(request_bytes, int) or request_bytes < 0:
        raise ValueError(f"manifest line {line_number} has invalid request_bytes")
    digest = value.get("body_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError(f"manifest line {line_number} has invalid body_sha256")
    value["_headers"] = _safe_capture_headers(value.get("headers"))
    return value


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        root = path.resolve(strict=True).parent
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError("cannot read UTF-8 capture manifest") from None
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSON in manifest line {line_number}") from None
        row = _validate_manifest_row(value, line_number)
        seq = row["seq"]
        if seq in seen:
            raise ValueError(f"duplicate seq in manifest line {line_number}")
        seen.add(seq)
        row["_manifest_root"] = str(root)
        records.append(row)
    records.sort(key=lambda row: row["seq"])
    return records


def _selected_records(
    rows: list[dict[str, Any]],
    minimum: int | None,
    maximum: int | None,
    excluded: set[int],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if (minimum is None or row["seq"] >= minimum)
        and (maximum is None or row["seq"] <= maximum)
        and row["seq"] not in excluded
    ]
    if not selected:
        raise ValueError("selection contains no manifest requests")
    return selected


def _load_body(row: dict[str, Any]) -> bytes:
    root = Path(row["_manifest_root"]).resolve()
    candidate = (root / row["body_file"]).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"seq {row['seq']} body_file escapes the manifest directory")
    try:
        body = candidate.read_bytes()
    except OSError:
        raise ValueError(f"cannot read body_file for seq {row['seq']}") from None
    if len(body) != row["request_bytes"]:
        raise ValueError(f"request_bytes mismatch for seq {row['seq']}")
    digest = hashlib.sha256(body).hexdigest()
    if digest.lower() != row["body_sha256"].lower():
        raise ValueError(f"body_sha256 mismatch for seq {row['seq']}")
    return body


def _request_seed_info(body: bytes, server_default_seed: int | None) -> dict[str, Any]:
    try:
        request = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {
            "seed_mode": "unknown",
            "seed_value": None,
            "request_modified": False,
        }
    if not isinstance(request, dict):
        return {
            "seed_mode": "unknown",
            "seed_value": None,
            "request_modified": False,
        }
    if "seed" in request:
        value = request["seed"]
        safe_value = value if isinstance(value, int) and not isinstance(value, bool) else None
        return {
            "seed_mode": "request",
            "seed_value": safe_value,
            "request_modified": False,
        }
    return {
        "seed_mode": "server_default" if server_default_seed is not None else "unspecified",
        "seed_value": server_default_seed,
        "request_modified": False,
    }


def _auth_header(path: str, api_key: str | None, auth_mode: str) -> dict[str, str]:
    if api_key is None:
        return {}
    if auth_mode == "auto":
        route = urlsplit(path).path
        mode = "x-api-key" if route == "/v1/messages" or route.startswith("/v1/messages/") else "bearer"
    else:
        mode = auth_mode
    if mode == "x-api-key":
        return {"x-api-key": api_key}
    return {"Authorization": f"Bearer {api_key}"}


def _new_connection(endpoint: Endpoint, timeout_seconds: float) -> http.client.HTTPConnection:
    connection_type = (
        http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
    )
    return connection_type(endpoint.host, endpoint.port, timeout=timeout_seconds)


def _elapsed_seconds(start_ns: int, event_ns: int | None) -> float | None:
    if event_ns is None:
        return None
    return max(0.0, (event_ns - start_ns) / 1e9)


def _replay_one(
    endpoint: Endpoint,
    row: dict[str, Any],
    api_key: str | None,
    auth_mode: str,
    timeout_seconds: float,
    server_default_seed: int | None,
) -> dict[str, Any]:
    body = _load_body(row)
    path = row["path"]
    target = _request_target(endpoint, path)
    headers = dict(row["_headers"])
    headers.update(_auth_header(path, api_key, auth_mode))
    metrics = ResponseMetrics(started_ns=time.perf_counter_ns())
    status: int | None = None
    response_content_type: str | None = None
    transport_error: str | None = None
    sse_parser: SSEParser | None = None
    response_chunks: bytearray | None = None
    conn = _new_connection(endpoint, timeout_seconds)
    try:
        conn.request(row["method"], target, body=body, headers=headers)
        response = conn.getresponse()
        status = response.status
        response_content_type = response.getheader("Content-Type")
        is_sse = bool(
            response_content_type
            and response_content_type.split(";", 1)[0].strip().lower() == "text/event-stream"
        )
        if is_sse:
            sse_parser = SSEParser(metrics)
        else:
            response_chunks = bytearray()
        while True:
            chunk = response.read1(64 * 1024)
            if not chunk:
                break
            now_ns = time.perf_counter_ns()
            metrics.note_first_byte(now_ns)
            if sse_parser is not None:
                sse_parser.feed(chunk, now_ns)
            elif response_chunks is not None:
                response_chunks.extend(chunk)
        if sse_parser is not None:
            sse_parser.finish(time.perf_counter_ns())
        elif response_chunks:
            try:
                metrics.observe_json(json.loads(bytes(response_chunks)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    except (OSError, http.client.HTTPException, TimeoutError) as error:
        transport_error = type(error).__name__
    finally:
        conn.close()
    finished_ns = time.perf_counter_ns()

    output_tokens = metrics.output_tokens
    decode_rate = metrics.decode_tok_per_s
    decode_rate_source = metrics.decode_rate_source
    if (
        decode_rate is None
        and output_tokens is not None
        and output_tokens > 1
        and metrics.first_token_ns is not None
    ):
        decode_seconds = max(0.0, (finished_ns - metrics.first_token_ns) / 1e9)
        if decode_seconds > 0:
            decode_rate = (output_tokens - 1) / decode_seconds
            decode_rate_source = "observed_stream"

    capture_timestamp = row.get("timestamp")
    capture_status = row.get("response_status")
    return {
        "artifact_type": "ninfer_serve_http_replay_request",
        "schema_version": 1,
        "seq": row["seq"],
        "capture_timestamp": capture_timestamp if isinstance(capture_timestamp, str) else None,
        "method": row["method"],
        "route": urlsplit(path).path,
        "capture_response_status": (
            capture_status
            if isinstance(capture_status, int) and not isinstance(capture_status, bool)
            else None
        ),
        "request_bytes_original": len(body),
        "request_bytes_sent": len(body),
        **_request_seed_info(body, server_default_seed),
        "http_status": status,
        "response_content_type": response_content_type,
        "client_time_to_first_byte_seconds": _elapsed_seconds(
            metrics.started_ns, metrics.first_byte_ns
        ),
        "client_time_to_first_token_seconds": _elapsed_seconds(
            metrics.started_ns, metrics.first_token_ns
        ),
        "time_to_first_token_seconds": _elapsed_seconds(
            metrics.started_ns, metrics.first_token_ns
        ),
        "time_to_first_token_source": (
            "client_sse_delta" if metrics.first_token_ns is not None else None
        ),
        "output_tokens": output_tokens,
        "decode_tok_per_s": decode_rate,
        "decode_tok_per_s_source": decode_rate_source,
        "server_log_match": "not_requested",
        "server_metrics": None,
        "transport_error": transport_error,
        "elapsed_seconds": max(0.0, (finished_ns - metrics.started_ns) / 1e9),
    }


def _request_key(row: dict[str, Any]) -> tuple[str, str] | None:
    request = row.get("request")
    if not isinstance(request, dict):
        return None
    request_id = request.get("request_id")
    if request_id is None:
        return None
    instance = row.get("server_instance_id")
    return (str(instance), str(request_id))


def _read_server_log(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    terminal: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError("cannot read UTF-8 server request log") from None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSON in server request log line {line_number}") from None
        if not isinstance(row, dict):
            continue
        event = row.get("event")
        if event in _REQUEST_EVENTS:
            if _request_key(row) is not None:
                attempts.append(row)
        elif event in _TERMINAL_EVENTS:
            key = _request_key(row)
            if key is not None:
                terminal[key] = row
    return attempts, terminal


def _server_metrics(done: dict[str, Any], request_id: Any) -> dict[str, Any]:
    result = done.get("result")
    result = result if isinstance(result, dict) else {}
    timings = done.get("timings_seconds")
    timings = timings if isinstance(timings, dict) else {}
    speculative = done.get("speculative")
    speculative = speculative if isinstance(speculative, dict) else {}

    def timing(name: str) -> float | None:
        return _nonnegative_float(timings.get(name))

    accepted_per_position = speculative.get("accepted_per_position")
    if isinstance(accepted_per_position, list):
        accepted_per_position = [
            count for item in accepted_per_position
            if (count := _nonnegative_int(item)) is not None
        ]
    else:
        accepted_per_position = None

    spec_metrics = {
        "accepted_per_position": accepted_per_position,
        "accepted_tokens": _nonnegative_int(speculative.get("accepted_tokens")),
        "drafted_tokens": _nonnegative_int(speculative.get("drafted_tokens")),
        "rounds": _nonnegative_int(speculative.get("rounds")),
        "draft_window": _nonnegative_int(speculative.get("draft_window")),
        "fallback_steps": _nonnegative_int(speculative.get("fallback_steps")),
        "backend": speculative.get("backend") if isinstance(speculative.get("backend"), str) else None,
    }
    return {
        "request_id": request_id,
        "event": "request_done",
        "prompt_tokens": _nonnegative_int(result.get("prompt_tokens")),
        "completion_tokens": _nonnegative_int(result.get("completion_tokens")),
        "timings_seconds": {
            "decode": timing("decode"),
            "ttft": timing("ttft"),
            "total": timing("total"),
            "prefill": timing("prefill"),
        },
        "speculative": spec_metrics,
    }


def _attach_server_log(
    records: list[dict[str, Any]],
    log_path: Path,
) -> tuple[int, int]:
    attempts, terminal = _read_server_log(log_path)
    if len(attempts) != len(records):
        for record in records:
            record["server_log_match"] = "count_mismatch"
            record["server_metrics"] = None
        return len(records), len(attempts)

    matched = 0
    for record, attempt in zip(records, attempts):
        key = _request_key(attempt)
        assert key is not None
        event = attempt.get("event")
        request = attempt.get("request")
        request_id = request.get("request_id") if isinstance(request, dict) else None
        if event == "request_rejected":
            record["server_log_match"] = "request_rejected"
            record["server_metrics"] = {
                "request_id": request_id,
                "event": "request_rejected",
            }
            matched += 1
            continue
        done = terminal.get(key)
        if done is None:
            record["server_log_match"] = "request_started_only"
            record["server_metrics"] = {
                "request_id": request_id,
                "event": "request_started_only",
            }
            matched += 1
            continue
        if done.get("event") != "request_done":
            record["server_log_match"] = "request_error"
            record["server_metrics"] = {
                "request_id": request_id,
                "event": "request_error",
            }
            matched += 1
            continue
        server_metrics = _server_metrics(done, request_id)
        record["server_log_match"] = "request_done"
        record["server_metrics"] = server_metrics
        completion_tokens = server_metrics["completion_tokens"]
        timing_values = server_metrics["timings_seconds"]
        decode_seconds = timing_values["decode"]
        if completion_tokens is not None:
            record["output_tokens"] = completion_tokens
        if decode_seconds is not None and decode_seconds > 0 and completion_tokens is not None:
            record["decode_tok_per_s"] = max(completion_tokens - 1, 0) / decode_seconds
            record["decode_tok_per_s_source"] = "server_request_log"
        server_ttft = timing_values["ttft"]
        if record.get("client_time_to_first_token_seconds") is None and server_ttft is not None:
            record["time_to_first_token_seconds"] = server_ttft
            record["time_to_first_token_source"] = "server_request_log"
        matched += 1
    return matched, len(attempts)


def _read_results(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError("cannot read UTF-8 replay results") from None
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSON in replay results line {line_number}") from None
        if not isinstance(row, dict):
            raise ValueError(f"replay results line {line_number} is not an object")
        records.append(row)
    return records


def _write_results(path: Path, records: list[dict[str, Any]], *, atomic: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if atomic:
        target = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(target, 0o600)
        if atomic:
            os.replace(target, path)
    finally:
        if atomic:
            try:
                target.unlink()
            except FileNotFoundError:
                pass


def _parse_seq_list(values: Sequence[str]) -> set[int]:
    result: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                seq = int(part)
            except ValueError:
                raise ValueError("sequence selectors must be integers") from None
            if seq < 0:
                raise ValueError("sequence selectors must be nonnegative")
            result.add(seq)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="capture JSONL manifest")
    parser.add_argument("--base-url", help="NInfer Serve origin, for example http://127.0.0.1:8081")
    parser.add_argument("--output", type=Path, help="per-request metrics JSONL output")
    parser.add_argument("--exclude-seq", action="append", default=[], help="comma-separated capture seqs to skip; repeatable")
    parser.add_argument("--min-seq", type=int, help="inclusive lower capture seq bound")
    parser.add_argument("--max-seq", type=int, help="inclusive upper capture seq bound")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--api-key-env", default="NINFER_API_KEY", help="environment variable holding the token; empty disables auth")
    parser.add_argument("--auth-mode", choices=("auto", "bearer", "x-api-key"), default="auto")
    parser.add_argument("--server-default-seed", type=int, help="record the NInfer --seed value; the server applies it only if the body has no seed")
    parser.add_argument("--server-log", type=Path, help="local fresh --request-log-jsonl output to merge after replay")
    parser.add_argument("--merge-server-log", type=Path, help="posthoc merge a copied server request log without replay")
    parser.add_argument("--results", type=Path, help="results JSONL to enrich in --merge-server-log mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    merge_mode = args.merge_server_log is not None
    if merge_mode:
        if args.results is None or args.manifest is not None or args.base_url is not None or args.output is not None:
            parser.error("--merge-server-log requires --results and cannot be combined with replay arguments")
        if args.server_log is not None:
            parser.error("--server-log is only for replay mode")
        try:
            records = _read_results(args.results)
            if not records:
                raise ValueError("replay results contain no request records")
            matched, attempts = _attach_server_log(records, args.merge_server_log)
            _write_results(args.results, records, atomic=True)
        except ValueError as error:
            parser.error(str(error))
        print(f"server log alignment: {matched}/{len(records)} replay rows; {attempts} server attempts")
        return 0 if matched == len(records) and attempts == len(records) else 2

    if args.manifest is None or args.base_url is None or args.output is None:
        parser.error("replay mode requires --manifest, --base-url, and --output")
    if args.results is not None:
        parser.error("--results is only for --merge-server-log mode")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.min_seq is not None and args.min_seq < 0:
        parser.error("--min-seq must be nonnegative")
    if args.max_seq is not None and args.max_seq < 0:
        parser.error("--max-seq must be nonnegative")
    if args.min_seq is not None and args.max_seq is not None and args.min_seq > args.max_seq:
        parser.error("--min-seq must not exceed --max-seq")

    try:
        endpoint = _parse_endpoint(args.base_url)
        rows = _read_manifest(args.manifest)
        excluded = _parse_seq_list(args.exclude_seq)
        selected = _selected_records(rows, args.min_seq, args.max_seq, excluded)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        result = _replay_one(
            endpoint,
            row,
            api_key,
            args.auth_mode,
            args.timeout_seconds,
            args.server_default_seed,
        )
        results.append(result)
        _write_results(args.output, results, atomic=False)
        status = result["http_status"] if result["http_status"] is not None else "transport_error"
        print(
            f"replayed {index}/{len(selected)} seq={row['seq']} status={status}",
            file=sys.stderr,
            flush=True,
        )

    alignment_ok = True
    if args.server_log is not None:
        try:
            matched, attempts = _attach_server_log(results, args.server_log)
            _write_results(args.output, results, atomic=True)
            print(
                f"server log alignment: {matched}/{len(results)} replay rows; {attempts} server attempts",
                file=sys.stderr,
            )
            alignment_ok = matched == len(results) and attempts == len(results)
        except ValueError as error:
            print(f"server log merge unavailable: {error}", file=sys.stderr)
            alignment_ok = False
    return 2 if not alignment_ok or any(item["transport_error"] for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
