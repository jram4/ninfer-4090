#!/usr/bin/env python3
"""Bounded private HTTP capture proxy for OpenAI chat and Anthropic Messages.

Run for one task/session with explicit labels and a private capture directory. To
route only one process through it, use a transient environment override, for
example ``ANTHROPIC_BASE_URL=http://127.0.0.1:18080 claude ...`` or
``OPENAI_BASE_URL=http://127.0.0.1:18080 client ...``. Do not edit global client
configuration or production services. The proxy stores exact transfer-decoded
bytes by default; use ``--body-storage redacted`` to retain metadata only (such
capsules cannot be replayed). Credentials and hop-by-hop headers are neither
forwarded nor retained. This module uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import parse_qsl, urlsplit


SCHEMA_VERSION = 1
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_REQUESTS = 20
DEFAULT_MAX_SECONDS = 3600.0
READ_CHUNK_BYTES = 64 * 1024
SUPPORTED_PATHS = {"/v1/chat/completions", "/v1/messages"}

_SECRET_HEADER = re.compile(
    r"(?:^|[-_])(?:authorization|authentication|auth|bearer|cookie|secret|token|password|credential|"
    r"signature|sig|api[-_]?key|access[-_]?key|subscription[-_]?key|key|session[-_]?token)(?:$|[-_])",
    re.IGNORECASE,
)
_SECRET_QUERY = re.compile(
    r"(?:^|[-_])(?:authorization|authentication|auth|bearer|cookie|secret|token|password|credential|"
    r"signature|sig|api[-_]?key|access[-_]?key|subscription[-_]?key|key|session[-_]?token)(?:$|[-_])",
    re.IGNORECASE,
)
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9_.:/+-]{1,160}$")
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_REQUEST_HOP_HEADERS = _HOP_BY_HOP | {"host", "content-length", "expect"}
_RESPONSE_HOP_HEADERS = _HOP_BY_HOP | {"content-length"}
_RESPONSE_ID_HEADERS = ("request-id", "anthropic-request-id", "x-request-id")
_SENSITIVE_ROUTE_MESSAGE = b'{"error":"credential-like query parameters are not accepted"}'


class CaptureError(RuntimeError):
    """Invalid request or capsule data that is safe to report to the caller."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256(encoded)


def _capsule_hash(manifest: dict[str, Any]) -> str:
    return _canonical_hash({key: value for key, value in manifest.items() if key != "capsule_sha256"})


def _valid_capture_route(method: str, target: str) -> bool:
    try:
        parsed = urlsplit(target)
    except ValueError:
        return False
    return method == "POST" and parsed.path in SUPPORTED_PATHS and not parsed.fragment


def _label(value: str, name: str, *, max_length: int = 160) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"{name} must be a nonempty string of at most {max_length} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value.strip()


def _safe_headers(
    raw_headers: Iterable[tuple[str, str]], *, excluded: set[str]
) -> list[dict[str, str]]:
    """Keep semantic headers while dropping credentials and transport framing."""
    connection_tokens: set[str] = set()
    pairs = list(raw_headers)
    for name, value in pairs:
        if name.lower() == "connection":
            connection_tokens.update(token.strip().lower() for token in value.split(","))
    denied = excluded | connection_tokens
    kept: list[dict[str, str]] = []
    for name, value in pairs:
        lowered = name.lower()
        if lowered in denied or lowered in {"cookie2", "set-cookie2"} or _SECRET_HEADER.search(lowered):
            continue
        if not _HEADER_NAME.fullmatch(name):
            continue
        if any((ord(char) < 32 and char != "\t") or ord(char) == 127 for char in value):
            continue
        kept.append({"name": name, "value": value})
    return kept


def _has_secret_query(target: str) -> bool:
    try:
        query = urlsplit(target).query
        return any(_SECRET_QUERY.search(name) for name, _ in parse_qsl(query, keep_blank_values=True))
    except ValueError:
        return True


def _request_semantics(body: bytes, headers: Sequence[dict[str, str]]) -> dict[str, Any]:
    header_map = {entry["name"].lower(): entry["value"] for entry in headers}
    content_type = header_map.get("content-type", "")
    info: dict[str, Any] = {
        "model": None,
        "stream_requested": False,
        "output_budget": None,
        "prompt_cache_control_present": False,
    }
    if "json" not in content_type.lower():
        return info
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return info
    if not isinstance(payload, dict):
        return info
    model = payload.get("model")
    if isinstance(model, str) and _SAFE_MODEL.fullmatch(model):
        info["model"] = model
    info["stream_requested"] = payload.get("stream") is True
    budget = payload.get("max_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int):
        budget = payload.get("max_completion_tokens")
    if isinstance(budget, int) and not isinstance(budget, bool) and budget >= 0:
        info["output_budget"] = budget

    def has_cache_control(value: Any) -> bool:
        if isinstance(value, dict):
            if isinstance(value.get("cache_control"), dict):
                return True
            return any(has_cache_control(item) for item in value.values())
        if isinstance(value, list):
            return any(has_cache_control(item) for item in value)
        return False

    info["prompt_cache_control_present"] = has_cache_control(payload)
    return info


def _response_identity(headers: Sequence[dict[str, str]], body_path: Path) -> dict[str, str]:
    header_map = {entry["name"].lower(): entry["value"] for entry in headers}
    identity: dict[str, str] = {}
    for name in _RESPONSE_ID_HEADERS:
        value = header_map.get(name)
        if value and _SAFE_IDENTIFIER.fullmatch(value):
            identity["header"] = value
            identity["header_name"] = name
            break
    try:
        if body_path.stat().st_size <= 8 * 1024 * 1024:
            payload = json.loads(body_path.read_bytes())
            if isinstance(payload, dict):
                value = payload.get("id")
                if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value):
                    identity["body_id"] = value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        pass
    return identity


def _request_identity(headers: Sequence[dict[str, str]], capsule_id: str) -> dict[str, str]:
    del headers
    return {"capsule_id": capsule_id}


@dataclass(frozen=True)
class Upstream:
    scheme: str
    host: str
    port: int
    prefix: str

    @classmethod
    def parse(cls, url: str) -> "Upstream":
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("upstream must be an http:// or https:// URL")
        if parsed.username or parsed.password:
            raise ValueError("upstream URLs must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("upstream URL must not contain a query or fragment")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise ValueError("upstream URL has an invalid port") from exc
        prefix = parsed.path.rstrip("/")
        return cls(parsed.scheme, parsed.hostname, port, prefix)

    @property
    def origin(self) -> str:
        default_port = 443 if self.scheme == "https" else 80
        host = f"[{self.host}]" if ":" in self.host else self.host
        suffix = "" if self.port == default_port else f":{self.port}"
        return f"{self.scheme}://{host}{suffix}"

    def target(self, request_target: str) -> str:
        if not request_target.startswith("/") or request_target.startswith("//"):
            raise CaptureError("only origin-form request targets are supported")
        if "\r" in request_target or "\n" in request_target:
            raise CaptureError("invalid request target")
        return f"{self.prefix}{request_target}"

    def connect(self, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)


class CapsuleStore:
    """Atomic restricted capsule writer and sanitized JSONL descriptor emitter."""

    def __init__(
        self,
        capture_dir: Path,
        *,
        descriptor_jsonl: Path | None = None,
        cache_state: str = "unknown",
        bucket_labels: Sequence[str] = (),
        session_id: str,
        task_label: str,
        run_label: str,
        traffic_class: str,
        stratum: str,
        derived_from: str | None = None,
        body_storage: str = "exact",
    ) -> None:
        self.capture_dir = capture_dir.expanduser().resolve()
        self.capture_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.capture_dir, 0o700)
        self.descriptor_jsonl = descriptor_jsonl.expanduser().resolve() if descriptor_jsonl else None
        if cache_state not in {"unknown", "cold", "warm"}:
            raise ValueError("cache_state must be unknown, cold, or warm")
        if traffic_class not in {"real_user", "agent_generated", "derived"}:
            raise ValueError("traffic_class must be real_user, agent_generated, or derived")
        if traffic_class == "derived" and not derived_from:
            raise ValueError("derived traffic requires --derived-from")
        if traffic_class != "derived" and derived_from is not None:
            raise ValueError("--derived-from is only valid for derived traffic")
        if body_storage not in {"exact", "redacted"}:
            raise ValueError("body_storage must be exact or redacted")
        self.cache_state = cache_state
        self.session_id = _label(session_id, "session_id")
        self.task_label = _label(task_label, "task_label")
        self.run_label = _label(run_label, "run_label")
        self.traffic_class = traffic_class
        self.stratum = _label(stratum, "stratum")
        self.derived_from = _label(derived_from, "derived_from") if derived_from is not None else None
        self.body_storage = body_storage
        self.bucket_labels = tuple(_label(label, "bucket_label") for label in bucket_labels)
        self._descriptor_lock = threading.Lock()
        self._sequence_lock = threading.Lock()
        self._sequence = 0

    def begin(self, method: str, target: str, body: bytes, request_headers: list[dict[str, str]]) -> "CapsuleBuilder":
        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence
        capsule_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex}"
        work_dir = self.capture_dir / f".{capsule_id}.partial"
        work_dir.mkdir(mode=0o700)
        builder = CapsuleBuilder(self, capsule_id, sequence, work_dir, method, target, body, request_headers)
        if self.body_storage == "exact":
            builder.request_body_path.write_bytes(body)
            os.chmod(builder.request_body_path, 0o600)
        return builder

    def _publish(self, builder: "CapsuleBuilder", manifest: dict[str, Any]) -> Path:
        manifest["capsule_sha256"] = _capsule_hash(manifest)
        manifest_path = builder.work_dir / "manifest.json"
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        manifest_path.write_bytes(manifest_bytes)
        for child in builder.work_dir.iterdir():
            if child.is_file():
                os.chmod(child, 0o400)
        final_dir = self.capture_dir / f"{builder.capsule_id}.capsule"
        builder.work_dir.rename(final_dir)
        os.chmod(final_dir, 0o500)
        descriptor = self._descriptor(manifest, final_dir)
        if self.descriptor_jsonl is not None:
            self._append_descriptor(descriptor)
        return final_dir

    def _descriptor(self, manifest: dict[str, Any], capsule_path: Path) -> dict[str, Any]:
        request = manifest["request"]
        response = manifest["response"]
        semantics = request["semantics"]
        labels = {
            f"route={request['method']} {request['target'].split('?', 1)[0]}",
            f"stream={str(semantics['stream_requested']).lower()}",
            f"cache_state={manifest['cache_state']}",
            f"response_status={response.get('status')}",
        }
        if semantics.get("model"):
            labels.add(f"model={semantics['model']}")
        labels.update(self.bucket_labels)
        clean_labels = sorted(label[:200] for label in labels if label and "\n" not in label)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "capsule_id": manifest["capsule_id"],
            "sequence": manifest["sequence"],
            "capsule_sha256": manifest["capsule_sha256"],
            "capsule_path": str(capsule_path),
            "session_id": self.session_id,
            "task_label": self.task_label,
            "run_label": self.run_label,
            "traffic_class": self.traffic_class,
            "stratum": self.stratum,
            "labels": clean_labels,
            "identity_scope": request["identity_scope"],
            "request_sha256": request["body_sha256"],
            "response_sha256": response.get("body_sha256"),
            "metadata": {
                "captured_at_utc": manifest["captured_at_utc"],
                "upstream_origin": manifest["upstream_origin"],
                "method": request["method"],
                "request_target": request["target"].split("?", 1)[0],
                "model": semantics.get("model"),
                "stream_requested": semantics["stream_requested"],
                "cache_state": manifest["cache_state"],
                "prompt_cache_control_present": semantics["prompt_cache_control_present"],
                "response_status": response.get("status"),
                "response_identity": response.get("identity", {}),
                "body_storage": self.body_storage,
            },
        }
        if self.derived_from is not None:
            result["derived_from"] = self.derived_from
        if semantics.get("output_budget") is not None:
            result["output_budget"] = semantics["output_budget"]
        return result

    def _append_descriptor(self, descriptor: dict[str, Any]) -> None:
        assert self.descriptor_jsonl is not None
        self.descriptor_jsonl.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        encoded = (json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self._descriptor_lock:
            fd = os.open(self.descriptor_jsonl, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                os.write(fd, encoded)
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)


class CapsuleBuilder:
    def __init__(
        self,
        store: CapsuleStore,
        capsule_id: str,
        sequence: int,
        work_dir: Path,
        method: str,
        target: str,
        body: bytes,
        request_headers: list[dict[str, str]],
    ) -> None:
        self.store = store
        self.capsule_id = capsule_id
        self.sequence = sequence
        self.work_dir = work_dir
        self.request_body_path = work_dir / "request.body"
        self.response_body_path = work_dir / "response.body"
        self.method = method
        self.target = target
        self.request_body = body
        self.request_headers = request_headers
        self.started_at = _utc_now()
        self.started_clock = time.monotonic()
        self.response_sha = hashlib.sha256()
        self.response_bytes = 0
        self.response_chunks: list[dict[str, int | float]] = []
        self.response_path = self.response_body_path.open("wb") if store.body_storage == "exact" else None
        self.sse_terminal_event: str | None = None
        self.sse_event_types: list[str] = []
        self._sse_buffer = bytearray()
        self._message_id: str | None = None

    def append_response(self, data: bytes) -> None:
        if not data:
            return
        offset = self.response_bytes
        if self.response_path is not None:
            self.response_path.write(data)
            self.response_path.flush()
            self.response_sha.update(data)
        self.response_bytes += len(data)
        elapsed_ms = (time.monotonic() - self.started_clock) * 1000.0
        self.response_chunks.append({"offset": offset, "length": len(data), "elapsed_ms": round(elapsed_ms, 3)})
        self._observe_sse(data)

    def _observe_sse(self, data: bytes) -> None:
        self._sse_buffer.extend(data)
        while True:
            pos = self._sse_buffer.find(b"\n")
            if pos < 0:
                if len(self._sse_buffer) > 16 * 1024:
                    del self._sse_buffer[:-16 * 1024]
                return
            line = bytes(self._sse_buffer[:pos]).rstrip(b"\r")
            del self._sse_buffer[: pos + 1]
            if line.startswith(b"event:"):
                event = line[6:].strip().decode("ascii", errors="ignore")
                if event and len(event) <= 80 and re.fullmatch(r"[A-Za-z0-9_.-]+", event):
                    if event not in self.sse_event_types:
                        self.sse_event_types.append(event)
                    if event in {"message_stop", "message_end", "response.completed", "response.failed"}:
                        self.sse_terminal_event = event
            elif line.startswith(b"data:"):
                value = line[5:].strip()
                if value == b"[DONE]":
                    self.sse_terminal_event = "DONE"
                    continue
                try:
                    payload = json.loads(value)
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                    continue
                if isinstance(payload, dict):
                    msg_id = payload.get("id")
                    message = payload.get("message")
                    if not isinstance(msg_id, str) and isinstance(message, dict):
                        msg_id = message.get("id")
                    if isinstance(msg_id, str) and _SAFE_IDENTIFIER.fullmatch(msg_id):
                        self._message_id = msg_id
                    event_type = payload.get("type")
                    if isinstance(event_type, str) and event_type in {
                        "message_stop", "message_end", "response.completed", "response.failed"
                    }:
                        self.sse_terminal_event = event_type

    def finalize(
        self,
        *,
        response_status: int | None,
        response_reason: str | None,
        response_headers: list[dict[str, str]],
        response_content_type: str | None,
        response_content_length: int | None,
        response_transfer_encoding: str | None,
        upstream_origin: str,
        termination: str,
        cache_state: str | None = None,
        proxy_error: str | None = None,
    ) -> Path:
        if self.response_path is not None:
            self.response_path.flush()
            os.fsync(self.response_path.fileno())
            self.response_path.close()
        semantics = _request_semantics(self.request_body, self.request_headers)
        if self._message_id is not None:
            identity = {"body_id": self._message_id}
        elif self.store.body_storage == "exact":
            identity = _response_identity(response_headers, self.response_body_path)
        else:
            identity = {}
        exact_storage = self.store.body_storage == "exact"
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "capsule_id": self.capsule_id,
            "sequence": self.sequence,
            "captured_at_utc": self.started_at,
            "completed_at_utc": _utc_now(),
            "upstream_origin": upstream_origin,
            "cache_state": cache_state or self.store.cache_state,
            "request": {
                "method": self.method,
                "target": self.target,
                "headers": self.request_headers,
                "body_file": "request.body" if exact_storage else None,
                "body_bytes": len(self.request_body),
                "body_sha256": _sha256(self.request_body) if exact_storage else None,
                "identity_scope": "exact_body_no_credentials" if exact_storage else "redacted_no_body_identity",
                "semantics": semantics,
                "identity": _request_identity(self.request_headers, self.capsule_id),
            },
            "response": {
                "status": response_status,
                "reason": response_reason,
                "headers": response_headers,
                "content_type": response_content_type,
                "content_length": response_content_length,
                "transfer_encoding": response_transfer_encoding,
                "body_file": "response.body" if exact_storage else None,
                "body_bytes": self.response_bytes,
                "body_sha256": self.response_sha.hexdigest() if exact_storage else None,
                "chunks": self.response_chunks,
                "stream": {
                    "requested": semantics["stream_requested"],
                    "observed": bool(response_content_type and "text/event-stream" in response_content_type.lower()),
                    "chunk_count": len(self.response_chunks),
                    "event_types": self.sse_event_types,
                    "terminal_event": self.sse_terminal_event,
                    "termination": termination,
                },
                "identity": identity,
            },
            "upstream_error": proxy_error,
        }
        return self.store._publish(self, manifest)


def _read_request_body(handler: BaseHTTPRequestHandler, max_bytes: int) -> bytes:
    length_values = handler.headers.get_all("Content-Length", [])
    transfer_values = handler.headers.get_all("Transfer-Encoding", [])
    if len(length_values) > 1 or len(transfer_values) > 1:
        raise CaptureError("duplicate request framing headers")
    length_value = length_values[0] if length_values else None
    transfer_encoding = transfer_values[0].lower() if transfer_values else ""
    if length_value is not None and transfer_encoding:
        raise CaptureError("request cannot contain both Content-Length and Transfer-Encoding")
    if length_value is not None:
        try:
            length = int(length_value)
        except ValueError as exc:
            raise CaptureError("invalid Content-Length") from exc
        if length < 0:
            raise CaptureError("invalid Content-Length")
        if length > max_bytes:
            raise OverflowError("request body exceeds configured limit")
        body = handler.rfile.read(length)
        if len(body) != length:
            raise CaptureError("request body ended before Content-Length")
        return body
    if transfer_encoding:
        codings = [item.strip() for item in transfer_encoding.split(",") if item.strip()]
        if codings != ["chunked"]:
            raise CaptureError("only chunked request transfer encoding is supported")
        body = bytearray()
        while True:
            line = handler.rfile.readline(8192)
            if not line or not line.endswith(b"\n"):
                raise CaptureError("invalid chunked request")
            size_line = line.rstrip(b"\r\n")
            if not re.fullmatch(rb"[0-9A-Fa-f]+(?:;[^\r\n]*)?", size_line):
                raise CaptureError("invalid chunk size")
            try:
                size = int(size_line.split(b";", 1)[0], 16)
            except ValueError as exc:
                raise CaptureError("invalid chunk size") from exc
            if size < 0 or len(body) + size > max_bytes:
                raise OverflowError("request body exceeds configured limit")
            if size == 0:
                while True:
                    trailer = handler.rfile.readline(8192)
                    if trailer in {b"\r\n", b"\n", b""}:
                        return bytes(body)
                    raise CaptureError("request chunk trailers are not supported")
            chunk = handler.rfile.read(size)
            ending = handler.rfile.read(2)
            if len(chunk) != size or ending != b"\r\n":
                raise CaptureError("incomplete chunked request")
            body.extend(chunk)
    return b""


def _write_error(handler: BaseHTTPRequestHandler, status: int, message: bytes) -> None:
    handler.close_connection = True
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(message)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(message)
        handler.wfile.flush()
    except OSError:
        pass


def _make_handler(
    store: CapsuleStore,
    upstream: Upstream,
    *,
    timeout: float,
    max_request_bytes: int,
    max_response_bytes: int,
    max_requests: int | None,
    max_output_tokens: int | None,
    stop_callback: Callable[[], None],
) -> Callable[..., BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "DreamRSICapture/1"
        sys_version = ""

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(timeout)

        def log_message(self, fmt: str, *args: Any) -> None:
            # Avoid logging request targets, which could contain private data.
            return

        def do_GET(self) -> None:
            self._proxy()

        def do_HEAD(self) -> None:
            self._proxy()

        def do_OPTIONS(self) -> None:
            self._proxy()

        def do_POST(self) -> None:
            self._proxy()

        def do_PUT(self) -> None:
            self._proxy()

        def do_PATCH(self) -> None:
            self._proxy()

        def do_DELETE(self) -> None:
            self._proxy()

        def _proxy(self) -> None:
            try:
                self.connection.settimeout(timeout)
                upstream_target = upstream.target(self.path)
                if _has_secret_query(self.path):
                    _write_error(self, 400, _SENSITIVE_ROUTE_MESSAGE)
                    return
                if (
                    self.command in {"GET", "HEAD"}
                    and urlsplit(self.path).path == "/api/hello"
                    and self.headers.get("Content-Length", "0") == "0"
                    and not self.headers.get("Transfer-Encoding")
                ):
                    body = b'{"message":"hello"}'
                    self.close_connection = True
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(body)
                        self.wfile.flush()
                    return
                if not _valid_capture_route(self.command, self.path):
                    _write_error(self, 404, b'{"error":"capture route is not supported"}')
                    return
                body = _read_request_body(self, max_request_bytes)
            except OverflowError:
                _write_error(self, 413, b'{"error":"request body too large"}')
                return
            except (CaptureError, OSError):
                _write_error(self, 400, b'{"error":"invalid request"}')
                return

            raw_headers = list(self.headers.raw_items())
            request_headers = _safe_headers(raw_headers, excluded=_REQUEST_HOP_HEADERS)
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or "json" not in content_types[0].lower():
                _write_error(self, 400, b'{"error":"provider requests must use JSON"}')
                return
            try:
                request_json = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                _write_error(self, 400, b'{"error":"provider request JSON is invalid"}')
                return
            if not isinstance(request_json, dict):
                _write_error(self, 400, b'{"error":"provider request JSON must be an object"}')
                return
            semantics = _request_semantics(body, request_headers)
            output_budget = semantics.get("output_budget")
            if max_output_tokens is not None and (
                output_budget is None or output_budget > max_output_tokens
            ):
                _write_error(
                    self,
                    413,
                    json.dumps(
                        {"error": "request exceeds capture output-token guard", "max_output_tokens": max_output_tokens},
                        separators=(",", ":"),
                    ).encode("ascii"),
                )
                print(
                    json.dumps(
                        {"event": "budget_rejected", "max_output_tokens": max_output_tokens},
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return
            if not claim_request():
                _write_error(self, 429, b'{"error":"capture request limit reached"}')
                stop_callback()
                return
            builder = store.begin(self.command, self.path, body, request_headers)
            client = upstream.connect(timeout)
            response: http.client.HTTPResponse | None = None
            response_status: int | None = None
            response_reason: str | None = None
            response_headers: list[dict[str, str]] = []
            response_content_type: str | None = None
            response_content_length: int | None = None
            response_transfer_encoding: str | None = None
            termination = "upstream_error"
            proxy_error: str | None = None
            response_started = False
            client_cancelled = False
            should_have_body = self.command != "HEAD"

            def send_response_chunk(chunk: bytes) -> None:
                if response_content_length is None or response_transfer_encoding:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()

            try:
                client.putrequest(self.command, upstream_target, skip_host=True, skip_accept_encoding=True)
                for item in request_headers:
                    client.putheader(item["name"], item["value"])
                host_value = f"[{upstream.host}]" if ":" in upstream.host else upstream.host
                default_port = 443 if upstream.scheme == "https" else 80
                if upstream.port != default_port:
                    host_value = f"{host_value}:{upstream.port}"
                client.putheader("Host", host_value)
                client.putheader("Content-Length", str(len(body)))
                client.putheader("Connection", "close")
                client.endheaders(body)
                response = client.getresponse()
                response_status = response.status
                response_reason = response.reason
                raw_response_headers = list(response.getheaders())
                response_headers = _safe_headers(raw_response_headers, excluded=_RESPONSE_HOP_HEADERS)
                response_content_type = response.getheader("Content-Type")
                content_length_header = response.getheader("Content-Length")
                if content_length_header is not None:
                    try:
                        parsed_content_length = int(content_length_header)
                        if parsed_content_length >= 0:
                            response_content_length = parsed_content_length
                    except ValueError:
                        response_content_length = None
                response_transfer_encoding = response.getheader("Transfer-Encoding")
                no_body_status = response_status in {204, 304} or 100 <= response_status < 200
                should_have_body = should_have_body and not no_body_status
                self.close_connection = True
                self.send_response(response_status)
                for item in response_headers:
                    name = item["name"].lower()
                    if name in {"server", "date"}:
                        continue
                    self.send_header(item["name"], item["value"])
                if should_have_body:
                    if response_content_length is not None and not response_transfer_encoding:
                        self.send_header("Content-Length", str(response_content_length))
                    else:
                        self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                response_started = True
                if should_have_body:
                    while True:
                        try:
                            chunk = response.read1(READ_CHUNK_BYTES)
                        except http.client.IncompleteRead as exc:
                            chunk = exc.partial
                            if chunk:
                                if builder.response_bytes + len(chunk) > max_response_bytes:
                                    chunk = chunk[: max_response_bytes - builder.response_bytes]
                                    termination = "response_limit"
                                    proxy_error = "response body exceeded configured limit"
                                builder.append_response(chunk)
                                try:
                                    if chunk:
                                        send_response_chunk(chunk)
                                except (BrokenPipeError, ConnectionResetError, OSError):
                                    client_cancelled = True
                                    termination = "downstream_cancelled"
                            if not client_cancelled and termination != "response_limit":
                                termination = "upstream_truncated"
                            break
                        if not chunk:
                            termination = "complete"
                            break
                        if builder.response_bytes + len(chunk) > max_response_bytes:
                            chunk = chunk[: max_response_bytes - builder.response_bytes]
                            if chunk:
                                builder.append_response(chunk)
                                try:
                                    send_response_chunk(chunk)
                                except (BrokenPipeError, ConnectionResetError, OSError):
                                    client_cancelled = True
                                    termination = "downstream_cancelled"
                                    break
                            if not client_cancelled:
                                termination = "response_limit"
                            proxy_error = "response body exceeded configured limit"
                            break
                        builder.append_response(chunk)
                        try:
                            send_response_chunk(chunk)
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            client_cancelled = True
                            termination = "downstream_cancelled"
                            break
                    if not client_cancelled and termination == "complete" and (
                        response_content_length is None or response_transfer_encoding
                    ):
                        try:
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            client_cancelled = True
                            termination = "downstream_cancelled"
                else:
                    termination = "complete"
            except (OSError, http.client.HTTPException, TimeoutError) as exc:
                proxy_error = type(exc).__name__
                if response_started:
                    termination = "upstream_truncated"
                else:
                    termination = "upstream_error"
                    _write_error(self, 502, b'{"error":"capture proxy could not reach upstream"}')
                    response_status = 502
                    response_reason = "Bad Gateway"
                    response_content_type = "application/json"
                    response_content_length = len(b'{"error":"capture proxy could not reach upstream"}')
                    response_headers = [{"name": "Content-Type", "value": "application/json"}]
                    builder.append_response(b'{"error":"capture proxy could not reach upstream"}')
            finally:
                if response is not None:
                    response.close()
                client.close()
                if client_cancelled:
                    termination = "downstream_cancelled"
                path = builder.finalize(
                    response_status=response_status,
                    response_reason=response_reason,
                    response_headers=response_headers,
                    response_content_type=response_content_type,
                    response_content_length=response_content_length,
                    response_transfer_encoding=response_transfer_encoding,
                    upstream_origin=upstream.origin,
                    termination=termination,
                    proxy_error=proxy_error,
                )
                print(
                    json.dumps(
                        {
                            "event": "capsule_complete",
                            "capsule_id": builder.capsule_id,
                            "capsule_path": str(path),
                            "status": response_status,
                            "request_sha256": _sha256(body) if store.body_storage == "exact" else None,
                            "response_sha256": builder.response_sha.hexdigest() if store.body_storage == "exact" else None,
                            "termination": termination,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                if max_requests is not None and request_count_value() >= max_requests:
                    self.close_connection = True
                    stop_callback()

    request_count = 0
    request_count_lock = threading.Lock()

    def request_count_value() -> int:
        with request_count_lock:
            return request_count

    def claim_request() -> bool:
        nonlocal request_count
        with request_count_lock:
            if max_requests is not None and request_count >= max_requests:
                return False
            request_count += 1
            return True

    def handler_factory(*args: Any, **kwargs: Any) -> Handler:
        return Handler(*args, **kwargs)

    return handler_factory


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(self, server_address: Any, request_handler: Any, *, max_workers: int) -> None:
        if max_workers <= 0:
            raise ValueError("max_active_connections must be positive")
        self._worker_slots = threading.BoundedSemaphore(max_workers)
        super().__init__(server_address, request_handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._worker_slots.acquire(blocking=False):
            try:
                body = b'{"error":"capture connection limit reached"}'
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                    + body
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()


class CaptureProxy:
    """Importable proxy wrapper; evaluator may also manage it as a subprocess."""

    def __init__(
        self,
        *,
        upstream_url: str,
        capture_dir: str | Path,
        listen_host: str = "127.0.0.1",
        listen_port: int = 18080,
        descriptor_jsonl: str | Path | None = None,
        cache_state: str = "unknown",
        bucket_labels: Sequence[str] = (),
        session_id: str,
        task_label: str,
        run_label: str,
        traffic_class: str,
        stratum: str = "unspecified",
        derived_from: str | None = None,
        body_storage: str = "exact",
        timeout: float = 600.0,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_requests: int | None = DEFAULT_MAX_REQUESTS,
        max_active_connections: int = 4,
        max_output_tokens: int | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0 or max_request_bytes <= 0 or max_response_bytes <= 0:
            raise ValueError("timeouts and body limits must be positive")
        if max_requests is not None and max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if max_active_connections <= 0:
            raise ValueError("max_active_connections must be positive")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.upstream = Upstream.parse(upstream_url)
        self.store = CapsuleStore(
            Path(capture_dir),
            descriptor_jsonl=Path(descriptor_jsonl) if descriptor_jsonl else None,
            cache_state=cache_state,
            bucket_labels=bucket_labels,
            session_id=session_id,
            task_label=task_label,
            run_label=run_label,
            traffic_class=traffic_class,
            stratum=stratum,
            derived_from=derived_from,
            body_storage=body_storage,
        )
        self.timeout = timeout
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.max_requests = max_requests
        self.max_active_connections = max_active_connections
        self.max_output_tokens = max_output_tokens
        self._server: _ReusableThreadingHTTPServer | None = None
        self._shutdown_started = False
        self._shutdown_lock = threading.Lock()
        handler_factory = _make_handler(
            self.store,
            self.upstream,
            timeout=timeout,
            max_request_bytes=max_request_bytes,
            max_response_bytes=max_response_bytes,
            max_requests=max_requests,
            max_output_tokens=max_output_tokens,
            stop_callback=self._stop_from_handler,
        )
        self._server = _ReusableThreadingHTTPServer(
            (listen_host, listen_port), handler_factory, max_workers=max_active_connections
        )

    @property
    def address(self) -> tuple[str, int]:
        assert self._server is not None
        return self._server.server_address[:2]

    def serve_forever(self) -> None:
        assert self._server is not None
        self._server.serve_forever(poll_interval=0.2)

    def shutdown(self) -> None:
        self._stop_from_handler()

    def server_close(self) -> None:
        if self._server is not None:
            self._server.server_close()

    def _stop_from_handler(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
        if self._server is not None:
            threading.Thread(target=self._server.shutdown, name="capture-shutdown", daemon=True).start()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (override one process only):\n"
            "  python -m tools.dreamrsi.capture --upstream http://127.0.0.1:8080 "
            "--capture-dir /private/capsules --session-id session-01 --task-label task "
            "--run-label run-01 --traffic-class real_user --descriptor-jsonl /private/descriptors.jsonl\n"
            "  ANTHROPIC_BASE_URL=http://127.0.0.1:18080 claude ...\n"
            "Use OPENAI_BASE_URL=http://127.0.0.1:18080 for an OpenAI-compatible client. "
            "No global config or production service changes are needed."
        ),
    )
    parser.add_argument("--upstream", required=True, help="Model-service origin, e.g. http://127.0.0.1:8080")
    parser.add_argument("--capture-dir", required=True, type=Path, help="Restricted directory for raw capsules on FNN")
    parser.add_argument("--descriptor-jsonl", type=Path, help="Optional sanitized descriptor output path")
    parser.add_argument("--listen-host", default="127.0.0.1", help="Bind address; expose only on a trusted private interface")
    parser.add_argument("--listen-port", type=int, default=18080, help="Research-only listening port")
    parser.add_argument("--timeout", type=float, default=600.0, help="Upstream connection/read timeout in seconds")
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS, help="Stop after this many captured requests (default: 20)")
    parser.add_argument("--max-active-connections", type=int, default=4, help="Bound in-flight HTTP clients (default: 4)")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS, help="Stop after this many seconds (default: 3600)")
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument("--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Reject requests above their JSON max_tokens/max_completion_tokens value; bodies are never rewritten",
    )
    parser.add_argument("--session-id", required=True, help="Stable label grouping requests from one session")
    parser.add_argument("--task-label", required=True, help="Non-sensitive task label")
    parser.add_argument("--run-label", required=True, help="Non-sensitive capture run label")
    parser.add_argument("--traffic-class", required=True, choices=("real_user", "agent_generated", "derived"))
    parser.add_argument("--derived-from", help="Parent capsule ID when traffic-class is derived")
    parser.add_argument("--stratum", default="unspecified", help="Single corpus stratum for weighted analysis")
    parser.add_argument("--cache-state", choices=("unknown", "cold", "warm"), default="unknown", help="Explicit evaluator-supplied model-cache state; defaults to unknown")
    parser.add_argument("--label", action="append", default=[], help="Additional non-sensitive descriptor tag; may be repeated")
    parser.add_argument("--body-storage", choices=("exact", "redacted"), default="exact", help="Exact private capsule bytes or metadata-only redacted capsule")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    try:
        proxy = CaptureProxy(
            upstream_url=args.upstream,
            capture_dir=args.capture_dir,
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            descriptor_jsonl=args.descriptor_jsonl,
            cache_state=args.cache_state,
            bucket_labels=args.label,
            session_id=args.session_id,
            task_label=args.task_label,
            run_label=args.run_label,
            traffic_class=args.traffic_class,
            derived_from=args.derived_from,
            stratum=args.stratum,
            body_storage=args.body_storage,
            timeout=args.timeout,
            max_request_bytes=args.max_request_bytes,
            max_response_bytes=args.max_response_bytes,
            max_requests=args.max_requests,
            max_active_connections=args.max_active_connections,
            max_output_tokens=args.max_output_tokens,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    host, port = proxy.address
    print(
        json.dumps(
            {"event": "capture_listening", "host": host, "port": port, "upstream": proxy.upstream.origin},
            separators=(",", ":"),
        ),
        flush=True,
    )
    timer = threading.Timer(args.max_seconds, proxy.shutdown)
    timer.daemon = True
    timer.start()
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        del signum
        proxy.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.signal(sig, handle_signal)
        except (ValueError, OSError):
            pass
    try:
        proxy.serve_forever()
    finally:
        timer.cancel()
        proxy.server_close()
        for sig, old_handler in previous_handlers.items():
            signal.signal(sig, old_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
