"""Focused end-to-end tests for the capture boundary and direct replay path."""

from __future__ import annotations

import http.client
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tools.dreamrsi.capture import CaptureProxy
from tools.dreamrsi.derive import DeriveError, _assert_only_expected_transform, _derive_request_body, derive_capsule
from tools.dreamrsi.replay import ReplayError, _load_capsule, replay_capsule


class _FakeModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length)
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "body": request_body,
                "headers": {name.lower(): value for name, value in self.headers.items()},
            }
        )
        response_chunks = [
            b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test_123\"}}\n\n",
            b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("request-id", "req_test_456")
        self.send_header("Set-Cookie", "response_session=never-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for chunk in response_chunks:
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.005)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class _CancelStreamingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunk = b"data: " + (b"x" * 32768) + b"\n\n"
        try:
            for _ in range(256):
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.003)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def _start_model(
    handler_type: type[BaseHTTPRequestHandler] = _FakeModelHandler,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _post(url: tuple[str, int], target: str, body: bytes) -> tuple[int, bytes, dict[str, str]]:
    connection = http.client.HTTPConnection(*url, timeout=5)
    connection.putrequest("POST", target)
    connection.putheader("Content-Type", "application/json")
    connection.putheader("anthropic-version", "2023-06-01")
    connection.putheader("User-Agent", "Claude Code/2.1.283")
    connection.putheader("x-request-id", "claude-request-test-123")
    connection.putheader("x-stainless-retry-count", "0")
    connection.putheader("Authorization", "Bearer never-store-this")
    connection.putheader("Cookie", "session=never-store-this")
    connection.putheader("x-api-key", "never-store-this-either")
    connection.putheader("Content-Length", str(len(body)))
    connection.endheaders(body)
    response = connection.getresponse()
    response_body = response.read()
    headers = {name.lower(): value for name, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, response_body, headers


class CaptureReplayTests(unittest.TestCase):
    def test_derived_capsules_apply_only_declared_transforms_and_keep_source_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture_dir = root / "captures"
            capture_dir.mkdir(mode=0o700)
            source = capture_dir / "raw-rejected.capsule"
            source.mkdir(mode=0o700)
            source_body = (
                b'{\n  "model": "claude-sonnet-5",\n  "max_tokens": 96,\n'
                b'  "stream": true,\n  "messages": [{"role":"user","content":"private fixture text"}],\n'
                b'  "tools": [{"name":"Read","input_schema":{"type":"object"}}],\n'
                b'  "thinking": {"type":"adaptive", "display": "omitted"},\n'
                b'  "output_config": {"effort":"high"}\n}\n'
            )
            source_hash = hashlib.sha256(source_body).hexdigest()
            manifest = {
                "schema_version": 1,
                "capsule_id": "raw-rejected",
                "upstream_origin": "http://127.0.0.1:18081",
                "cache_state": "unknown",
                "request": {
                    "method": "POST",
                    "target": "/v1/messages?beta=true",
                    "headers": [
                        {"name": "Content-Type", "value": "application/json"},
                        {"name": "User-Agent", "value": "Claude Code/2.1.283"},
                    ],
                    "body_file": "request.body",
                    "body_bytes": len(source_body),
                    "body_sha256": source_hash,
                    "semantics": {"model": "claude-sonnet-5", "stream_requested": True, "output_budget": 96},
                    "identity": {"capsule_id": "raw-rejected"},
                },
                "response": {"status": 400},
            }
            manifest_body = (json.dumps(manifest, sort_keys=True) + "\n").encode()
            (source / "request.body").write_bytes(source_body)
            (source / "manifest.json").write_bytes(manifest_body)
            (source / "response.body").write_bytes(b'{"error":{"type":"invalid_request_error"}}')
            for path in source.iterdir():
                path.chmod(0o400)
            source.chmod(0o500)

            derived_body, diff = _derive_request_body(source_body)
            self.assertEqual(
                diff,
                [{"path": "$.thinking.display", "from": "omitted", "to": "summarized"}],
            )
            self.assertEqual(
                derived_body,
                source_body.replace(b'"display": "omitted"', b'"display": "summarized"', 1),
            )
            _assert_only_expected_transform(source_body, derived_body)
            altered = json.loads(derived_body)
            altered["tools"][0]["name"] = "ChangedTool"
            with self.assertRaisesRegex(DeriveError, "outside the declared transform paths"):
                _assert_only_expected_transform(source_body, json.dumps(altered).encode())

            original_mode = source.stat().st_mode & 0o777
            request_mode = (source / "request.body").stat().st_mode & 0o777
            result = derive_capsule(source)
            derived = Path(result["capsule_path"])
            derived_manifest = json.loads((derived / "manifest.json").read_text())
            derived_request = (derived / "request.body").read_bytes()
            _loaded_capsule, _loaded_manifest, loaded_body = _load_capsule(derived)
            derived_object = json.loads(derived_request)
            source_object = json.loads(source_body)
            expected = json.loads(json.dumps(source_object))
            expected["thinking"]["display"] = "summarized"
            self.assertEqual(derived_object, expected)
            self.assertEqual(loaded_body, derived_request)
            self.assertEqual(result["request_sha256"], hashlib.sha256(derived_request).hexdigest())
            self.assertEqual(derived_manifest["derivation"]["parent_raw_request_sha256"], source_hash)
            self.assertEqual(derived_manifest["derivation"]["transformed_body_sha256"], result["request_sha256"])
            self.assertEqual(derived_manifest["derivation"]["diff"], diff)
            self.assertTrue(derived_manifest["derivation"]["semantic_transform"])
            self.assertFalse(derived_manifest["derivation"]["production_compatible"])
            self.assertEqual(derived_manifest["derivation"]["parent_raw_location"], str(source.resolve()))
            self.assertEqual((derived / "response.body").read_bytes(), b"")
            self.assertEqual(derived_manifest["response"]["stream"]["termination"], "derived_not_replayed")
            self.assertEqual((source / "request.body").read_bytes(), source_body)
            self.assertEqual((source / "manifest.json").read_bytes(), manifest_body)
            self.assertEqual(source.stat().st_mode & 0o777, original_mode)
            self.assertEqual((source / "request.body").stat().st_mode & 0o777, request_mode)
            self.assertEqual(derived.stat().st_mode & 0o777, 0o500)
            for path in derived.iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o400)

            h04a_request_hash = hashlib.sha256(derived_request).hexdigest()
            h04b_body, h04b_diff = _derive_request_body(source_body, effort_override="xhigh")
            expected_h04b_diff = [
                {"path": "$.thinking.display", "from": "omitted", "to": "summarized"},
                {"path": "$.output_config.effort", "from": "high", "to": "xhigh"},
            ]
            self.assertEqual(h04b_diff, expected_h04b_diff)
            self.assertEqual(
                h04b_body,
                source_body.replace(b'"display": "omitted"', b'"display": "summarized"', 1).replace(
                    b'"effort":"high"', b'"effort":"xhigh"', 1
                ),
            )
            _assert_only_expected_transform(source_body, h04b_body, effort_override="xhigh")
            altered_h04b = json.loads(h04b_body)
            altered_h04b["messages"][0]["content"] = "unintended mutation"
            with self.assertRaisesRegex(DeriveError, "outside the declared transform paths"):
                _assert_only_expected_transform(
                    source_body,
                    json.dumps(altered_h04b).encode(),
                    effort_override="xhigh",
                )

            h04b_result = derive_capsule(source, effort_override="xhigh")
            h04b_path = Path(h04b_result["capsule_path"])
            h04b_manifest = json.loads((h04b_path / "manifest.json").read_text())
            h04b_request = (h04b_path / "request.body").read_bytes()
            expected_h04b = json.loads(json.dumps(source_object))
            expected_h04b["thinking"]["display"] = "summarized"
            expected_h04b["output_config"]["effort"] = "xhigh"
            self.assertEqual(json.loads(h04b_request), expected_h04b)
            self.assertNotEqual(h04b_path, derived)
            self.assertEqual(h04b_manifest["derivation"]["diff"], expected_h04b_diff)
            self.assertEqual(h04b_manifest["derivation"]["parent_raw_request_sha256"], source_hash)
            self.assertEqual(h04b_manifest["derivation"]["transformed_body_sha256"], h04b_result["request_sha256"])
            self.assertEqual(h04b_manifest["derivation"]["h04a_request_sha256"], h04a_request_hash)
            self.assertEqual(
                h04b_manifest["derivation"]["h04a_capsule_id"],
                derived_manifest["capsule_id"],
            )
            self.assertTrue(h04b_manifest["derivation"]["semantic_transform"])
            self.assertFalse(h04b_manifest["derivation"]["production_compatible"])
            self.assertEqual(h04b_path.stat().st_mode & 0o777, 0o500)
            for path in h04b_path.iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o400)
            self.assertEqual((derived / "request.body").read_bytes(), derived_request)
            self.assertEqual((source / "request.body").read_bytes(), source_body)

            descriptor_text = (capture_dir / "descriptors.jsonl").read_text()
            descriptor_rows = [json.loads(line) for line in descriptor_text.splitlines()]
            descriptor = next(row for row in descriptor_rows if row["capsule_id"] == derived_manifest["capsule_id"])
            h04b_descriptor = next(row for row in descriptor_rows if row["capsule_id"] == h04b_manifest["capsule_id"])
            self.assertEqual(descriptor["metadata"]["diff"], diff)
            self.assertEqual(h04b_descriptor["metadata"]["diff"], expected_h04b_diff)
            self.assertEqual(h04b_descriptor["metadata"]["h04a_request_sha256"], h04a_request_hash)
            self.assertNotIn(str(source.resolve()), descriptor_text)
            self.assertNotIn("private fixture text", descriptor_text)
            self.assertNotIn("parent_raw_location", descriptor_text)
            _make_capsules_removable(capture_dir)

    def test_capsule_filters_credentials_and_replays_without_proxy(self) -> None:
        model, model_thread = _start_model()
        replay_model, replay_thread = _start_model()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                capture_dir = root / "raw-capsules"
                descriptor_path = root / "descriptors.jsonl"
                proxy = CaptureProxy(
                    upstream_url=f"http://127.0.0.1:{model.server_address[1]}",
                    capture_dir=capture_dir,
                    listen_host="127.0.0.1",
                    listen_port=0,
                    descriptor_jsonl=descriptor_path,
                    bucket_labels=("condition=smoke",),
                )
                proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                proxy_thread.start()
                body = (
                    b'{ "model" : "claude-test", "max_tokens": 12, "stream": true, '
                    b'"messages": [{"role":"user","content":"capsule-private-prompt"}] }'
                )
                status, response_body, _headers = _post(proxy.address, "/v1/messages?beta=1", body)
                self.assertEqual(status, 200)
                self.assertIn(b"message_stop", response_body)
                proxy.shutdown()
                proxy_thread.join(timeout=5)
                proxy.server_close()

                self.assertEqual(len(model.requests), 1)  # type: ignore[attr-defined]
                received = model.requests[0]  # type: ignore[attr-defined]
                self.assertEqual(received["path"], "/v1/messages?beta=1")
                self.assertEqual(received["body"], body)
                for forbidden in ("authorization", "cookie", "x-api-key"):
                    self.assertNotIn(forbidden, received["headers"])
                self.assertEqual(received["headers"]["anthropic-version"], "2023-06-01")
                self.assertEqual(received["headers"]["x-stainless-retry-count"], "0")

                capsules = list(capture_dir.glob("*.capsule"))
                self.assertEqual(len(capsules), 1)
                capsule = capsules[0]
                manifest = json.loads((capsule / "manifest.json").read_text())
                self.assertEqual((capsule / "request.body").read_bytes(), body)
                self.assertEqual((capsule / "response.body").read_bytes(), response_body)
                self.assertEqual(manifest["request"]["target"], "/v1/messages?beta=1")
                self.assertEqual(manifest["request"]["body_sha256"], hashlib.sha256(body).hexdigest())
                self.assertTrue(manifest["request"]["semantics"]["stream_requested"])
                self.assertEqual(manifest["cache_state"], "unknown")
                self.assertEqual(manifest["request"]["semantics"]["cli_version"], "2.1.283")
                self.assertEqual(
                    manifest["request"]["identity"]["client_header"],
                    "claude-request-test-123",
                )
                self.assertEqual(manifest["response"]["stream"]["terminal_event"], "message_stop")
                self.assertEqual(manifest["response"]["identity"]["body_id"], "msg_test_123")
                stored_headers = {item["name"].lower() for item in manifest["request"]["headers"]}
                for forbidden in ("authorization", "cookie", "x-api-key"):
                    self.assertNotIn(forbidden, stored_headers)
                stored_response_headers = {item["name"].lower() for item in manifest["response"]["headers"]}
                self.assertNotIn("set-cookie", stored_response_headers)
                self.assertEqual(capsule.stat().st_mode & 0o777, 0o500)
                self.assertEqual((capsule / "manifest.json").stat().st_mode & 0o777, 0o400)

                descriptor = json.loads(descriptor_path.read_text().splitlines()[0])
                self.assertEqual(descriptor["schema_version"], 1)
                self.assertTrue(descriptor["bucket_labels"])
                self.assertEqual(descriptor["request_sha256"], manifest["request"]["body_sha256"])
                self.assertEqual(descriptor["metadata"]["cli_version"], "2.1.283")
                self.assertEqual(descriptor["metadata"]["request_target"], "/v1/messages")
                self.assertNotIn("capsule-private-prompt", descriptor_path.read_text())

                replay_output = root / "replay-response.body"
                replay_result = replay_capsule(
                    capsule,
                    upstream_url=f"http://127.0.0.1:{replay_model.server_address[1]}",
                    response_output=replay_output,
                    timeout=5,
                )
                self.assertEqual(replay_result["status"], 200)
                self.assertEqual(replay_result["response_identity"]["header"], "req_test_456")
                self.assertEqual(replay_output.read_bytes(), response_body)
                self.assertEqual(len(replay_model.requests), 1)  # type: ignore[attr-defined]
                replayed = replay_model.requests[0]  # type: ignore[attr-defined]
                self.assertEqual(replayed["body"], body)
                self.assertEqual(replayed["path"], "/v1/messages?beta=1")
                for forbidden in ("authorization", "cookie", "x-api-key"):
                    self.assertNotIn(forbidden, replayed["headers"])
                _make_capsules_removable(capture_dir)
        finally:
            model.shutdown()
            model.server_close()
            replay_model.shutdown()
            replay_model.server_close()
            model_thread.join(timeout=5)
            replay_thread.join(timeout=5)

    def test_downstream_disconnect_is_recorded_as_cancellation(self) -> None:
        model, model_thread = _start_model(_CancelStreamingHandler)
        proxy: CaptureProxy | None = None
        proxy_thread: threading.Thread | None = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                capture_dir = Path(tmp) / "capsules"
                proxy = CaptureProxy(
                    upstream_url=f"http://127.0.0.1:{model.server_address[1]}",
                    capture_dir=capture_dir,
                    listen_host="127.0.0.1",
                    listen_port=0,
                )
                proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                proxy_thread.start()
                connection = http.client.HTTPConnection(*proxy.address, timeout=5)
                body = b'{"model":"test","stream":true,"messages":[]}'
                connection.putrequest("POST", "/v1/messages")
                connection.putheader("Content-Type", "application/json")
                connection.putheader("Content-Length", str(len(body)))
                connection.endheaders(body)
                response = connection.getresponse()
                self.assertTrue(response.read1(64))
                socket_file = response.fp
                assert socket_file is not None
                client_socket = getattr(getattr(socket_file, "raw", None), "_sock", None)
                assert client_socket is not None
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                connection.close()
                response.close()

                deadline = time.monotonic() + 8
                capsules: list[Path] = []
                while time.monotonic() < deadline:
                    capsules = list(capture_dir.glob("*.capsule"))
                    if capsules:
                        break
                    time.sleep(0.02)
                self.assertEqual(len(capsules), 1, "proxy did not publish the cancelled stream capsule")
                manifest = json.loads((capsules[0] / "manifest.json").read_text())
                self.assertEqual(manifest["response"]["stream"]["termination"], "downstream_cancelled")
                self.assertGreater(manifest["response"]["stream"]["chunk_count"], 0)
                proxy.shutdown()
                proxy_thread.join(timeout=5)
                proxy.server_close()
                _make_capsules_removable(capture_dir)
        finally:
            if proxy is not None:
                proxy.shutdown()
                if proxy_thread is not None:
                    proxy_thread.join(timeout=5)
                proxy.server_close()
            model.shutdown()
            model.server_close()
            model_thread.join(timeout=5)

    def test_api_hello_probe_does_not_create_capsule_or_consume_request_slot(self) -> None:
        model, model_thread = _start_model()
        proxy: CaptureProxy | None = None
        proxy_thread: threading.Thread | None = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                capture_dir = root / "capsules"
                descriptor_path = root / "descriptors.jsonl"
                proxy = CaptureProxy(
                    upstream_url=f"http://127.0.0.1:{model.server_address[1]}",
                    capture_dir=capture_dir,
                    listen_host="127.0.0.1",
                    listen_port=0,
                    descriptor_jsonl=descriptor_path,
                    max_requests=1,
                    max_output_tokens=8,
                )
                proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                proxy_thread.start()

                for method in ("GET", "HEAD"):
                    connection = http.client.HTTPConnection(*proxy.address, timeout=5)
                    connection.request(method, "/api/hello")
                    response = connection.getresponse()
                    response_body = response.read()
                    self.assertEqual(response.status, 200)
                    if method == "GET":
                        self.assertEqual(json.loads(response_body), {"message": "hello"})
                    else:
                        self.assertEqual(response_body, b"")
                    connection.close()
                    self.assertEqual(list(capture_dir.glob("*.capsule")), [])
                    self.assertFalse(descriptor_path.exists())

                self.assertEqual(model.requests, [])  # type: ignore[attr-defined]
                accepted = (
                    b'{"model":"claude-sonnet-5","max_tokens":8,"stream":true,'
                    b'"messages":[{"role":"user","content":"bounded prompt"}]}'
                )
                status, response_body, _headers = _post(proxy.address, "/v1/messages", accepted)
                self.assertEqual(status, 200)
                self.assertIn(b"message_stop", response_body)
                capsules = _wait_for_capsules(capture_dir, 1)
                self.assertEqual(len(capsules), 1)
                self.assertEqual(len(model.requests), 1)  # type: ignore[attr-defined]
                received = model.requests[0]  # type: ignore[attr-defined]
                self.assertEqual(received["path"], "/v1/messages")
                self.assertEqual(received["body"], accepted)
                descriptor_lines = descriptor_path.read_text().splitlines()
                self.assertEqual(len(descriptor_lines), 1)
                self.assertEqual(json.loads(descriptor_lines[0])["metadata"]["request_target"], "/v1/messages")

                proxy_thread.join(timeout=5)
                proxy.server_close()
                _make_capsules_removable(capture_dir)
        finally:
            if proxy is not None:
                proxy.shutdown()
                if proxy_thread is not None:
                    proxy_thread.join(timeout=5)
                proxy.server_close()
            model.shutdown()
            model.server_close()
            model_thread.join(timeout=5)

    def test_output_token_guard_rejects_without_mutating_or_spending_request_slot(self) -> None:
        model, model_thread = _start_model()
        proxy: CaptureProxy | None = None
        proxy_thread: threading.Thread | None = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                capture_dir = Path(tmp) / "capsules"
                proxy = CaptureProxy(
                    upstream_url=f"http://127.0.0.1:{model.server_address[1]}",
                    capture_dir=capture_dir,
                    listen_host="127.0.0.1",
                    listen_port=0,
                    max_requests=1,
                    max_output_tokens=8,
                )
                proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                proxy_thread.start()
                oversized = b'{"model":"test","max_tokens":9,"messages":[]}'
                status, _body, _headers = _post(proxy.address, "/v1/messages", oversized)
                self.assertEqual(status, 413)
                self.assertEqual(list(capture_dir.glob("*.capsule")), [])
                without_budget = b'{"model":"test","messages":[]}'
                status, _body, _headers = _post(proxy.address, "/v1/messages", without_budget)
                self.assertEqual(status, 413)
                self.assertEqual(list(capture_dir.glob("*.capsule")), [])

                accepted = b'{"model":"test","max_tokens":8,"messages":[]}'
                status, _body, _headers = _post(proxy.address, "/v1/messages", accepted)
                self.assertEqual(status, 200)
                capsules = _wait_for_capsules(capture_dir, 1)
                manifest = json.loads((capsules[0] / "manifest.json").read_text())
                self.assertEqual((capsules[0] / "request.body").read_bytes(), accepted)
                self.assertEqual(manifest["request"]["semantics"]["output_budget"], 8)
                self.assertEqual(len(model.requests), 1)  # type: ignore[attr-defined]
                proxy.shutdown()
                proxy_thread.join(timeout=5)
                proxy.server_close()
                _make_capsules_removable(capture_dir)
        finally:
            if proxy is not None:
                proxy.shutdown()
                if proxy_thread is not None:
                    proxy_thread.join(timeout=5)
                proxy.server_close()
            model.shutdown()
            model.server_close()
            model_thread.join(timeout=5)

    def test_replay_rejects_tampered_request_body(self) -> None:
        model, model_thread = _start_model()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                proxy = CaptureProxy(
                    upstream_url=f"http://127.0.0.1:{model.server_address[1]}",
                    capture_dir=root / "capsules",
                    listen_host="127.0.0.1",
                    listen_port=0,
                )
                proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                proxy_thread.start()
                _post(proxy.address, "/v1/messages", b'{"model":"test","messages":[]}')
                proxy.shutdown()
                proxy_thread.join(timeout=5)
                proxy.server_close()
                capsule = next((root / "capsules").glob("*.capsule"))
                manifest_path = capsule / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                original_body = (capsule / "request.body").read_bytes()
                request_body = capsule / "request.body"
                request_body.chmod(0o600)
                request_body.write_bytes(b"tampered")
                request_body.chmod(0o400)
                with self.assertRaisesRegex(ReplayError, "hash does not match"):
                    replay_capsule(capsule, upstream_url=f"http://127.0.0.1:{model.server_address[1]}")

                request_body.chmod(0o600)
                request_body.write_bytes(original_body)
                request_body.chmod(0o400)
                manifest["request"]["target"] = "/v1/messages?api_key=never-replay-this"
                manifest_path.chmod(0o600)
                manifest_path.write_text(json.dumps(manifest))
                manifest_path.chmod(0o400)
                with self.assertRaisesRegex(ReplayError, "credential-like query"):
                    replay_capsule(capsule, upstream_url=f"http://127.0.0.1:{model.server_address[1]}")
                _make_capsules_removable(root / "capsules")
        finally:
            model.shutdown()
            model.server_close()
            model_thread.join(timeout=5)

    def test_flat_replay_deployment_imports_sibling_capture_module(self) -> None:
        model, model_thread = _start_model()
        proxy: CaptureProxy | None = None
        proxy_thread: threading.Thread | None = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                capture_dir = root / "capsules"
                proxy = CaptureProxy(
                    upstream_url=f"http://127.0.0.1:{model.server_address[1]}",
                    capture_dir=capture_dir,
                    listen_host="127.0.0.1",
                    listen_port=0,
                )
                proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                proxy_thread.start()
                body = b'{"model":"claude-test","messages":[]}'
                status, _response_body, _headers = _post(proxy.address, "/v1/messages", body)
                self.assertEqual(status, 200)
                proxy.shutdown()
                proxy_thread.join(timeout=5)
                proxy.server_close()
                capsule = next(capture_dir.glob("*.capsule"))

                # Simulate the FNN deployment where both scripts are siblings and
                # the repository's `tools` package is absent from Python's path.
                flat_tools = root / "flat-tools"
                flat_tools.mkdir(mode=0o700)
                source_tools = Path(__file__).resolve().parent
                shutil.copyfile(source_tools / "capture.py", flat_tools / "capture.py")
                shutil.copyfile(source_tools / "replay.py", flat_tools / "replay.py")
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(flat_tools / "replay.py"),
                        str(capsule),
                        "--upstream",
                        f"http://127.0.0.1:{model.server_address[1]}",
                    ],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                result = json.loads(completed.stdout)
                self.assertEqual(result["status"], 200)
                self.assertTrue(result["complete"])
                self.assertEqual(len(model.requests), 2)  # type: ignore[attr-defined]
                replayed = model.requests[1]  # type: ignore[attr-defined]
                self.assertEqual(replayed["body"], body)
                self.assertEqual(replayed["path"], "/v1/messages")
                _make_capsules_removable(capture_dir)
        finally:
            if proxy is not None:
                proxy.shutdown()
                if proxy_thread is not None:
                    proxy_thread.join(timeout=5)
                proxy.server_close()
            model.shutdown()
            model.server_close()
            model_thread.join(timeout=5)


def _make_capsules_removable(capture_dir: Path) -> None:
    for capsule in capture_dir.glob("*.capsule"):
        capsule.chmod(0o700)
        for path in capsule.iterdir():
            path.chmod(0o600)


def _wait_for_capsules(capture_dir: Path, expected: int) -> list[Path]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        capsules = list(capture_dir.glob("*.capsule"))
        if len(capsules) == expected:
            return capsules
        time.sleep(0.01)
    return list(capture_dir.glob("*.capsule"))


if __name__ == "__main__":
    unittest.main()
