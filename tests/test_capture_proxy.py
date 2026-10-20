from __future__ import annotations

import http.client
import io
import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest

from tools.dreamrsi.capture import CaptureError, CaptureProxy, _read_request_body


class _ChunkedSSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received: dict[str, Any] = {}
    finished = threading.Event()
    infinite = False

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        body_size = int(self.headers.get("Content-Length", "0"))
        type(self).received = {
            "path": self.path,
            "headers": {name.lower(): value for name, value in self.headers.items()},
            "body": self.rfile.read(body_size),
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Set-Cookie", "private-response-cookie=must-not-be-captured")
        self.send_header("X-Api-Key", "response-secret-value")
        self.end_headers()
        if type(self).infinite:
            try:
                for _ in range(100000):
                    payload = b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"
                    self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                type(self).finished.set()
            return
        chunks = [
            b"event: message_start\r\n",
            b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-safe-1\"}}\r\n\r\n",
            b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]
        for chunk in chunks:
            self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        type(self).finished.set()

    def do_GET(self) -> None:
        self.send_error(500)


def _proxy(tmp_path: Path, upstream_server: ThreadingHTTPServer, **kwargs: Any) -> CaptureProxy:
    return CaptureProxy(
        upstream_url=f"http://127.0.0.1:{upstream_server.server_port}",
        capture_dir=tmp_path / "private-capsules",
        descriptor_jsonl=tmp_path / "private-descriptors.jsonl",
        session_id="session-01",
        task_label="task-test",
        run_label="run-01",
        traffic_class="real_user",
        stratum="interactive",
        timeout=5,
        max_requests=1,
        **kwargs,
    )


def _serve(proxy: CaptureProxy) -> threading.Thread:
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    return thread


class CaptureProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)
        _ChunkedSSEHandler.received = {}
        _ChunkedSSEHandler.finished = threading.Event()
        _ChunkedSSEHandler.infinite = False
        self.upstream_server = ThreadingHTTPServer(("127.0.0.1", 0), _ChunkedSSEHandler)
        self.upstream_server.daemon_threads = True
        self.upstream_thread = threading.Thread(target=self.upstream_server.serve_forever, daemon=True)
        self.upstream_thread.start()

    def tearDown(self) -> None:
        self.upstream_server.shutdown()
        self.upstream_server.server_close()
        self.upstream_thread.join(timeout=2)
        self.tempdir.cleanup()

    def test_proxy_preserves_chunked_sse_and_strips_credentials(self) -> None:
        proxy = _proxy(self.tmp_path, self.upstream_server)
        thread = _serve(proxy)
        body = b'{"model":"example-model","messages":[{"role":"user","content":"private prompt"}],"stream":true,"max_tokens":31}'
        conn = http.client.HTTPConnection(*proxy.address, timeout=5)
        conn.request(
            "POST",
            "/v1/messages",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer private-auth-token",
                "Cookie": "private-cookie=secret",
                "X-Api-Key": "private-api-key",
                "Connection": "keep-alive, X-Private-Hop",
                "X-Private-Hop": "must-not-forward",
            },
        )
        response = conn.getresponse()
        response_body = response.read()
        conn.close()
        thread.join(timeout=5)
        proxy.server_close()

        expected = (
            b"event: message_start\r\n"
            b'data: {"type":"message_start","message":{"id":"msg-safe-1"}}\r\n\r\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response_body, expected)
        self.assertEqual(_ChunkedSSEHandler.received["body"], body)
        received_headers = _ChunkedSSEHandler.received["headers"]
        self.assertFalse({"authorization", "cookie", "x-api-key", "x-private-hop"} & set(received_headers))

        capsule_dir = next((self.tmp_path / "private-capsules").glob("*.capsule"))
        manifest = json.loads((capsule_dir / "manifest.json").read_text())
        self.assertEqual(manifest["sequence"], 1)
        self.assertEqual((capsule_dir / "request.body").read_bytes(), body)
        self.assertEqual((capsule_dir / "response.body").read_bytes(), expected)
        self.assertEqual(manifest["response"]["stream"]["termination"], "complete")
        self.assertEqual(manifest["response"]["stream"]["terminal_event"], "message_stop")
        self.assertEqual(manifest["request"]["semantics"]["output_budget"], 31)
        for secret in ("private-auth-token", "private-cookie", "private-api-key", "response-secret-value"):
            self.assertNotIn(secret, json.dumps(manifest))
        self.assertEqual(capsule_dir.stat().st_mode & 0o777, 0o500)
        self.assertEqual((capsule_dir / "request.body").stat().st_mode & 0o777, 0o400)
        descriptor = json.loads((self.tmp_path / "private-descriptors.jsonl").read_text().splitlines()[0])
        self.assertEqual(descriptor["session_id"], "session-01")
        self.assertEqual(descriptor["sequence"], 1)
        self.assertEqual(descriptor["traffic_class"], "real_user")
        self.assertEqual(descriptor["identity_scope"], "exact_body_no_credentials")
        self.assertNotIn("private prompt", json.dumps(descriptor))

    def test_proxy_refuses_unmatched_provider_route_without_capsule(self) -> None:
        proxy = _proxy(self.tmp_path, self.upstream_server)
        thread = _serve(proxy)
        conn = http.client.HTTPConnection(*proxy.address, timeout=5)
        conn.request("POST", "/v1/embeddings", body=b"{}", headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 404)
        response.read()
        conn.close()
        proxy.shutdown()
        thread.join(timeout=3)
        proxy.server_close()
        self.assertFalse(list((self.tmp_path / "private-capsules").glob("*.capsule")))
        self.assertEqual(_ChunkedSSEHandler.received, {})

    def test_credential_query_is_rejected_before_capsule_identity_hashing(self) -> None:
        proxy = _proxy(self.tmp_path, self.upstream_server)
        thread = _serve(proxy)
        conn = http.client.HTTPConnection(*proxy.address, timeout=5)
        conn.request(
            "POST",
            "/v1/messages?api_key=do-not-retain",
            body=b'{"model":"example-model","messages":[],"stream":true,"max_tokens":8}',
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        response.read()
        conn.close()
        proxy.shutdown()
        thread.join(timeout=3)
        proxy.server_close()
        self.assertFalse(list((self.tmp_path / "private-capsules").glob("*.capsule")))
        self.assertEqual(_ChunkedSSEHandler.received, {})

    def test_active_connection_limit_bounds_slow_clients(self) -> None:
        proxy = _proxy(self.tmp_path, self.upstream_server, max_active_connections=1)
        thread = _serve(proxy)
        slow = socket.create_connection(proxy.address, timeout=3)
        slow.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: incomplete\r\n")
        conn = http.client.HTTPConnection(*proxy.address, timeout=3)
        conn.request("POST", "/v1/messages", body=b"{}", headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 503)
        response.read()
        conn.close()
        slow.close()
        proxy.shutdown()
        thread.join(timeout=3)
        proxy.server_close()

    def test_redacted_capture_has_no_body_hash_or_replayable_body(self) -> None:
        proxy = _proxy(self.tmp_path, self.upstream_server, body_storage="redacted")
        thread = _serve(proxy)
        body = b'{"model":"example-model","messages":[{"role":"user","content":"do not retain"}],"stream":true,"max_tokens":8}'
        conn = http.client.HTTPConnection(*proxy.address, timeout=5)
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        response.read()
        conn.close()
        thread.join(timeout=5)
        proxy.server_close()

        capsule_dir = next((self.tmp_path / "private-capsules").glob("*.capsule"))
        manifest = json.loads((capsule_dir / "manifest.json").read_text())
        self.assertIsNone(manifest["request"]["body_file"])
        self.assertIsNone(manifest["request"]["body_sha256"])
        self.assertIsNone(manifest["response"]["body_file"])
        self.assertFalse((capsule_dir / "request.body").exists())
        self.assertFalse((capsule_dir / "response.body").exists())
        descriptor = json.loads((self.tmp_path / "private-descriptors.jsonl").read_text().splitlines()[0])
        self.assertIsNone(descriptor["request_sha256"])
        self.assertEqual(descriptor["identity_scope"], "redacted_no_body_identity")
        self.assertNotIn("do not retain", json.dumps(descriptor))

    def test_response_body_limit_is_enforced_and_recorded(self) -> None:
        proxy = _proxy(self.tmp_path, self.upstream_server, max_response_bytes=16)
        thread = _serve(proxy)
        conn = http.client.HTTPConnection(*proxy.address, timeout=5)
        conn.request(
            "POST",
            "/v1/messages",
            body=b'{"model":"example-model","messages":[],"stream":true,"max_tokens":8}',
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(len(response.read1(16)), 16)
        conn.close()
        thread.join(timeout=5)
        proxy.server_close()
        capsule_dir = next((self.tmp_path / "private-capsules").glob("*.capsule"))
        manifest = json.loads((capsule_dir / "manifest.json").read_text())
        self.assertEqual(manifest["response"]["stream"]["termination"], "response_limit")
        self.assertEqual(manifest["response"]["body_bytes"], 16)
        self.assertEqual(len((capsule_dir / "response.body").read_bytes()), 16)

    def test_chunked_request_parser_rejects_malformed_framing(self) -> None:
        class Headers:
            def get(self, name: str, default: str | None = None) -> str | None:
                return "chunked" if name == "Transfer-Encoding" else default

            def get_all(self, name: str, default: list[str] | None = None) -> list[str]:
                return ["chunked"] if name == "Transfer-Encoding" else (default or [])

        class Handler:
            headers = Headers()

        handler = Handler()
        handler.rfile = io.BytesIO(b"Z\r\nnot-a-chunk\r\n")
        with self.assertRaisesRegex(CaptureError, "chunk size"):
            _read_request_body(handler, 1024)

        handler.rfile = io.BytesIO(b"3\r\nabc\r\n2;tag=ok\r\nde\r\n0\r\n\r\n")
        self.assertEqual(_read_request_body(handler, 1024), b"abcde")

        handler.rfile = io.BytesIO(b"1\r\nx\r\n0\r\nAuthorization: leaked\r\n\r\n")
        with self.assertRaisesRegex(CaptureError, "trailers are not supported"):
            _read_request_body(handler, 1024)

    def test_downstream_cancel_is_recorded(self) -> None:
        _ChunkedSSEHandler.infinite = True
        _ChunkedSSEHandler.finished = threading.Event()
        proxy = _proxy(self.tmp_path, self.upstream_server, max_response_bytes=64 * 1024 * 1024)
        thread = _serve(proxy)
        body = b'{"model":"example-model","messages":[{"role":"user","content":"cancel"}],"stream":true,"max_tokens":8}'
        conn = socket.create_connection(proxy.address, timeout=5)
        conn.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        headers = bytearray()
        while b"\r\n\r\n" not in headers:
            headers.extend(conn.recv(4096))
        self.assertIn(b"200 OK", headers)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()

        deadline = time.monotonic() + 8
        capsule_dir = None
        while time.monotonic() < deadline:
            found = list((self.tmp_path / "private-capsules").glob("*.capsule"))
            if found:
                capsule_dir = found[0]
                break
            time.sleep(0.02)
        proxy.shutdown()
        thread.join(timeout=3)
        proxy.server_close()
        if capsule_dir is None:
            self.fail("cancelled response did not publish a capsule")
        manifest = json.loads((capsule_dir / "manifest.json").read_text())
        self.assertEqual(manifest["response"]["stream"]["termination"], "downstream_cancelled")


if __name__ == "__main__":
    unittest.main()
