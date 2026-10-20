from __future__ import annotations

import hashlib
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest

from tools.dreamrsi.capture import _REQUEST_HOP_HEADERS, _capsule_hash, _safe_headers
from tools.dreamrsi.replay import ReplayError, _load_capsule, replay_capsule


class _ReplayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: dict[str, Any] = {}

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).seen = {
            "path": self.path,
            "body": body,
            "headers": {name.lower(): value for name, value in self.headers.items()},
        }
        response = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)
        self.wfile.flush()


def _capsule(
    tmp_path: Path,
    *,
    body: bytes = b'{"model":"test","messages":[],"stream":true,"max_tokens":4}',
    identity_scope: str = "exact_body_no_credentials",
    body_file: str | None = "request.body",
    request_hash: str | None = None,
) -> Path:
    capsule = tmp_path / "fixture.capsule"
    capsule.mkdir(mode=0o700, parents=True)
    response_body = b"data: [DONE]\n\n"
    if body_file:
        (capsule / body_file).write_bytes(body)
    (capsule / "response.body").write_bytes(response_body)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "capsule_id": "fixture",
        "captured_at_utc": "2026-10-20T00:00:00Z",
        "completed_at_utc": "2026-10-20T00:00:01Z",
        "upstream_origin": "http://127.0.0.1:1",
        "cache_state": "unknown",
        "request": {
            "method": "POST",
            "target": "/v1/chat/completions",
            "headers": [{"name": "Content-Type", "value": "application/json"}],
            "body_file": body_file,
            "body_bytes": len(body),
            "body_sha256": request_hash or hashlib.sha256(body).hexdigest() if body_file else None,
            "identity_scope": identity_scope,
            "semantics": {"model": "test", "stream_requested": True, "output_budget": 4},
            "identity": {"capsule_id": "fixture"},
        },
        "response": {
            "status": 200,
            "reason": "OK",
            "headers": [{"name": "Content-Type", "value": "text/event-stream"}],
            "content_type": "text/event-stream",
            "content_length": len(response_body),
            "transfer_encoding": None,
            "body_file": "response.body",
            "body_bytes": len(response_body),
            "body_sha256": hashlib.sha256(response_body).hexdigest(),
            "chunks": [],
            "stream": {"requested": True, "observed": True, "chunk_count": 1, "event_types": [], "terminal_event": "DONE", "termination": "complete"},
            "identity": {},
        },
        "upstream_error": None,
    }
    manifest["capsule_sha256"] = _capsule_hash(manifest)
    (capsule / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return capsule


class ReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)
        _ReplayHandler.seen = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ReplayHandler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tempdir.cleanup()

    def test_replay_is_direct_and_validates_immutable_capsule(self) -> None:
        capsule = _capsule(self.tmp_path)
        output = self.tmp_path / "replayed.body"
        result = replay_capsule(
            capsule,
            upstream_url=f"http://127.0.0.1:{self.server.server_port}",
            response_output=output,
            timeout=5,
        )
        self.assertEqual(_ReplayHandler.seen["path"], "/v1/chat/completions")
        self.assertEqual(_ReplayHandler.seen["body"], b'{"model":"test","messages":[],"stream":true,"max_tokens":4}')
        self.assertIs(result["complete"], True)
        self.assertEqual(result["response_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(output.stat().st_mode & 0o777, 0o400)

    def test_replay_rejects_modified_body_and_bad_manifest_hash(self) -> None:
        capsule = _capsule(self.tmp_path)
        (capsule / "request.body").write_bytes(b"{}")
        with self.assertRaisesRegex(ReplayError, "request body hash"):
            _load_capsule(capsule)
        other = _capsule(self.tmp_path / "other")
        manifest_path = other / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cache_state"] = "warm"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ReplayError, "manifest hash"):
            _load_capsule(other)

    def test_replay_rejects_redacted_or_unsafe_capsule(self) -> None:
        redacted = _capsule(
            self.tmp_path / "redacted",
            identity_scope="redacted_no_body_identity",
            body_file=None,
        )
        with self.assertRaisesRegex(ReplayError, "redacted"):
            _load_capsule(redacted)

        traversal = _capsule(self.tmp_path / "traversal")
        manifest_path = traversal / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["request"]["body_file"] = "../outside"
        manifest["capsule_sha256"] = _capsule_hash(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ReplayError, "body path"):
            _load_capsule(traversal)

    def test_replay_rejects_credentials_before_capsule_hashing(self) -> None:
        capsule = _capsule(self.tmp_path)
        manifest_path = capsule / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["request"]["headers"].extend(
            [
                {"name": "Authorization", "value": "Bearer edited-secret"},
                {"name": "Cookie", "value": "edited-cookie=secret"},
                {"name": "X-Api-Key", "value": "edited-api-secret"},
            ]
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        filtered = _safe_headers(
            [(item["name"], item["value"]) for item in manifest["request"]["headers"]],
            excluded=_REQUEST_HOP_HEADERS,
        )
        self.assertFalse({"authorization", "cookie", "x-api-key"} & {item["name"].lower() for item in filtered})
        with self.assertRaisesRegex(ReplayError, "credential header"):
            replay_capsule(capsule, upstream_url=f"http://127.0.0.1:{self.server.server_port}", timeout=5)


if __name__ == "__main__":
    unittest.main()
