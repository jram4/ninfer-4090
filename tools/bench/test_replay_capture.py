from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from tools.bench import replay_capture


class ReplayServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler]) -> None:
        super().__init__(address, handler)
        self.seen: list[dict[str, object]] = []


class ReplayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        body_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(body_length)
        self.server.seen.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "body": body,
                "x_api_key": self.headers.get("x-api-key"),
                "authorization": self.headers.get("Authorization"),
                "content_type": self.headers.get("Content-Type"),
                "anthropic_beta": self.headers.get("anthropic-beta"),
            }
        )
        if self.path == "/v1/messages?beta=true":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            events = [
                b'event: message_start\r\ndata: {"type":"message_start","message":{"usage":{"output_tokens":0}}}\r\n\r\n',
                b'event: content_block_delta\r\ndata: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"private generated phrase"}}\r\n\r\n',
                b'event: message_delta\r\ndata: {"type":"message_delta","usage":{"output_tokens":3}}\r\n\r\n',
                b'event: message_stop\r\ndata: {"type":"message_stop"}\r\n\r\n',
            ]
            for event in events:
                self.wfile.write(f"{len(event):X}\r\n".encode("ascii"))
                self.wfile.write(event)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        payload = {
            "usage": {"prompt_tokens": 4, "completion_tokens": 5},
            "timings": {"predicted_n": 5, "predicted_per_second": 33.5},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ReplayCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.server = ReplayServer(("127.0.0.1", 0), ReplayHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()
        self.temp_dir.cleanup()

    def _write_capture(self, rows: list[tuple[int, str, bytes, dict[str, str]]]) -> Path:
        bodies = self.root / "bodies"
        bodies.mkdir(exist_ok=True)
        manifest_rows = []
        for seq, path, body, headers in rows:
            name = f"{seq}.body"
            (bodies / name).write_bytes(body)
            manifest_rows.append(
                {
                    "seq": seq,
                    "timestamp": "2026-09-23T20:00:00Z",
                    "method": "POST",
                    "path": path,
                    "headers": headers,
                    "body_file": f"bodies/{name}",
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "request_bytes": len(body),
                    "response_status": 200,
                }
            )
        manifest = self.root / "manifest.jsonl"
        # Deliberately write out of order to verify capture-sequence ordering.
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )
        return manifest

    def test_replay_selection_raw_bytes_sse_auth_and_posthoc_metrics(self) -> None:
        anthropic_body = (
            b'{"model":"qwen","stream":true,"messages":[{"role":"user",'
            b'"content":"TOPSECRET PROMPT"}]}'
        )
        chat_body = (
            b'{"model":"qwen","seed":17,"messages":[{"role":"user",'
            b'"content":"PRIVATE CHAT PROMPT"}]}'
        )
        synthetic_body = b'{"messages":[{"role":"user","content":"excluded"}]}'
        manifest = self._write_capture(
            [
                (3, "/v1/chat/completions", chat_body, {"content-type": "application/json"}),
                (
                    1,
                    "/v1/messages",
                    synthetic_body,
                    {"content-type": "application/json"},
                ),
                (
                    2,
                    "/v1/messages?beta=true",
                    anthropic_body,
                    {
                        "content-type": "application/json",
                        "accept": "text/event-stream",
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": "prompt-caching-2024-07-31",
                        "authorization": "must-not-be-forwarded",
                    },
                ),
            ]
        )
        output = self.root / "results.jsonl"
        with patch.dict(os.environ, {"REPLAY_TEST_TOKEN": "do-not-print-this-token"}):
            result = replay_capture.main(
                [
                    "--manifest",
                    str(manifest),
                    "--base-url",
                    self.base_url,
                    "--output",
                    str(output),
                    "--exclude-seq",
                    "1,5,6",
                    "--api-key-env",
                    "REPLAY_TEST_TOKEN",
                    "--server-default-seed",
                    "42",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(len(self.server.seen), 2)
        self.assertEqual(
            [item["path"] for item in self.server.seen],
            ["/v1/messages?beta=true", "/v1/chat/completions"],
        )
        self.assertEqual(self.server.seen[0]["body"], anthropic_body)
        self.assertEqual(self.server.seen[0]["x_api_key"], "do-not-print-this-token")
        self.assertIsNone(self.server.seen[0]["authorization"])
        self.assertEqual(self.server.seen[0]["anthropic_beta"], "prompt-caching-2024-07-31")
        self.assertEqual(self.server.seen[1]["body"], chat_body)
        self.assertEqual(self.server.seen[1]["authorization"], "Bearer do-not-print-this-token")

        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["seq"] for row in records], [2, 3])
        self.assertEqual(records[0]["http_status"], 200)
        self.assertIsNotNone(records[0]["client_time_to_first_byte_seconds"])
        self.assertIsNotNone(records[0]["client_time_to_first_token_seconds"])
        self.assertEqual(records[0]["output_tokens"], 3)
        self.assertEqual(records[0]["seed_mode"], "server_default")
        self.assertEqual(records[0]["seed_value"], 42)
        self.assertFalse(records[0]["request_modified"])
        self.assertEqual(records[1]["seed_mode"], "request")
        self.assertEqual(records[1]["seed_value"], 17)
        self.assertEqual(records[1]["decode_tok_per_s"], 33.5)
        self.assertEqual(records[1]["decode_tok_per_s_source"], "response_metrics")
        serialized = output.read_text(encoding="utf-8")
        for private_value in (
            "TOPSECRET PROMPT",
            "PRIVATE CHAT PROMPT",
            "private generated phrase",
            "do-not-print-this-token",
        ):
            self.assertNotIn(private_value, serialized)

        server_log = self.root / "server.jsonl"
        events = [
            {"event": "server_start"},
            {
                "event": "request_start",
                "server_instance_id": "test-instance",
                "request": {"request_id": 101},
            },
            {
                "event": "request_done",
                "server_instance_id": "test-instance",
                "request": {"request_id": 101},
                "result": {"prompt_tokens": 11, "completion_tokens": 7},
                "timings_seconds": {
                    "decode": 0.2,
                    "ttft": 0.4,
                    "total": 0.9,
                    "prefill": 0.5,
                },
                "speculative": {
                    "accepted_per_position": [2, 1, 0],
                    "accepted_tokens": 3,
                    "drafted_tokens": 8,
                    "rounds": 4,
                    "draft_window": 3,
                    "fallback_steps": 0,
                    "backend": "mtp",
                },
            },
            {
                "event": "request_start",
                "server_instance_id": "test-instance",
                "request": {"request_id": 102},
            },
            {
                "event": "request_done",
                "server_instance_id": "test-instance",
                "request": {"request_id": 102},
                "result": {"prompt_tokens": 9, "completion_tokens": 5},
                "timings_seconds": {"decode": 0.1, "ttft": 0.2, "total": 0.4, "prefill": 0.2},
                "speculative": {"accepted_per_position": [1], "rounds": 3},
            },
        ]
        server_log.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        merge_result = replay_capture.main(
            ["--merge-server-log", str(server_log), "--results", str(output)]
        )
        self.assertEqual(merge_result, 0)
        enriched = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(enriched[0]["server_log_match"], "request_done")
        self.assertEqual(enriched[0]["server_metrics"]["completion_tokens"], 7)
        self.assertEqual(enriched[0]["server_metrics"]["timings_seconds"]["decode"], 0.2)
        self.assertEqual(enriched[0]["server_metrics"]["timings_seconds"]["ttft"], 0.4)
        self.assertEqual(enriched[0]["server_metrics"]["speculative"]["accepted_per_position"], [2, 1, 0])
        self.assertEqual(enriched[0]["server_metrics"]["speculative"]["rounds"], 4)
        self.assertEqual(enriched[0]["decode_tok_per_s"], 30.0)
        self.assertEqual(enriched[0]["decode_tok_per_s_source"], "server_request_log")


if __name__ == "__main__":
    unittest.main()
