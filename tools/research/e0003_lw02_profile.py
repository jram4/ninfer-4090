#!/usr/bin/env python3
"""Run one fresh public K7 request, optionally under a short Nsight Systems capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


MODEL_ID = "qwen/qwen3.8-27b"
TEMPERATURE = 1.0
SEED = 736_205_0925
MAX_TOKENS = 128
PROMPT = (
    "Write exactly 12 numbered acceptance criteria for a single-GPU inference server. "
    "Cover bounded FIFO ingress, decode-batch formation, CUDA Graph lifetime, mutable-state "
    "ownership, cancellation, backpressure, stochastic token sampling, cache reuse, error "
    "handling, shutdown, request logging, and health reporting. Use one short sentence per item."
)


def unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def request_once(port: int) -> tuple[dict[str, object], float]:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read())
    elapsed = time.perf_counter() - started
    choice = result["choices"][0]
    content = choice.get("message", {}).get("content", "")
    summary: dict[str, object] = {
        "http_status": 200,
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "thinking_enabled": False,
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage"),
        "completion_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    return summary, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "profile"))
    parser.add_argument("--binary", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    binary = Path(args.binary).resolve(strict=True)
    artifact = Path(args.artifact).resolve(strict=True)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out, 0o700)
    port = unused_loopback_port()
    request_log = out / "request-log.jsonl"
    server_log = out / "server.log"
    server = [
        str(binary),
        str(artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        MODEL_ID,
        "--max-context",
        "188416",
        "--kv-capacity",
        "188416",
        "--kv-dtype",
        "k8v4",
        "--max-concurrency",
        "1",
        "--max-pending-requests",
        "16",
        "--pending-timeout-ms",
        "600000",
        "--prefill-chunk",
        "1024",
        "--spec",
        "mtp",
        "--draft-tokens",
        "7",
        "--lm-head-draft",
        "--preserve-thinking",
        "--default-max-tokens",
        "8192",
        "--request-log-jsonl",
        str(request_log),
    ]
    profile_prefix = out / "round-profile"
    if args.mode == "profile":
        command = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--trace=cuda,nvtx,osrt",
            "--cuda-graph-trace=node:host-only",
            "--sample=none",
            "--cpuctxsw=none",
            "--output",
            str(profile_prefix),
            *server,
        ]
    else:
        command = server

    measurement: dict[str, object] = {
        "mode": args.mode,
        "server_argv": server,
        "server_binary": str(binary),
        "artifact": str(artifact),
        "model_id": MODEL_ID,
        "max_context": 188416,
        "kv_capacity": 188416,
        "kv_dtype": "k8v4",
        "max_concurrency": 1,
        "prefill_chunk": 1024,
        "speculative_backend": "mtp",
        "draft_window": 7,
        "proposal_head": "lm-head-draft",
        "prefix_reuse": "enabled default; no prior request in fresh server",
        "cuda_graph": True,
        "port": port,
        "profile_command": command if args.mode == "profile" else None,
    }

    with server_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(900):
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=2
                    ):
                        break
                except Exception:
                    if process.poll() is not None:
                        raise RuntimeError("server exited before becoming ready")
                    time.sleep(1)
            else:
                raise TimeoutError("server did not become ready within 900 seconds")

            request_result, request_wall = request_once(port)
            measurement["request"] = request_result
            measurement["request_wall_seconds"] = request_wall
        except urllib.error.HTTPError as error:
            measurement["request_error"] = {
                "http_status": error.code,
                "body": error.read().decode("utf-8", errors="replace")[:2000],
            }
        except Exception as error:
            measurement["request_error"] = f"{type(error).__name__}: {error}"
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=30)
        measurement["server_exit_code"] = process.returncode

    if args.mode == "profile" and (out / "round-profile.nsys-rep").exists():
        subprocess.run(
            [
                "nsys",
                "export",
                "--type",
                "sqlite",
                "--force-overwrite=true",
                "--output",
                str(out / "round-profile.sqlite"),
                str(out / "round-profile.nsys-rep"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    measurement_path = out / "measurement.json"
    measurement_path.write_text(json.dumps(measurement, indent=2, sort_keys=True) + "\n")
    os.chmod(measurement_path, 0o600)
    print(json.dumps(measurement, indent=2, sort_keys=True))
    return 0 if "request_error" not in measurement and measurement["server_exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
