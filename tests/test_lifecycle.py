from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tools.dreamrsi import lifecycle


def test_run_managed_uses_argv_and_captures_log(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    result = lifecycle.run_managed(
        [sys.executable, "-c", "import os; print('managed child', os.getenv('TEST_SECRET'))"],
        log_path=log,
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "TEST_SECRET": "never-forward"},
        timeout_seconds=5,
    )
    assert result["returncode"] == 0
    assert result["cleanup"]["process_group_alive"] is False
    assert "managed child" in log.read_text(encoding="utf-8")
    assert "never-forward" not in log.read_text(encoding="utf-8")


def test_run_managed_rejects_shell_string(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="argv sequences"):
        lifecycle.run_managed(
            "echo unsafe",  # type: ignore[arg-type]
            log_path=tmp_path / "run.log",
            cwd=tmp_path,
            env={},
            timeout_seconds=1,
        )


def test_cancelled_process_group_is_terminated_and_reaped(tmp_path: Path) -> None:
    child_script = "import time; time.sleep(60)"
    parent_script = "\n".join(
        [
            "import signal,subprocess,sys,time",
            f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}])",
            "def stop(sig, frame):",
            "    child.terminate()",
            "    child.wait()",
            "    raise SystemExit(0)",
            "signal.signal(signal.SIGTERM, stop)",
            "print(child.pid, flush=True)",
            "time.sleep(60)",
        ]
    )
    cancelled = threading.Event()
    timer = threading.Thread(target=lambda: (time.sleep(0.3), cancelled.set()), daemon=True)
    timer.start()
    with pytest.raises(lifecycle.ProcessCancelled) as error:
        lifecycle.run_managed(
            [sys.executable, "-c", parent_script],
            log_path=tmp_path / "cancel.log",
            cwd=tmp_path,
            env={"PATH": os.environ["PATH"]},
            timeout_seconds=10,
            cancelled=cancelled.is_set,
            terminate_timeout_seconds=1,
        )
    timer.join(timeout=1)
    assert error.value.cleanup is not None
    assert error.value.cleanup["process_group_alive"] is False
    assert "process cancelled" in str(error.value)


def test_timeout_kills_own_process_group(tmp_path: Path) -> None:
    with pytest.raises(lifecycle.ProcessTimeout) as error:
        lifecycle.run_managed(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            log_path=tmp_path / "timeout.log",
            cwd=tmp_path,
            env={"PATH": os.environ["PATH"]},
            timeout_seconds=0.2,
            terminate_timeout_seconds=0.2,
        )
    assert error.value.cleanup["process_group_alive"] is False
    assert error.value.cleanup["sigterm_sent"] is True


def test_gpu_admission_rechecks_service_and_compute_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service_calls: list[tuple[str, ...]] = []
    compute_calls = 0

    def services(units: Any) -> list[dict[str, str]]:
        service_calls.append(tuple(units))
        return [{"unit": unit, "active_state": "inactive", "sub_state": "dead"}
                for unit in units]

    def apps() -> list[dict[str, int]]:
        nonlocal compute_calls
        compute_calls += 1
        return []

    monkeypatch.setattr(lifecycle, "assert_service_units_inactive", services)
    monkeypatch.setattr(lifecycle, "gpu_snapshot", lambda: {
        "available": True,
        "gpus": [{"index": 0, "uuid": "GPU-test", "memory_used_mib": 0}],
    })
    monkeypatch.setattr(lifecycle, "gpu_compute_apps", apps)

    with lifecycle.gpu_admission(
        tmp_path / "gpu.lock", ["production.service"], target_uuid="GPU-test"
    ) as admission:
        assert admission["gpu_inventory"]["available"] is True
    assert len(service_calls) == 3  # before, after flock, and after stage
    assert compute_calls == 2
    assert (tmp_path / "gpu.lock").is_file()


def test_gpu_admission_blocks_active_production_before_nvml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def active(_units: Any) -> list[dict[str, str]]:
        raise lifecycle.ProductionServiceActive("production active")

    monkeypatch.setattr(lifecycle, "assert_service_units_inactive", active)
    monkeypatch.setattr(
        lifecycle, "gpu_snapshot",
        lambda: pytest.fail("GPU probe must not run while production is active"),
    )
    with pytest.raises(lifecycle.ProductionServiceActive):
        with lifecycle.gpu_admission(tmp_path / "gpu.lock", ["production.service"]):
            pass


def test_broken_gpu_inventory_is_an_admission_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle, "assert_service_units_inactive", lambda _units: [])
    monkeypatch.setattr(lifecycle, "gpu_snapshot", lambda: {
        "available": False, "error": "NVML driver/library version mismatch"
    })
    with pytest.raises(lifecycle.GpuAdmissionError, match="inventory unavailable"):
        with lifecycle.gpu_admission(tmp_path / "gpu.lock", ["production.service"]):
            pass


class _JsonHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(size))
        body = json.dumps({"received": value}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def test_bounded_json_http_and_endpoint_health_use_local_fake_server() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _JsonHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, payload, elapsed = lifecycle.http_json(
            f"{base_url}/echo", {"case": "fake-http"}, timeout_seconds=2
        )
        assert status == 200
        assert payload == {"received": {"case": "fake-http"}}
        assert elapsed >= 0
        health = lifecycle.wait_for_health(base_url, timeout_seconds=2, poll_seconds=0.01)
        assert health["status"] == 200
        assert health["body"]["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_response_bound_is_enforced() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _JsonHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(lifecycle.LifecycleError, match="byte limit"):
            lifecycle.http_json(f"http://127.0.0.1:{server.server_port}/", max_response_bytes=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
