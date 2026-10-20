#!/usr/bin/env python3
"""Small, auditable process and HTTP helpers for an isolated NInfer run."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LifecycleError(RuntimeError):
    pass


class ProductionServiceActive(LifecycleError):
    pass


@dataclass
class ManagedProcess:
    process: subprocess.Popen[bytes]
    log_path: Path

    @classmethod
    def start(cls, command: list[str], *, log_path: Path, cwd: Path | None = None,
              env: dict[str, str] | None = None) -> "ManagedProcess":
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_file = log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()
        return cls(process=process, log_path=log_path)

    @property
    def pid(self) -> int:
        return self.process.pid

    def is_running(self) -> bool:
        return self.process.poll() is None

    def terminate_group(self, *, timeout_seconds: float = 15.0) -> dict[str, Any]:
        """Terminate the complete session and wait for the leader and remaining group."""
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            leader_done = self.process.poll() is not None
            if leader_done and not _process_group_exists(self.pid):
                break
            time.sleep(0.1)
        forced = _process_group_exists(self.pid)
        if forced:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + 2.0
        while _process_group_exists(self.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        return {
            "pid": self.pid,
            "returncode": self.process.poll(),
            "sigkill_sent": forced,
            "process_group_alive": _process_group_exists(self.pid),
        }


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class GpuLease:
    """Hold the same Linux flock lease used by the existing FNN GPU workflow."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.fd: int | None = None

    def __enter__(self) -> "GpuLease":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        self.fd = os.open(self.lock_path, flags)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def http_json(url: str, body: dict[str, Any] | None = None, *, timeout_seconds: float = 30.0,
              headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any], float]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = Request(url, data=data, headers=request_headers)
    start = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            status = response.status
    except HTTPError as exc:
        raw = exc.read()
        status = exc.code
    except (URLError, TimeoutError, OSError) as exc:
        elapsed = time.monotonic() - start
        raise LifecycleError(f"HTTP request failed for {url}: {exc}") from exc
    elapsed = time.monotonic() - start
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = {"_body_text": raw.decode("utf-8", errors="replace")[:1024]}
    if not isinstance(parsed, dict):
        parsed = {"_body": parsed}
    return status, parsed, elapsed


def wait_for_health(base_url: str, process: ManagedProcess, *, timeout_seconds: float = 900.0,
                    poll_seconds: float = 1.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "health endpoint has not returned"
    while time.monotonic() < deadline:
        if not process.is_running():
            raise LifecycleError(
                f"server exited before health (exit={process.process.returncode}, log={process.log_path})"
            )
        try:
            status, payload, elapsed = http_json(f"{base_url}/health", timeout_seconds=3.0)
            if status == 200 and payload.get("status") == "ok":
                return {"status": status, "body": payload, "elapsed_seconds": elapsed}
            last_error = f"health returned HTTP {status}: {payload}"
        except LifecycleError as exc:
            last_error = str(exc)
        time.sleep(poll_seconds)
    raise LifecycleError(f"health timed out after {timeout_seconds}s: {last_error}")


def assert_service_units_inactive(units: list[str]) -> list[dict[str, str]]:
    found = []
    for unit in units:
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "SubState"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        values = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        state = values.get("ActiveState", "unknown")
        found.append({"unit": unit, "active_state": state,
                      "sub_state": values.get("SubState", "unknown"),
                      "show_returncode": str(result.returncode)})
        if result.returncode != 0 or state not in {"active", "inactive", "failed"}:
            raise LifecycleError(f"could not verify {unit} is inactive: {result.stdout.strip()}")
        if state == "active":
            raise ProductionServiceActive(f"refusing to use GPU while {unit} is active")
    return found


def listening_sockets() -> str:
    result = subprocess.run(
        ["ss", "-ltnH"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise LifecycleError(f"ss listener audit failed: {result.stdout.strip()}")
    return result.stdout


def assert_ports_free(ports: tuple[int, ...]) -> str:
    listeners = listening_sockets()
    occupied = []
    for line in listeners.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local = fields[3]
        try:
            port = int(local.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if port in ports:
            occupied.append((port, line))
    if occupied:
        detail = "; ".join(f"{port}: {line}" for port, line in occupied)
        raise LifecycleError(f"research port(s) already have listeners: {detail}")
    return listeners


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return {"available": False, "error": result.stdout.strip()}
    rows = []
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) == 6:
            rows.append({
                "index": int(columns[0]),
                "name": columns[1],
                "uuid": columns[2],
                "memory_total_mib": int(columns[3]),
                "memory_used_mib": int(columns[4]),
                "utilization_percent": int(columns[5]),
            })
    return {"available": True, "gpus": rows}


class GpuSampler:
    """Sample whole-GPU memory while a child server runs."""

    def __init__(self, interval_seconds: float = 0.5,
                 sample: Callable[[], dict[str, Any]] = gpu_snapshot) -> None:
        self.interval_seconds = interval_seconds
        self.sample = sample
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise LifecycleError("GPU sampler is already running")
        self._take()
        self._thread = threading.Thread(target=self._run, name="e0007-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        self._take()
        return list(self.samples)

    def _take(self) -> None:
        value = self.sample()
        value["monotonic_seconds"] = time.monotonic()
        self.samples.append(value)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._take()


def assert_loopback_listener_stopped(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return False
    except OSError:
        return True
