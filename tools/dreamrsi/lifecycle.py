"""Bounded process, HTTP, and model-GPU admission primitives for DreamRSI.

Model-backed stages are admitted only while configured production units are
verified inactive, an advisory local flock is held, GPU inventory is readable,
and no other compute process is present.  The flock coordinates research
workers; it is deliberately not treated as production isolation by itself.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


Cancelled = Callable[[], bool]


class LifecycleError(RuntimeError):
    """A process or host lifecycle condition prevented a trustworthy run."""


class ProductionServiceActive(LifecycleError):
    """A configured production service is active; do not admit a GPU stage."""


class GpuAdmissionError(LifecycleError):
    """GPU exclusivity or target-device preflight could not be established."""


class ProcessTimeout(LifecycleError):
    def __init__(self, command: Sequence[str], cleanup: dict[str, Any]) -> None:
        self.command = list(command)
        self.cleanup = cleanup
        super().__init__(f"process exceeded its timeout (exit={cleanup.get('returncode')})")


class ProcessCancelled(LifecycleError):
    def __init__(self, cleanup: dict[str, Any] | None = None) -> None:
        self.cleanup = cleanup
        super().__init__("process cancelled")


class ProcessCleanupError(LifecycleError):
    def __init__(self, cleanup: dict[str, Any]) -> None:
        self.cleanup = cleanup
        super().__init__("process group remained alive after teardown")


def sanitized_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a child environment without ambient credential-bearing variables."""
    source = os.environ if base is None else base
    secret_markers = ("API_KEY", "SECRET", "TOKEN", "CREDENTIAL", "PASSWORD", "AUTH")
    result: dict[str, str] = {}
    for name, value in source.items():
        upper = name.upper()
        if upper.startswith(("ANTHROPIC_", "CLAUDE_")):
            continue
        if any(marker in upper for marker in secret_markers):
            continue
        result[name] = value
    return result


@dataclass
class ManagedProcess:
    """A process that owns a fresh session/process group and an immutable log path."""

    process: subprocess.Popen[bytes]
    log_path: Path

    @classmethod
    def start(
        cls,
        command: Sequence[str],
        *,
        log_path: Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> "ManagedProcess":
        argv = _validate_argv(command)
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(log_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "ab", buffering=0) as log_file:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=sanitized_environment(env),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    close_fds=True,
                    start_new_session=True,
                )
        except BaseException:
            log_path.unlink(missing_ok=True)
            raise
        return cls(process=process, log_path=log_path)

    @property
    def pid(self) -> int:
        return self.process.pid

    def is_running(self) -> bool:
        return self.process.poll() is None

    def terminate_group(
        self,
        *,
        timeout_seconds: float = 5.0,
        kill_wait_seconds: float = 2.0,
    ) -> dict[str, Any]:
        """Stop the whole session, escalate if needed, and report teardown evidence."""
        if timeout_seconds < 0 or kill_wait_seconds < 0:
            raise ValueError("process teardown timeouts must be nonnegative")
        term_sent = False
        try:
            os.killpg(self.pid, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass

        deadline = time.monotonic() + timeout_seconds
        while _process_group_exists(self.pid) and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        kill_sent = _process_group_exists(self.pid)
        if kill_sent:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        try:
            self.process.wait(timeout=max(0.1, kill_wait_seconds))
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + kill_wait_seconds
        while _process_group_exists(self.pid) and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        alive = _process_group_exists(self.pid)
        return {
            "pid": self.pid,
            "returncode": self.process.poll(),
            "sigterm_sent": term_sent,
            "sigkill_sent": kill_sent,
            "process_group_alive": alive,
        }


def run_managed(
    command: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    cancelled: Cancelled = lambda: False,
    poll_seconds: float = 0.05,
    terminate_timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Run one argv-only command with a deadline and cancellation-aware group cleanup."""
    argv = _validate_argv(command)
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("command timeout and poll interval must be positive")
    if cancelled():
        raise ProcessCancelled()
    process = ManagedProcess.start(argv, log_path=log_path, cwd=cwd, env=env)
    started = time.monotonic()
    deadline = started + timeout_seconds
    try:
        while True:
            if cancelled():
                cleanup = process.terminate_group(timeout_seconds=terminate_timeout_seconds)
                raise ProcessCancelled(cleanup)
            returncode = process.process.poll()
            if returncode is not None:
                cleanup = process.terminate_group(timeout_seconds=terminate_timeout_seconds)
                if cleanup["process_group_alive"]:
                    raise ProcessCleanupError(cleanup)
                return {
                    "returncode": returncode,
                    "elapsed_seconds": time.monotonic() - started,
                    "cleanup": cleanup,
                    "log_path": str(log_path),
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                cleanup = process.terminate_group(timeout_seconds=terminate_timeout_seconds)
                raise ProcessTimeout(argv, cleanup)
            time.sleep(min(poll_seconds, remaining))
    except (ProcessCancelled, ProcessTimeout, ProcessCleanupError):
        raise
    except BaseException:
        if process.process.poll() is None or _process_group_exists(process.pid):
            process.terminate_group(timeout_seconds=terminate_timeout_seconds)
        raise


def _validate_argv(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("commands must be nonempty argv sequences, never shell strings")
    argv = list(command)
    if any(not isinstance(argument, str) or not argument or "\0" in argument for argument in argv):
        raise ValueError("command argv contains an invalid argument")
    return argv


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class GpuLease:
    """A bounded advisory flock shared by local research workers."""

    def __init__(
        self,
        lock_path: Path,
        *,
        wait_seconds: float = 30.0,
        cancelled: Cancelled = lambda: False,
    ) -> None:
        self.lock_path = Path(lock_path)
        self.wait_seconds = wait_seconds
        self.cancelled = cancelled
        self.fd: int | None = None

    def __enter__(self) -> "GpuLease":
        if self.wait_seconds < 0:
            raise ValueError("GPU lease wait must be nonnegative")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.lock_path, flags, 0o600)
        info = os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(self.fd)
            self.fd = None
            raise GpuAdmissionError("GPU lock path is not a regular file")
        os.fchmod(self.fd, 0o600)
        deadline = time.monotonic() + self.wait_seconds
        try:
            while True:
                if self.cancelled():
                    raise ProcessCancelled()
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise GpuAdmissionError("timed out waiting for the local GPU research flock")
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def assert_service_units_inactive(units: Sequence[str]) -> list[dict[str, str]]:
    if not units:
        raise GpuAdmissionError("GPU policy has no production service units to audit")
    found: list[dict[str, str]] = []
    for unit in units:
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "SubState"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        active = values.get("ActiveState", "unknown")
        sub = values.get("SubState", "unknown")
        found.append({"unit": unit, "active_state": active, "sub_state": sub})
        if result.returncode != 0 or active != "inactive":
            if active == "active":
                raise ProductionServiceActive(f"production service {unit} is active")
            raise GpuAdmissionError(f"could not verify production service {unit} is inactive")
    return found


def gpu_runtime_health() -> dict[str, Any]:
    """Probe whether a fresh CUDA context can actually be created.

    ``nvidia-smi`` and ``cuDriverGetVersion`` can both succeed while a context
    cannot be created, which is exactly the failure mode a mismatched driver
    produces. This probe initializes the driver and counts devices, so a host
    that cannot run a new process is reported as unhealthy rather than being
    mistaken for a candidate failure.
    """
    import ctypes

    result: dict[str, Any] = {"ok": False, "cu_init": None, "device_count": 0, "error": None}
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        result["error"] = f"libcuda.so.1 could not be loaded: {exc}"
        return result
    try:
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.c_char_p]
        driver.cuGetErrorString.restype = ctypes.c_int
    except AttributeError as exc:  # pragma: no cover - defensive
        result["error"] = f"the CUDA driver API is incomplete: {exc}"
        return result
    status = driver.cuInit(0)
    result["cu_init"] = status
    if status != 0:
        message = ctypes.c_char_p()
        driver.cuGetErrorString(status, ctypes.byref(message))
        result["error"] = f"cuInit returned {status}: {message.value.decode('utf-8', 'replace')}"
        return result
    count = ctypes.c_int(0)
    status = driver.cuDeviceGetCount(ctypes.byref(count))
    result["device_count"] = count.value
    if status != 0 or count.value <= 0:
        result["error"] = f"cuDeviceGetCount returned {status} with {count.value} devices"
        return result
    result["ok"] = True
    return result


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return {"available": False, "error": result.stdout.strip()[:500]}
    rows: list[dict[str, Any]] = []
    try:
        for line in result.stdout.splitlines():
            columns = [value.strip() for value in line.split(",")]
            if len(columns) != 6:
                raise ValueError("malformed nvidia-smi GPU inventory")
            rows.append(
                {
                    "index": int(columns[0]),
                    "name": columns[1],
                    "uuid": columns[2],
                    "memory_total_mib": int(columns[3]),
                    "memory_used_mib": int(columns[4]),
                    "utilization_percent": int(columns[5]),
                }
            )
    except ValueError as exc:
        return {"available": False, "error": str(exc)}
    if not rows:
        return {"available": False, "error": "nvidia-smi returned no GPUs"}
    return {"available": True, "gpus": rows}


def gpu_compute_apps() -> list[dict[str, int]] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    apps: list[dict[str, int]] = []
    try:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            columns = [value.strip() for value in line.split(",")]
            if len(columns) != 2:
                return None
            apps.append({"pid": int(columns[0]), "used_gpu_memory_mib": int(columns[1])})
    except ValueError:
        return None
    return apps


@contextmanager
def gpu_admission(
    lock_path: Path,
    production_units: Sequence[str],
    *,
    target_uuid: str | None = None,
    lease_wait_seconds: float = 30.0,
    cancelled: Cancelled = lambda: False,
) -> Iterator[dict[str, Any]]:
    """Admit a model-backed stage only after independent host and GPU checks."""
    services_before = assert_service_units_inactive(production_units)
    with GpuLease(lock_path, wait_seconds=lease_wait_seconds, cancelled=cancelled):
        # Recheck after the flock: the lock does not stop a noncooperating service.
        services_locked = assert_service_units_inactive(production_units)
        inventory = gpu_snapshot()
        if inventory.get("available") is not True:
            raise GpuAdmissionError(f"GPU inventory unavailable: {inventory.get('error', 'unknown error')}")
        gpus = inventory.get("gpus")
        if not isinstance(gpus, list) or not gpus:
            raise GpuAdmissionError("GPU inventory did not identify a target device")
        if target_uuid is not None and not any(
            isinstance(gpu, dict) and gpu.get("uuid") == target_uuid for gpu in gpus
        ):
            raise GpuAdmissionError("configured target GPU UUID is not visible")
        apps = gpu_compute_apps()
        if apps is None:
            raise GpuAdmissionError("could not verify GPU compute-process inventory")
        if apps:
            raise GpuAdmissionError("another compute process is using the target GPU")
        admission = {
            "production_units_before": services_before,
            "production_units_locked": services_locked,
            "gpu_inventory": inventory,
            "compute_apps_before": apps,
        }
        yield admission
        # A service starting during a measurement invalidates the observation.
        assert_service_units_inactive(production_units)
        apps_after = gpu_compute_apps()
        if apps_after is None or apps_after:
            raise GpuAdmissionError("GPU compute-process inventory did not clear after the stage")
        admission["compute_apps_after"] = apps_after


def http_json(
    url: str,
    body: dict[str, Any] | None = None,
    *,
    timeout_seconds: float = 30.0,
    headers: dict[str, str] | None = None,
    max_response_bytes: int = 1_048_576,
) -> tuple[int, dict[str, Any], float]:
    """Perform a bounded JSON HTTP request for explicitly labelled endpoint plumbing."""
    if timeout_seconds <= 0 or max_response_bytes <= 0:
        raise ValueError("HTTP timeout and response bound must be positive")
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = Request(url, data=data, headers=request_headers)
    started = time.monotonic()
    try:
        try:
            response = urlopen(request, timeout=timeout_seconds)
        except HTTPError as exc:
            response = exc
        with response:
            status = response.status
            raw = response.read(max_response_bytes + 1)
    except (URLError, TimeoutError, OSError) as exc:
        raise LifecycleError(f"bounded HTTP request failed: {exc}") from exc
    elapsed = time.monotonic() - started
    if len(raw) > max_response_bytes:
        raise LifecycleError("HTTP response exceeded its configured byte limit")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {"_body_text": raw.decode("utf-8", errors="replace")[:1024]}
    if not isinstance(payload, dict):
        payload = {"_body": payload}
    return status, payload, elapsed


def wait_for_health(
    base_url: str,
    process: ManagedProcess | None = None,
    *,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 0.25,
    cancelled: Cancelled = lambda: False,
) -> dict[str, Any]:
    """Wait for an owned process or labelled endpoint to serve a healthy JSON response."""
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("health timeout and poll interval must be positive")
    deadline = time.monotonic() + timeout_seconds
    last_error = "health endpoint has not returned"
    while time.monotonic() < deadline:
        if cancelled():
            raise ProcessCancelled()
        if process is not None and not process.is_running():
            raise LifecycleError(
                f"server exited before health (exit={process.process.returncode}, log={process.log_path})"
            )
        try:
            status, payload, elapsed = http_json(
                f"{base_url.rstrip('/')}/health", timeout_seconds=min(3.0, timeout_seconds)
            )
            if status == 200 and payload.get("status") == "ok":
                return {"status": status, "body": payload, "elapsed_seconds": elapsed}
            last_error = f"health returned HTTP {status}"
        except LifecycleError as exc:
            last_error = str(exc)
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    raise LifecycleError(f"health timed out after {timeout_seconds}s: {last_error}")


def assert_loopback_listener_stopped(port: int) -> bool:
    """Return false when a TCP listener still accepts connections on loopback."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return False
    except OSError:
        return True
