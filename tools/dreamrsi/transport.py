"""SSH transport for frozen candidates and the separately trusted evaluator.

Importing this module is side-effect free. ``installation_plan`` is an exact
operator plan; it never opens SSH. Live commands exist for a later explicit
coordinator invocation and are exercised here only through an injected fake
runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping
import uuid

from . import contracts


DEFAULT_REMOTE_PYTHON = "/home/aramirezfamily/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
PRIVATE_RESEARCH_ROOT = PurePosixPath("/home/aramirezfamily/.local/share/ninfer-bench")
_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_SAFE_BASENAME = re.compile(r"[^A-Za-z0-9._-]+")
_REMOTE_HASH_SCRIPT = r"""import hashlib,json,os,stat,sys
root=os.path.realpath(sys.argv[1])
names=json.load(sys.stdin)
result={}
for name in names:
 p=os.path.join(root,*name.split('/'))
 info=os.lstat(p)
 if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
  raise SystemExit(3)
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''): h.update(block)
 result[name]=h.hexdigest()
print(json.dumps(result,sort_keys=True,separators=(',',':')))
"""
_REMOTE_SUPERVISOR = r"""import os,signal,subprocess,sys,time
cancel_path=sys.argv[1]
command=sys.argv[2:]
if not command: raise SystemExit(126)
child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
sent=False
while child.poll() is None:
 if os.path.exists(cancel_path) and not sent:
  try: os.killpg(child.pid,signal.SIGINT)
  except ProcessLookupError: pass
  sent=True
 time.sleep(0.1)
out,err=child.communicate()
sys.stdout.buffer.write(out)
sys.stderr.buffer.write(err)
sys.exit(child.returncode if child.returncode is not None else 1)
"""


class TransportError(RuntimeError):
    """Transport setup, SSH execution, or evidence verification failed."""


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def _default_runner(
    argv: list[str],
    *,
    input: bytes | None = None,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        input=input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _safe_remote_root(value: str | Path) -> PurePosixPath:
    raw = str(value)
    root = PurePosixPath(raw)
    if (
        not root.is_absolute()
        or ".." in root.parts
        or "\\" in raw
        or any(ord(character) < 0x20 for character in raw)
        or not (root == PRIVATE_RESEARCH_ROOT or PRIVATE_RESEARCH_ROOT in root.parents)
    ):
        raise ValueError(
            f"remote_root must be an explicit private path under {PRIVATE_RESEARCH_ROOT}"
        )
    return root


def _within(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path == root or root in path.parents


def _read_harness_files(root: Path, contracts_path: Path | None) -> tuple[Path, dict[str, Path]]:
    project_package = root / "tools" / "dreamrsi"
    package = project_package if (project_package / "evaluator.py").is_dir() else root
    # evaluator.py is a file; use the package root when it contains the full
    # trusted runner, or accept a repository root with tools/dreamrsi beneath it.
    if not (package / "evaluator.py").is_file() and (root / "tools" / "dreamrsi" / "evaluator.py").is_file():
        package = root / "tools" / "dreamrsi"
    selected_contracts = contracts_path or Path(__file__).with_name("contracts.py")
    files = {
        "evaluator.py": package / "evaluator.py",
        "lifecycle.py": package / "lifecycle.py",
        "contracts.py": selected_contracts,
    }
    for relative, path in files.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"trusted harness file is missing or unsafe: {relative}")
    return package.resolve(strict=True), files


def _hash_harness(files: Mapping[str, Path]) -> tuple[str, list[dict[str, str]]]:
    manifest = [
        {"path": relative, "sha256": contracts.sha256_file(path)}
        for relative, path in sorted(files.items())
    ]
    return contracts.canonical_hash(manifest), manifest


def _run_argv(argv: list[str], *, input: bytes | None = None, timeout: float) -> subprocess.CompletedProcess[bytes]:
    return _default_runner(argv, input=input, timeout=timeout)


def _result_bytes(result: subprocess.CompletedProcess[bytes], stream: str) -> bytes:
    value = getattr(result, stream, b"")
    if isinstance(value, str):
        return value.encode("utf-8")
    return value or b""


class SSHExecutor:
    """Execute a frozen submission through a preinstalled trusted remote runner.

    The candidate archive/metadata, immutable evaluator modules, and trusted
    policy occupy distinct remote trees. The runner validates the FNN GPU lock
    from policy and the evaluator owns the actual admission/flock. SSH commands
    never use candidate-provided commands or shell fragments.
    """

    def __init__(
        self,
        remote_root: str | Path,
        *,
        trusted_harness: str | Path,
        alias: str = "fnn",
        remote_python: str = DEFAULT_REMOTE_PYTHON,
        contracts_path: str | Path | None = None,
        runner: Runner | None = None,
        timeout_seconds: float = 24 * 60 * 60,
    ) -> None:
        if not isinstance(alias, str) or not _ALIAS.fullmatch(alias):
            raise ValueError("SSH alias must be an explicit safe alias")
        if not isinstance(remote_python, str) or not PurePosixPath(remote_python).is_absolute() or "\x00" in remote_python:
            raise ValueError("remote_python must be an explicit absolute Python 3.11 path")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.alias = alias
        self.remote_root = _safe_remote_root(remote_root)
        self.remote_python = remote_python
        source_root = Path(trusted_harness).expanduser().resolve(strict=True)
        selected_contracts = Path(contracts_path).expanduser().resolve(strict=True) if contracts_path else None
        self.harness_package, self.harness_files = _read_harness_files(source_root, selected_contracts)
        self.harness_id, self.harness_manifest = _hash_harness(self.harness_files)
        self.remote_harness_root = self.remote_root / "TRUSTEDHARNESS" / self.harness_id
        self.runner = runner
        self.timeout_seconds = float(timeout_seconds)
        self.last_receipt_path: Path | None = None
        self.last_receipt_root: Path | None = None

    def _run(self, argv: list[str], *, input: bytes | None = None,
             timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
        selected = self.runner or _run_argv
        try:
            return selected(argv, input=input, timeout=timeout or self.timeout_seconds)
        except (OSError, subprocess.SubprocessError) as exc:
            raise TransportError(f"transport command failed: {type(exc).__name__}") from exc

    def _ssh(self, remote_argv: list[str], *, input: bytes | None = None,
             timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
        command = shlex.join(remote_argv)
        return self._run(["ssh", "-T", self.alias, command], input=input, timeout=timeout)

    def _scp_upload(self, local: Path, remote: PurePosixPath) -> None:
        result = self._run(["scp", "-p", "--", str(local), f"{self.alias}:{remote.as_posix()}"],
                           timeout=self.timeout_seconds)
        if result.returncode != 0:
            raise TransportError(f"SCP upload failed with exit code {result.returncode}")

    def _scp_download(self, remote: PurePosixPath, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        result = self._run(["scp", "--", f"{self.alias}:{remote.as_posix()}", str(local)],
                           timeout=self.timeout_seconds)
        if result.returncode != 0:
            raise TransportError(f"SCP download failed with exit code {result.returncode}")
        if local.is_symlink() or not local.is_file():
            raise TransportError("SCP did not produce a regular local file")

    def installation_plan(self) -> dict[str, Any]:
        """Return a manual, non-executing trusted-runner installation plan."""
        commands = [
            {
                "argv": ["ssh", "-T", self.alias,
                         shlex.join(["mkdir", "-m", "700", "-p", "--", self.remote_harness_root.as_posix()])],
                "purpose": "create isolated content-addressed TRUSTEDHARNESS directory",
            }
        ]
        for relative, source in sorted(self.harness_files.items()):
            destination = self.remote_harness_root / relative
            commands.append({
                "argv": ["scp", "-p", "--", str(source), f"{self.alias}:{destination.as_posix()}"],
                "sha256": contracts.sha256_file(source),
                "purpose": f"install trusted {relative}; never from candidate archive",
            })
        commands.append({
            "argv": ["ssh", "-T", self.alias, shlex.join(
                ["chmod", "400"] + [(self.remote_harness_root / name).as_posix()
                                   for name in sorted(self.harness_files)]
            )],
            "purpose": "seal trusted evaluator modules read-only",
        })
        return {
            "dry_run": True,
            "mutated": False,
            "alias": self.alias,
            "remote_root": self.remote_root.as_posix(),
            "trusted_harness_root": self.remote_harness_root.as_posix(),
            "remote_python": self.remote_python,
            "harness_id": self.harness_id,
            "files": self.harness_manifest,
            "commands": commands,
            "operator_action": "install only if the matching trusted runner is not already present",
        }

    def _verify_remote_harness(self) -> None:
        spec = {entry["path"]: entry["sha256"] for entry in self.harness_manifest}
        result = self._ssh(
            [self.remote_python, "-c", _REMOTE_HASH_SCRIPT, self.remote_harness_root.as_posix()],
            input=json.dumps(list(spec), separators=(",", ":")).encode("utf-8"),
            timeout=30,
        )
        if result.returncode != 0:
            raise TransportError(
                "matching trusted runner is not installed on FNN; see installation_plan()"
            )
        try:
            observed = json.loads(_result_bytes(result, "stdout"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TransportError("remote trusted-runner verification returned invalid JSON") from exc
        if observed != spec:
            raise TransportError("remote trusted-runner files differ; see installation_plan()")

    def plan_submission(self, submission: dict[str, Any], policy: dict[str, Any],
                        run_root: Path) -> dict[str, Any]:
        normalized_submission = contracts.validate_submission(submission)
        normalized_policy = contracts.validate_policy(policy)
        if normalized_submission["policy_id"] != normalized_policy["policy_id"]:
            raise ValueError("submission policy_id differs from the trusted policy")
        if normalized_submission["contract_id"] != normalized_policy["contract_id"]:
            raise ValueError("submission contract_id differs from the trusted policy")
        attempt_id = uuid.uuid4().hex
        package_root = self.remote_root / "candidate-packages" / normalized_submission["submission_id"] / attempt_id
        archive_path = package_root / "source.tar"
        metadata_path = package_root / "submission.json"
        policy_path = self.remote_root / "TRUSTEDHARNESS" / "policies" / normalized_policy["policy_id"] / f"{attempt_id}.json"
        remote_run_root = self.remote_root / "runs" / normalized_submission["submission_id"] / attempt_id
        return {
            "dry_run": True,
            "mutated": False,
            "alias": self.alias,
            "harness_id": self.harness_id,
            "candidate_archive": archive_path.as_posix(),
            "submission_metadata": metadata_path.as_posix(),
            "trusted_policy": policy_path.as_posix(),
            "trusted_harness": self.remote_harness_root.as_posix(),
            "run_root": remote_run_root.as_posix(),
            "identity_change": "source.archive_path only; submission_id and snapshot/archive hashes are unchanged",
        }

    def __call__(self, submission: dict[str, Any], policy: dict[str, Any], run_root: Path,
                 *, cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
        normalized_submission = contracts.validate_submission(submission)
        normalized_policy = contracts.validate_policy(policy)
        if normalized_submission["policy_id"] != normalized_policy["policy_id"]:
            raise TransportError("submission policy_id differs from trusted policy")
        if normalized_submission["contract_id"] != normalized_policy["contract_id"]:
            raise TransportError("submission contract_id differs from trusted policy")
        if cancelled():
            raise TransportError("evaluation cancelled before SSH setup")
        self._verify_remote_harness()
        plan = self.plan_submission(normalized_submission, normalized_policy, run_root)
        remote_package = PurePosixPath(plan["submission_metadata"]).parent
        remote_archive = PurePosixPath(plan["candidate_archive"])
        remote_submission_path = PurePosixPath(plan["submission_metadata"])
        remote_policy_path = PurePosixPath(plan["trusted_policy"])
        remote_run_root = PurePosixPath(plan["run_root"])

        local_root = Path(run_root).expanduser().absolute()
        local_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging = local_root / "transport-staging" / normalized_submission["submission_id"] / uuid.uuid4().hex
        staging.mkdir(parents=True, exist_ok=False, mode=0o700)
        local_archive = Path(normalized_submission["source"]["archive_path"])
        expected_archive_hash = normalized_submission["source"]["archive_sha256"]
        if local_archive.is_symlink() or not local_archive.is_file() or contracts.sha256_file(local_archive) != expected_archive_hash:
            raise TransportError("local frozen source archive is missing, linked, or changed")

        remote_submission = json.loads(json.dumps(normalized_submission))
        remote_submission["source"]["archive_path"] = remote_archive.as_posix()
        # The core submission identity explicitly excludes only this transport path.
        remote_submission = contracts.validate_submission(remote_submission)
        local_submission_file = staging / "submission.json"
        local_policy_file = staging / "policy.json"
        contracts.write_json(local_submission_file, remote_submission)
        contracts.write_json(local_policy_file, normalized_policy)

        directory_result = self._ssh(["mkdir", "-m", "700", "-p", "--",
                                      remote_package.as_posix(), remote_policy_path.parent.as_posix(),
                                      remote_run_root.as_posix()], timeout=30)
        if directory_result.returncode != 0:
            raise TransportError(f"remote private directory setup failed with exit code {directory_result.returncode}")
        self._scp_upload(local_archive, remote_archive)
        self._scp_upload(local_submission_file, remote_submission_path)
        self._scp_upload(local_policy_file, remote_policy_path)
        seal_result = self._ssh(["chmod", "400", remote_archive.as_posix(),
                                 remote_submission_path.as_posix(), remote_policy_path.as_posix()], timeout=30)
        if seal_result.returncode != 0:
            raise TransportError(f"could not seal remote package/policy files (exit {seal_result.returncode})")
        if cancelled():
            raise TransportError("evaluation cancelled before remote evaluator invocation")

        remote_cancel_file = remote_run_root / "operator-cancel.request"
        evaluator_command = [
            self.remote_python,
            (self.remote_harness_root / "evaluator.py").as_posix(),
            "--submission", remote_submission_path.as_posix(),
            "--policy", remote_policy_path.as_posix(),
            "--run-root", remote_run_root.as_posix(),
        ]
        supervisor_command = [
            self.remote_python,
            "-c",
            _REMOTE_SUPERVISOR,
            remote_cancel_file.as_posix(),
            *evaluator_command,
        ]
        remote_shell = shlex.join(supervisor_command)
        result = self._run_remote_evaluator(remote_shell, cancelled)
        if result.returncode not in {0, 1, 130}:
            raise TransportError(f"remote evaluator SSH session failed with exit code {result.returncode}")
        try:
            remote_receipt = contracts.validate_receipt(json.loads(_result_bytes(result, "stdout")))
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TransportError("remote evaluator did not return a valid receipt") from exc
        _verify_receipt_bindings(remote_receipt, normalized_submission, normalized_policy)

        receipt_root = local_root / "trusted-receipts"
        receipt_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        attempt_name, remote_attempt_root = _remote_attempt_root(remote_run_root, remote_receipt)
        local_attempt = receipt_root / normalized_submission["submission_id"] / attempt_name
        local_attempt.mkdir(parents=True, exist_ok=False, mode=0o700)
        remote_receipt_bytes = _fetch_receipt_bytes(
            self, remote_attempt_root, remote_receipt, _result_bytes(result, "stdout")
        )
        (local_attempt / "remote-receipt.json").write_bytes(remote_receipt_bytes)
        (local_attempt / "remote-receipt.json").chmod(0o400)

        local_refs: list[dict[str, Any]] = []
        for index, reference in enumerate(remote_receipt["evidence"]):
            remote_evidence = PurePosixPath(reference["path"])
            if not _within(remote_evidence, remote_attempt_root):
                raise TransportError("remote receipt evidence escapes evaluator attempt directory")
            name = _SAFE_BASENAME.sub("_", remote_evidence.name) or f"artifact-{index}"
            relative = PurePosixPath(normalized_submission["submission_id"], attempt_name,
                                     "evidence", f"{index:03d}-{name}")
            local_path = receipt_root.joinpath(*relative.parts)
            self._scp_download(remote_evidence, local_path)
            if contracts.sha256_file(local_path) != reference["sha256"]:
                raise TransportError(f"remote evidence hash mismatch: {reference['path']}")
            local_path.chmod(0o400)
            copied_reference = dict(reference)
            copied_reference["path"] = relative.as_posix()
            local_refs.append(copied_reference)

        remote_receipt_path = remote_attempt_root / "receipt.json"
        remote_receipt_local = local_attempt / "fetched-receipt.json"
        if remote_attempt_root != remote_run_root:
            self._scp_download(remote_receipt_path, remote_receipt_local)
            if contracts.canonical_hash(contracts.read_json(remote_receipt_local)) != contracts.canonical_hash(remote_receipt):
                raise TransportError("fetched remote receipt differs from evaluator stdout receipt")
            remote_receipt_local.chmod(0o400)
        else:
            remote_receipt_local.write_bytes(remote_receipt_bytes)
            remote_receipt_local.chmod(0o400)

        local_receipt = json.loads(json.dumps(remote_receipt))
        local_receipt["evidence"] = local_refs
        normalized_local_receipt = contracts.validate_receipt(local_receipt)
        local_receipt_path = local_attempt / "receipt.json"
        contracts.write_json(local_receipt_path, normalized_local_receipt)
        local_receipt_path.chmod(0o400)
        self.last_receipt_root = receipt_root
        self.last_receipt_path = local_receipt_path
        return normalized_local_receipt

    def _run_remote_evaluator(self, remote_command: str,
                              cancelled: Callable[[], bool]) -> subprocess.CompletedProcess[bytes]:
        argv = ["ssh", "-T", self.alias, remote_command]
        if self.runner is not None:
            if cancelled():
                raise TransportError("remote evaluation cancelled before launch")
            return self._run(argv, timeout=self.timeout_seconds)

        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise TransportError(f"could not start SSH client: {type(exc).__name__}") from exc
        cancel_sent = False
        try:
            while process.poll() is None:
                if cancelled() and not cancel_sent:
                    cancel_sent = True
                    # The remote supervisor forwards this cancellation as SIGINT
                    # to the evaluator process group, whose lifecycle tears down
                    # the active stage. We never terminate a host service.
                    self._ssh(["touch", "--", self._active_cancel_path], timeout=15)
                time.sleep(0.2)
            stdout, stderr = process.communicate(timeout=30)
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise

    @property
    def _active_cancel_path(self) -> str:
        if self._cancel_path is None:
            raise TransportError("remote cancellation path was not initialized")
        return self._cancel_path


def _verify_receipt_bindings(receipt: dict[str, Any], submission: dict[str, Any], policy: dict[str, Any]) -> None:
    expected = {
        "submission_id": submission["submission_id"],
        "policy_id": policy["policy_id"],
        "contract_id": policy["contract_id"],
        "source_snapshot_sha256": submission["source"]["snapshot_sha256"],
        "candidate_commit": submission.get("candidate_commit"),
        "base_commit": submission["base_commit"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise TransportError(f"remote receipt does not bind {field}")


def _remote_attempt_root(remote_run_root: PurePosixPath,
                         receipt: dict[str, Any]) -> tuple[str, PurePosixPath]:
    evidence = receipt.get("evidence", [])
    if not evidence:
        return "no-attempt", remote_run_root
    attempt_names: set[str] = set()
    for reference in evidence:
        path = PurePosixPath(reference["path"])
        if not _within(path, remote_run_root):
            raise TransportError("remote evidence path is outside configured run_root")
        relative = path.relative_to(remote_run_root)
        if len(relative.parts) < 3 or relative.parts[0] != "attempts":
            raise TransportError("remote evidence path is not within an evaluator attempt")
        if ".." in relative.parts or any(not part for part in relative.parts):
            raise TransportError("remote evidence path is not canonical")
        attempt_names.add(relative.parts[1])
    if len(attempt_names) != 1:
        raise TransportError("remote receipt evidence spans multiple evaluator attempts")
    attempt_name = next(iter(attempt_names))
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", attempt_name):
        raise TransportError("remote evaluator attempt directory has an unsafe name")
    return attempt_name, remote_run_root / "attempts" / attempt_name


def _fetch_receipt_bytes(executor: SSHExecutor, attempt_root: PurePosixPath,
                         receipt: dict[str, Any], stdout_bytes: bytes) -> bytes:
    if not receipt.get("evidence"):
        return stdout_bytes
    local_temp = Path(executor.last_local_attempt) / "remote-receipt.tmp"
    executor._scp_download(attempt_root / "receipt.json", local_temp)
    raw = local_temp.read_bytes()
    try:
        fetched = contracts.validate_receipt(json.loads(raw))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TransportError("remote receipt file is invalid JSON") from exc
    if contracts.canonical_hash(fetched) != contracts.canonical_hash(receipt):
        raise TransportError("fetched remote receipt differs from evaluator stdout receipt")
    local_temp.unlink()
    return raw


def installation_plan(
    *,
    remote_root: str | Path,
    trusted_harness: str | Path,
    alias: str = "fnn",
    contracts_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create an exact non-executing installation plan for the trusted runner."""
    executor = SSHExecutor(
        remote_root,
        trusted_harness=trusted_harness,
        alias=alias,
        contracts_path=contracts_path,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""),
    )
    return executor.installation_plan()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--trusted-harness", type=Path, required=True)
    parser.add_argument("--contracts", type=Path)
    parser.add_argument("--ssh-alias", default="fnn")
    args = parser.parse_args(argv)
    try:
        plan = installation_plan(
            remote_root=args.remote_root,
            trusted_harness=args.trusted_harness,
            alias=args.ssh_alias,
            contracts_path=args.contracts,
        )
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, TransportError) as exc:
        print(f"transport.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
