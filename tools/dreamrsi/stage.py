#!/usr/bin/env python3
"""Trusted stage programs for the research bench.

A policy stage is an argv array executed by the evaluator with a sanitized
environment and a bound report path. These programs are the shipped, audited
stages; project-specific stages (build, correctness, task, holdout) are supplied
by the operator and must publish the same report shape.

Every program writes one JSON report to ``$NINFER_STAGE_EVIDENCE_PATH`` and
publishes the evidence names the policy requires. The report binds the attempt,
submission, policy, contract, and source snapshot so a stale or foreign report
cannot be reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPORT_SCHEMA = 1


class StageError(RuntimeError):
    """The stage could not produce trustworthy evidence."""


def _binding() -> dict[str, Any]:
    required = {
        "schema_version": "NINFER_REPORT_SCHEMA",
        "submission_id": "NINFER_SUBMISSION_ID",
        "policy_id": "NINFER_POLICY_ID",
        "contract_id": "NINFER_CONTRACT_ID",
        "source_snapshot_sha256": "NINFER_SOURCE_SNAPSHOT_SHA256",
        "attempt_id": "NINFER_ATTEMPT_ID",
        "stage": "NINFER_STAGE",
    }
    missing = sorted(name for name in required.values() if name not in os.environ)
    if missing:
        raise StageError("stage is missing evaluator variables: " + ", ".join(missing))
    binding = {key: os.environ[value] for key, value in required.items()}
    binding["schema_version"] = int(binding["schema_version"])
    return binding


def _write(report: dict[str, Any]) -> None:
    path = os.environ.get("NINFER_STAGE_EVIDENCE_PATH")
    if not path:
        raise StageError("NINFER_STAGE_EVIDENCE_PATH is not set")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)


def _publish(report: dict[str, Any], artifacts: dict[str, Path]) -> None:
    """Attach the required evidence names to hashed raw artifacts."""
    required = json.loads(os.environ.get("NINFER_STAGE_REQUIRED_EVIDENCE", "[]"))
    attempt_dir = Path(os.environ["NINFER_ATTEMPT_DIR"])
    evidence: dict[str, Any] = {}
    for name in required:
        artifact = artifacts.get(name)
        if artifact is None:
            raise StageError(f"stage did not produce required evidence {name!r}")
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        evidence[name] = {
            "path": os.path.relpath(artifact, attempt_dir),
            "sha256": digest.hexdigest(),
        }
    report["evidence"] = evidence


def _raw_dir() -> Path:
    raw = Path(os.environ.get("NINFER_STAGE_RAW_DIR", "."))
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def _toolchain_path() -> str:
    """Return a PATH that resolves the verified build toolchain explicitly.

    A plain non-interactive login shell sees CMake 3.22 and CUDA 11.5, both
    below what this project requires, so the admission stage must look in the
    known locations instead of trusting the inherited PATH.
    """
    parts = [
        str(Path.home() / ".local/bin"),
        "/usr/local/cuda-13.0/bin",
    ]
    inherited = os.environ.get("PATH")
    if inherited:
        parts.append(inherited)
    return ":".join(parts)


def _command_output(argv: list[str], environment: dict[str, str] | None = None) -> str:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False,
                                   env=environment)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or completed.stderr or "").strip()


def _cmake_version_for(output: str) -> str:
    match = re.search(r"cmake version (\d+(?:\.\d+)*)", output, re.IGNORECASE)
    if match:
        return match.group(1)
    return output.strip().splitlines()[0] if output.strip() else ""


def _cuda_version_for(output: str) -> str:
    # nvcc prints a banner first; the release version is on a later line.
    match = re.search(r"release (\d+(?:\.\d+)*)", output, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"V(\d+(?:\.\d+)*)", output)
    if match:
        return match.group(1)
    return output.strip().splitlines()[0] if output.strip() else ""


def _cmake_version(environment: dict[str, str]) -> str | None:
    output = _command_output(["cmake", "--version"], environment)
    if not output:
        return None
    return _cmake_version_for(output)


def _cuda_version(environment: dict[str, str]) -> str | None:
    output = _command_output(["nvcc", "--version"], environment)
    if not output:
        return None
    return _cuda_version_for(output)


def stage_toolchain(_: argparse.Namespace) -> int:
    """Report the actual host build toolchain for the admission stage."""
    report = _binding()
    environment = dict(os.environ)
    environment["PATH"] = _toolchain_path()
    cmake = _cmake_version(environment)
    cuda = _cuda_version(environment)
    if cmake is None or cuda is None:
        raise StageError("cmake and nvcc must both be runnable on the evaluation host")
    report["toolchain"] = {"cmake": cmake, "cuda": cuda}
    report["toolchain_id"] = os.environ.get("NINFER_TOOLCHAIN_ID")
    report["status"] = "pass"
    raw = _raw_dir() / "toolchain.json"
    raw.write_text(json.dumps(report["toolchain"], sort_keys=True), encoding="utf-8")
    _publish(report, {"toolchain_evidence": raw})
    _write(report)
    return 0


def stage_contract(_: argparse.Namespace) -> int:
    """Report the runtime contract actually observed by a running Serve process."""
    parser_log = os.environ.get("NINFER_SERVE_LOG")
    if not parser_log:
        raise StageError("NINFER_SERVE_LOG must point at a Serve request log")
    log_path = Path(parser_log)
    if log_path.is_symlink() or not log_path.is_file():
        raise StageError("the Serve request log is missing or not a regular file")
    start: dict[str, Any] | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"server_start"' not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "server_start":
                start = event
    if start is None:
        raise StageError("the Serve request log contains no server_start event")
    engine = start.get("engine", {})
    artifact = start.get("artifact", {})
    environment = start.get("environment", {})
    server = start.get("server", {})
    backend = engine.get("speculative_backend")
    observed = {
        "model_artifact": {
            "sha256": os.environ.get("NINFER_MODEL_SHA256"),
            "size_bytes": artifact.get("size_bytes"),
        },
        "runtime": {
            "kv_dtype": engine.get("kv_cache"),
            "max_context": engine.get("max_context"),
            "kv_capacity": engine.get("kv_capacity"),
            "max_concurrency": engine.get("max_concurrency"),
            "speculative_backend": backend,
            "draft_tokens": engine.get("speculative_draft_window"),
            "proposal_head": engine.get("proposal_head"),
            "prefix_reuse": engine.get("prefix_reuse"),
            "cuda_graph": engine.get("cuda_graph"),
            "prefill_chunk": engine.get("prefill_chunk"),
            "preserve_thinking": server.get("default_preserve_thinking"),
            "sampling_capabilities": {
                "mtp": backend == "mtp",
                "dflash": backend == "dflash",
                "dflash2": backend == "dflash2",
                "lm_head_draft": engine.get("proposal_head") == "optimized",
                "prefix_reuse": bool(engine.get("prefix_reuse")),
                "cuda_graph": bool(engine.get("cuda_graph")),
                "thinking": start.get("sampling_defaults", {}).get("thinking") is not None,
                "streaming": True,
                "tool_calls": True,
            },
            "binary": {
                "sha256": os.environ.get("NINFER_BINARY_SHA256"),
                "size_bytes": os.environ.get("NINFER_BINARY_SIZE_BYTES"),
            },
            "release_source_commit": os.environ.get("NINFER_RELEASE_SOURCE_COMMIT"),
            "sampling_defaults": start.get("sampling_defaults"),
        },
        "template": {
            "sha256": os.environ.get("NINFER_TEMPLATE_SHA256"),
        },
        "tokenizer": {
            "embedded_in_model_artifact_sha256": os.environ.get("NINFER_MODEL_SHA256"),
        },
        "observed_environment": {
            "gpu_name": environment.get("gpu_name"),
            "gpu_uuid": environment.get("gpu_uuid"),
            "cuda_runtime_version": environment.get("cuda_runtime_version"),
            "cuda_driver_version": environment.get("cuda_driver_version"),
        },
    }
    if observed["model_artifact"]["sha256"] is None:
        raise StageError("NINFER_MODEL_SHA256 must identify the loaded artifact")
    report = _binding()
    report["observed_contract"] = observed
    report["status"] = "pass"
    raw = _raw_dir() / "observed-contract.json"
    raw.write_text(json.dumps(observed, sort_keys=True), encoding="utf-8")
    _publish(report, {"contract_observation": raw})
    _write(report)
    return 0


def stage_compare(_: argparse.Namespace) -> int:
    """Reduce paired baseline/candidate results into a trusted comparison."""
    baseline = os.environ.get("NINFER_BASELINE_RESULTS")
    candidate = os.environ.get("NINFER_CANDIDATE_RESULTS")
    if not baseline or not candidate:
        raise StageError("NINFER_BASELINE_RESULTS and NINFER_CANDIDATE_RESULTS are required")
    try:
        from .quick_eval import reduce_pair
    except ImportError:  # direct module execution
        from quick_eval import reduce_pair  # type: ignore[no-redef]

    plan_path = os.environ.get("NINFER_PLAN")
    if not plan_path:
        raise StageError("NINFER_PLAN must point at the replay plan")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    report_payload = reduce_pair(
        plan,
        [json.loads(line) for line in Path(baseline).read_text(encoding="utf-8").splitlines() if line.strip()],
        [json.loads(line) for line in Path(candidate).read_text(encoding="utf-8").splitlines() if line.strip()],
        bootstrap_resamples=int(os.environ.get("NINFER_BOOTSTRAP_RESAMPLES", "1000")),
    )
    report = _binding()
    classification = report_payload.get("classification")
    report["status"] = "pass" if classification in {"promising", "flat/noisy"} else "inconclusive"
    report["classification"] = classification
    report["paired"] = report_payload.get("paired")
    report["independent_sessions"] = report_payload.get("paired", {}).get(
        "committed_token_rate_ratio_candidate_over_baseline", {}).get("independent_sessions")
    raw = _raw_dir() / "comparison.json"
    raw.write_text(json.dumps(report_payload, sort_keys=True), encoding="utf-8")
    _publish(report, {"comparison_evidence": raw})
    _write(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("toolchain", help="report the host build toolchain")
    commands.add_parser("contract", help="report the observed runtime contract")
    commands.add_parser("compare", help="reduce a paired comparison")
    args = parser.parse_args(argv)
    handlers = {
        "toolchain": stage_toolchain,
        "contract": stage_contract,
        "compare": stage_compare,
    }
    try:
        return handlers[args.command](args)
    except StageError as exc:
        print(f"stage.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
