"""Versioned, local contracts shared by the research benchmark tools.

The module intentionally uses only the Python standard library.  Identifiers
are SHA-256 hashes of canonical JSON; a document's own identifier is omitted
when checking that document's identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any


SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_STAGES = ("admission", "build", "correctness", "smoke", "screen", "full")
_SUBMISSION_REQUIRED = {
    "schema_version", "submission_id", "epoch_id", "worker_id",
    "hypothesis_id", "base_commit", "source", "contract_id", "policy_id",
    "created_at", "idempotency_key",
}
_SUBMISSION_OPTIONAL = {"candidate_commit"}
_SOURCE_KEYS = {
    "archive_path", "archive_sha256", "snapshot_sha256", "git_commit",
    "dirty", "manifest",
}
_POLICY_KEYS = {
    "schema_version", "policy_id", "contract_id", "contract", "baseline_id",
    "corpus_id", "toolchain_id", "stages", "qualification_enabled",
}
_RECEIPT_REQUIRED = {
    "schema_version", "submission_id", "policy_id", "contract_id",
    "source_snapshot_sha256", "verdict", "stage", "failure_class",
    "evidence", "created_at",
}
_RECEIPT_OPTIONAL = {
    "candidate_commit", "base_commit", "qualification_enabled",
    "promotion_authorized",
}
_VERDICTS = {"rejected", "regression", "inconclusive", "blocked", "promising", "qualified"}
_EVIDENCE_KEYS = {"path", "sha256", "kind", "stage", "media_type"}


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 identity of canonical JSON for *value*."""
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one JSON object, rejecting duplicate keys and non-finite values."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant: {token}")
                ),
            )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically write readable, deterministic JSON with private file mode."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        text = f"{text}\n"
        text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"value is not writable JSON: {exc}") from exc
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _document_identity(document: dict[str, Any], own_id: str, *, omit: set[str] | None = None) -> str:
    identity = dict(document)
    identity.pop(own_id, None)
    for field in omit or ():
        identity.pop(field, None)
    return canonical_hash(identity)


def submission_identity(submission: dict[str, Any]) -> dict[str, Any]:
    """Return fields that define a submission, excluding time and local path."""
    identity = dict(submission)
    identity.pop("submission_id", None)
    identity.pop("created_at", None)
    source = dict(identity["source"])
    source.pop("archive_path", None)
    identity["source"] = source
    return identity


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or any(ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _identifier(value: Any, label: str) -> str:
    _nonempty(value, label)
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be an opaque identifier, not free text")
    return value


def _schema_version(value: Any, label: str) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ValueError(f"unsupported {label} schema_version")


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-digit SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _GIT_OBJECT.fullmatch(value):
        raise ValueError(f"{label} must be a full lowercase Git object ID")
    return value


def _safe_source_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("source manifest path must be a safe POSIX relative path")
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        raise ValueError("source manifest path contains a control character")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe source manifest path: {value!r}")
    if path.as_posix() != value or (path.parts and ":" in path.parts[0]):
        raise ValueError(f"non-canonical source manifest path: {value!r}")
    return value


def _validate_link_target(path: str, target: Any) -> str:
    if not isinstance(target, str) or not target or target.startswith("/") or "\\" in target or "\x00" in target:
        raise ValueError(f"unsafe symlink target for {path!r}")
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in target):
        raise ValueError(f"symlink target contains a control character: {path!r}")
    if ":" in target.split("/", 1)[0]:
        raise ValueError(f"non-portable symlink target for {path!r}")
    components = list(PurePosixPath(path).parent.parts)
    for component in target.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            if not components:
                raise ValueError(f"symlink escapes source root: {path!r}")
            components.pop()
        else:
            components.append(component)
    if not components:
        raise ValueError(f"symlink resolves to source root: {path!r}")
    return target


def validate_source_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("source manifest must be a nonempty array")
    normalized: list[dict[str, Any]] = []
    previous = ""
    entry_paths: set[str] = set()
    symlinks: set[str] = set()
    for index, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            raise ValueError(f"source manifest entry {index} must be an object")
        kind = entry.get("type")
        expected = {"path", "type", "mode", "sha256"} if kind == "file" else {"path", "type", "mode", "target"} if kind == "symlink" else set()
        if not expected or set(entry) != expected:
            raise ValueError(f"invalid source manifest entry {index}")
        path = _safe_source_path(entry["path"])
        if path <= previous:
            raise ValueError("source manifest entries must be uniquely sorted by path")
        previous = path
        entry_paths.add(path)
        mode = entry["mode"]
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise ValueError(f"invalid mode for source path {path!r}")
        if kind == "file":
            if mode not in {0o100644, 0o100755}:
                raise ValueError(f"unsupported regular-file mode for {path!r}")
            _hash(entry["sha256"], f"source {path} sha256")
            normalized.append({"path": path, "type": kind, "mode": mode, "sha256": entry["sha256"]})
        else:
            if mode != 0o120000:
                raise ValueError(f"unsupported symlink mode for {path!r}")
            target = _validate_link_target(path, entry["target"])
            symlinks.add(path)
            normalized.append({"path": path, "type": kind, "mode": mode, "target": target})
    for path in entry_paths:
        parts = PurePosixPath(path).parts
        for n in range(1, len(parts)):
            ancestor = "/".join(parts[:n])
            if ancestor in symlinks:
                raise ValueError(f"source path {path!r} descends through symlink {ancestor!r}")
    for entry in normalized:
        if entry["type"] != "symlink":
            continue
        resolved = list(PurePosixPath(entry["path"]).parent.parts)
        for component in entry["target"].split("/"):
            if component in {"", "."}:
                continue
            if component == "..":
                if not resolved:
                    raise ValueError(f"symlink escapes source root: {entry['path']!r}")
                resolved.pop()
            else:
                resolved.append(component)
            if "/".join(resolved) in symlinks:
                raise ValueError(f"symlink target traverses another symlink: {entry['path']!r}")
    return normalized


def validate_submission(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("submission must be an object")
    if not _SUBMISSION_REQUIRED <= value.keys() or value.keys() - (_SUBMISSION_REQUIRED | _SUBMISSION_OPTIONAL):
        raise ValueError("submission has missing or unsupported fields")
    result = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    _schema_version(result["schema_version"], "submission")
    for key in ("epoch_id", "worker_id", "hypothesis_id", "idempotency_key"):
        _identifier(result[key], f"submission.{key}")
    _nonempty(result["created_at"], "submission.created_at")
    _hash(result["contract_id"], "submission.contract_id")
    _hash(result["policy_id"], "submission.policy_id")
    _hash(result["submission_id"], "submission.submission_id")
    _commit(result["base_commit"], "submission.base_commit")
    if "candidate_commit" in result and result["candidate_commit"] is not None:
        _commit(result["candidate_commit"], "submission.candidate_commit")
    source = result["source"]
    if not isinstance(source, dict) or set(source) != _SOURCE_KEYS:
        raise ValueError("submission.source has missing or unsupported fields")
    archive_path = _nonempty(source["archive_path"], "submission.source.archive_path")
    archive = Path(archive_path)
    if not archive.is_absolute() or ".." in archive.parts or str(archive) != os.path.normpath(archive_path):
        raise ValueError("submission.source.archive_path must be an absolute normalized path")
    _hash(source["archive_sha256"], "submission.source.archive_sha256")
    _hash(source["snapshot_sha256"], "submission.source.snapshot_sha256")
    _commit(source["git_commit"], "submission.source.git_commit")
    if type(source["dirty"]) is not bool:
        raise ValueError("submission.source.dirty must be boolean")
    manifest = validate_source_manifest(source["manifest"])
    source["manifest"] = manifest
    if canonical_hash(manifest) != source["snapshot_sha256"]:
        raise ValueError("submission source snapshot hash does not match its manifest")
    if result.get("candidate_commit") is not None and result["candidate_commit"] != source["git_commit"]:
        raise ValueError("candidate_commit differs from source.git_commit")
    if source["dirty"] and result.get("candidate_commit") is not None:
        raise ValueError("dirty submission cannot claim a commit-backed candidate_commit")
    if canonical_hash(submission_identity(result)) != result["submission_id"]:
        raise ValueError("submission_id does not match submission identity")
    return result


def validate_policy(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.keys() != _POLICY_KEYS:
        raise ValueError("policy has missing or unsupported fields")
    result = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    _schema_version(result["schema_version"], "policy")
    contract = result["contract"]
    if not isinstance(contract, dict) or not {"model_artifact", "runtime", "template", "tokenizer"} <= contract.keys():
        raise ValueError("policy.contract must lock model_artifact, runtime, template, and tokenizer")
    for key in ("model_artifact", "runtime", "template", "tokenizer"):
        lock = contract[key]
        if isinstance(lock, str):
            _nonempty(lock, f"policy.contract.{key}")
        elif not isinstance(lock, dict) or not lock:
            raise ValueError(f"policy.contract.{key} must be a nonempty identity or lock object")
    contract_identity = canonical_hash(contract)
    _hash(result["contract_id"], "policy.contract_id")
    if result["contract_id"] != contract_identity:
        raise ValueError("policy contract_id does not match locked contract")
    for key in ("baseline_id", "corpus_id", "toolchain_id"):
        _identifier(result[key], f"policy.{key}")
    if type(result["qualification_enabled"]) is not bool:
        raise ValueError("policy.qualification_enabled must be boolean")
    if result["qualification_enabled"]:
        _validate_qualification_contract(contract)
    stages = result["stages"]
    if not isinstance(stages, list) or [item.get("name") if isinstance(item, dict) else None for item in stages] != list(_STAGES):
        raise ValueError("policy stages must be admission, build, correctness, smoke, screen, full in order")
    for stage in stages:
        if set(stage) != {"name", "argv", "timeout_seconds", "required_evidence", "criteria"}:
            raise ValueError(f"invalid fields in policy stage {stage.get('name')!r}")
        argv = stage["argv"]
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv):
            raise ValueError(f"policy stage {stage['name']!r} argv must be a nonempty string array")
        timeout = stage["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"invalid timeout for policy stage {stage['name']!r}")
        evidence = stage["required_evidence"]
        if not isinstance(evidence, list) or any(not isinstance(name, str) or not name.strip() for name in evidence):
            raise ValueError(f"invalid required_evidence for policy stage {stage['name']!r}")
        if len(set(evidence)) != len(evidence):
            raise ValueError(f"duplicate required_evidence for policy stage {stage['name']!r}")
        if not isinstance(stage["criteria"], dict):
            raise ValueError(f"policy stage {stage['name']!r} criteria must be an object")
    expected_policy_id = _document_identity(result, "policy_id")
    _hash(result["policy_id"], "policy.policy_id")
    if result["policy_id"] != expected_policy_id:
        raise ValueError("policy_id does not match policy identity")
    return result


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _content_identity(value: Any, label: str, *, size_required: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a content-identity object")
    _hash(value.get("sha256"), f"{label}.sha256")
    if size_required or "size_bytes" in value:
        _positive_int(value.get("size_bytes"), f"{label}.size_bytes")
    return value


def _validate_qualification_contract(contract: dict[str, Any]) -> None:
    """Reject unknown/incomplete locks whenever qualification is enabled."""
    model = _content_identity(contract["model_artifact"], "policy.contract.model_artifact", size_required=True)
    _content_identity(contract["template"], "policy.contract.template")
    tokenizer = contract["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise ValueError("policy.contract.tokenizer must carry a content identity")
    if (
        "embedded_in_model_artifact_sha256" in tokenizer
        and tokenizer["embedded_in_model_artifact_sha256"] != model["sha256"]
    ):
        raise ValueError("embedded tokenizer identity must match model_artifact.sha256")
    if "sha256" in tokenizer:
        _content_identity(tokenizer, "policy.contract.tokenizer")
    elif tokenizer.get("embedded_in_model_artifact_sha256") != model["sha256"]:
        raise ValueError(
            "policy.contract.tokenizer requires sha256 or an embedded artifact hash matching model_artifact"
        )
    elif set(tokenizer) != {"embedded_in_model_artifact_sha256"}:
        raise ValueError("embedded tokenizer lock has unsupported fields")
    runtime = contract["runtime"]
    if not isinstance(runtime, dict):
        raise ValueError("policy.contract.runtime must carry locked runtime details")
    _content_identity(runtime.get("binary"), "policy.contract.runtime.binary", size_required=True)
    _commit(runtime.get("release_source_commit"), "policy.contract.runtime.release_source_commit")
    kv_dtype = _nonempty(runtime.get("kv_dtype"), "policy.contract.runtime.kv_dtype")
    if kv_dtype.lower() in {"unknown", "unspecified", "pending", "none"}:
        raise ValueError("qualification policy must explicitly lock runtime.kv_dtype")
    _positive_int(runtime.get("max_context"), "policy.contract.runtime.max_context")
    _positive_int(runtime.get("kv_capacity"), "policy.contract.runtime.kv_capacity")
    _validate_sampling_defaults(runtime.get("sampling_defaults"))


def _validate_sampling_defaults(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("policy.contract.runtime.sampling_defaults must lock logged sampling defaults")
    required_groups = {"greedy", "omitted_seed", "server_overrides", "thinking", "non_thinking"}
    if not required_groups <= value.keys():
        raise ValueError("sampling_defaults must include greedy, omitted_seed, server_overrides, thinking, non_thinking")
    if type(value["greedy"]) is not bool:
        raise ValueError("sampling_defaults.greedy must be boolean")
    _nonempty(value["omitted_seed"], "sampling_defaults.omitted_seed")
    if not isinstance(value["server_overrides"], dict):
        raise ValueError("sampling_defaults.server_overrides must be an object")
    numeric_fields = {"frequency_penalty", "min_p", "presence_penalty", "temperature", "top_p"}
    for mode in ("thinking", "non_thinking"):
        values = value[mode]
        if not isinstance(values, dict) or not numeric_fields | {"top_k"} <= values.keys():
            raise ValueError(f"sampling_defaults.{mode} is missing locked sampling fields")
        for field in numeric_fields:
            number = values[field]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError(f"sampling_defaults.{mode}.{field} must be finite numeric data")
        if type(values["top_k"]) is not int:
            raise ValueError(f"sampling_defaults.{mode}.top_k must be an integer")


def validate_receipt(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("receipt must be an object")
    if not _RECEIPT_REQUIRED <= value.keys() or value.keys() - (_RECEIPT_REQUIRED | _RECEIPT_OPTIONAL):
        raise ValueError("receipt has missing or unsupported fields")
    result = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    _schema_version(result["schema_version"], "receipt")
    for key in ("submission_id", "policy_id", "contract_id", "created_at"):
        _nonempty(result[key], f"receipt.{key}")
    _hash(result["submission_id"], "receipt.submission_id")
    _hash(result["policy_id"], "receipt.policy_id")
    _hash(result["contract_id"], "receipt.contract_id")
    _hash(result["source_snapshot_sha256"], "receipt.source_snapshot_sha256")
    if result["stage"] not in _STAGES:
        raise ValueError("invalid receipt stage")
    if not isinstance(result["verdict"], str) or result["verdict"] not in _VERDICTS:
        raise ValueError("invalid receipt verdict")
    if result["failure_class"] is not None and (
        not isinstance(result["failure_class"], str)
        or result["failure_class"] not in {"candidate", "infrastructure"}
    ):
        raise ValueError("invalid receipt failure_class")
    evidence = result["evidence"]
    if not isinstance(evidence, list):
        raise ValueError("receipt.evidence must be an array of immutable artifact references")
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or not {"path", "sha256"} <= item.keys() or item.keys() - _EVIDENCE_KEYS:
            raise ValueError(f"invalid receipt evidence reference {index}")
        _nonempty(item["path"], f"receipt.evidence[{index}].path")
        _hash(item["sha256"], f"receipt.evidence[{index}].sha256")
        for optional in ("kind", "stage", "media_type"):
            if optional in item:
                _nonempty(item[optional], f"receipt.evidence[{index}].{optional}")
    if result["verdict"] == "qualified" and (
        result["stage"] != "full" or result["failure_class"] is not None or not evidence
    ):
        raise ValueError("qualified receipt requires a full-stage result, evidence, and no failure_class")
    for key in ("candidate_commit", "base_commit"):
        if key in result:
            _commit(result[key], f"receipt.{key}")
    for key in ("qualification_enabled", "promotion_authorized"):
        if key in result and type(result[key]) is not bool:
            raise ValueError(f"receipt.{key} must be boolean")
    return result


def compare_contract(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Return deterministic dotted paths where observed differs from expected."""
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        raise ValueError("contracts must be JSON objects")
    differences: list[str] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child = f"{path}.{key}" if path else key
                if key not in left or key not in right:
                    differences.append(child)
                else:
                    compare(left[key], right[key], child)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(path)
            else:
                for index, (a, b) in enumerate(zip(left, right)):
                    compare(a, b, f"{path}[{index}]")
        elif type(left) is not type(right) or left != right:
            differences.append(path)

    compare(expected, observed, "")
    return differences
