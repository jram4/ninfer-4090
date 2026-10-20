#!/usr/bin/env python3
"""Create a private exploratory request capsule from a rejected capture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

try:
    from tools.dreamrsi.capture import SCHEMA_VERSION, _request_identity, _request_semantics
    from tools.dreamrsi.replay import ReplayError, _load_capsule
except ModuleNotFoundError:  # Support direct and flattened FNN tool-directory execution.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from tools.dreamrsi.capture import SCHEMA_VERSION, _request_identity, _request_semantics
        from tools.dreamrsi.replay import ReplayError, _load_capsule
    except ModuleNotFoundError:
        from capture import SCHEMA_VERSION, _request_identity, _request_semantics  # type: ignore[no-redef]
        from replay import ReplayError, _load_capsule  # type: ignore[no-redef]


TRANSFORM_PATH = "$.thinking.display"
TRANSFORM_FROM = "omitted"
TRANSFORM_TO = "summarized"
EFFORT_TRANSFORM_PATH = "$.output_config.effort"
EFFORT_TRANSFORM_FROM = "high"
EFFORT_TRANSFORM_TO = "xhigh"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_CAPSULE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class DeriveError(RuntimeError):
    """Invalid source capsule or unsafe derived-capsule destination."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeriveError("request JSON contains duplicate object keys")
        result[key] = value
    return result


def _load_json_object(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeriveError("request body is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DeriveError("request body must be a JSON object")
    return value


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _object_member_spans(text: str, object_start: int) -> list[tuple[str, int, int]]:
    """Return direct member value spans without reserializing the surrounding JSON."""
    if object_start >= len(text) or text[object_start] != "{":
        raise DeriveError("expected a JSON object while locating the transform field")
    decoder = json.JSONDecoder()
    result: list[tuple[str, int, int]] = []
    index = _skip_space(text, object_start + 1)
    if index < len(text) and text[index] == "}":
        return result
    while index < len(text):
        try:
            key, key_end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise DeriveError("request JSON could not be structurally inspected") from exc
        if not isinstance(key, str):
            raise DeriveError("request JSON object has a non-string key")
        colon = _skip_space(text, key_end)
        if colon >= len(text) or text[colon] != ":":
            raise DeriveError("request JSON object is malformed")
        value_start = _skip_space(text, colon + 1)
        try:
            _value, value_end = decoder.raw_decode(text, value_start)
        except json.JSONDecodeError as exc:
            raise DeriveError("request JSON could not be structurally inspected") from exc
        result.append((key, value_start, value_end))
        index = _skip_space(text, value_end)
        if index < len(text) and text[index] == ",":
            index = _skip_space(text, index + 1)
            continue
        if index < len(text) and text[index] == "}":
            return result
        raise DeriveError("request JSON object is malformed")
    raise DeriveError("request JSON object is incomplete")


def _unique_member_span(text: str, object_start: int, key: str) -> tuple[int, int]:
    matches = [(start, end) for name, start, end in _object_member_spans(text, object_start) if name == key]
    if len(matches) != 1:
        raise DeriveError(f"request must contain exactly one {key!r} field at the expected path")
    return matches[0]


def _assert_only_expected_transform(
    source_body: bytes,
    derived_body: bytes,
    *,
    effort_override: str | None = None,
) -> None:
    source = _load_json_object(source_body)
    derived = _load_json_object(derived_body)
    expected = copy.deepcopy(source)
    thinking = expected.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("display") != TRANSFORM_FROM:
        raise DeriveError("source thinking.display is not the expected omitted value")
    thinking["display"] = TRANSFORM_TO
    if effort_override is not None:
        output_config = expected.get("output_config")
        if effort_override != EFFORT_TRANSFORM_TO or not isinstance(output_config, dict):
            raise DeriveError("effort override must map output_config.effort high to xhigh")
        if output_config.get("effort") != EFFORT_TRANSFORM_FROM:
            raise DeriveError("source output_config.effort is not high")
        output_config["effort"] = effort_override
    if derived != expected:
        raise DeriveError("derived request changes fields outside the declared transform paths")


def _path_value_span(text: str, path: tuple[str, ...]) -> tuple[int, int]:
    root_start = _skip_space(text, 0)
    if root_start >= len(text) or text[root_start] != "{":
        raise DeriveError("request body must start with a JSON object")
    object_start = root_start
    for key in path[:-1]:
        value_start, _value_end = _unique_member_span(text, object_start, key)
        if text[value_start] != "{":
            raise DeriveError(f"source field before {path[-1]!r} must be an object")
        object_start = value_start
    return _unique_member_span(text, object_start, path[-1])


def _derive_request_body(
    source_body: bytes,
    *,
    effort_override: str | None = None,
) -> tuple[bytes, list[dict[str, Any]]]:
    source = _load_json_object(source_body)
    thinking = source.get("thinking")
    if not isinstance(thinking, dict):
        raise DeriveError("source request has no thinking object")
    if thinking.get("type") != "adaptive":
        raise DeriveError("source request must use adaptive thinking")
    if thinking.get("display") != TRANSFORM_FROM:
        raise DeriveError("source request thinking.display must be omitted")

    text = source_body.decode("utf-8")
    replacements: list[tuple[int, int, str, str, str]] = []
    display_start, display_end = _path_value_span(text, ("thinking", "display"))
    try:
        raw_display = json.loads(text[display_start:display_end])
    except json.JSONDecodeError as exc:
        raise DeriveError("source thinking.display value is malformed") from exc
    if raw_display != TRANSFORM_FROM:
        raise DeriveError("source request thinking.display must be omitted")
    replacements.append((display_start, display_end, TRANSFORM_FROM, TRANSFORM_TO, TRANSFORM_PATH))

    if effort_override is not None:
        if effort_override != EFFORT_TRANSFORM_TO:
            raise DeriveError("only the high-to-xhigh exploratory effort mapping is supported")
        output_config = source.get("output_config")
        if not isinstance(output_config, dict) or output_config.get("effort") != EFFORT_TRANSFORM_FROM:
            raise DeriveError("source output_config.effort must be high for the xhigh variant")
        effort_start, effort_end = _path_value_span(text, ("output_config", "effort"))
        try:
            raw_effort = json.loads(text[effort_start:effort_end])
        except json.JSONDecodeError as exc:
            raise DeriveError("source output_config.effort value is malformed") from exc
        if raw_effort != EFFORT_TRANSFORM_FROM:
            raise DeriveError("source output_config.effort must be high for the xhigh variant")
        replacements.append(
            (effort_start, effort_end, EFFORT_TRANSFORM_FROM, effort_override, EFFORT_TRANSFORM_PATH)
        )

    derived_text = text
    for start, end, _old, new, _path in sorted(replacements, key=lambda item: item[0], reverse=True):
        derived_text = derived_text[:start] + json.dumps(new, ensure_ascii=False) + derived_text[end:]
    derived_body = derived_text.encode("utf-8")
    _assert_only_expected_transform(source_body, derived_body, effort_override=effort_override)
    diff = [{"path": path, "from": old, "to": new} for _start, _end, old, new, path in replacements]
    return derived_body, diff


def _require_owner_only(path: Path, *, directory: bool | None = None) -> None:
    try:
        stat_result = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DeriveError("private capsule path could not be inspected") from exc
    if stat_result.st_uid != os.geteuid() or stat_result.st_mode & 0o077:
        raise DeriveError("capsule paths must be owned by the current user and owner-only")
    if directory is True and not path.is_dir():
        raise DeriveError("capsule path is not a directory")
    if directory is False and not path.is_file():
        raise DeriveError("capsule entry is not a regular file")


def _verify_private_source(capsule: Path) -> None:
    _require_owner_only(capsule, directory=True)
    for entry in capsule.iterdir():
        _require_owner_only(entry, directory=False)


def _write_owner_only(path: Path, data: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(fd)
        os.fchmod(fd, 0o400)
    finally:
        os.close(fd)


def _append_descriptor(path: Path, descriptor: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_owner_only(path.parent, directory=True)
    encoded = (json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "ab", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def derive_capsule(
    source_capsule: str | Path,
    *,
    output_dir: str | Path | None = None,
    descriptor_jsonl: str | Path | None = None,
    effort_override: str | None = None,
) -> dict[str, Any]:
    """Derive one sanitized-display request capsule without modifying its source."""
    source_path = Path(source_capsule).expanduser().resolve(strict=True)
    _verify_private_source(source_path)
    try:
        capsule, parent_manifest, parent_body = _load_capsule(source_path)
    except ReplayError as exc:
        raise DeriveError(str(exc)) from exc

    parent_request = parent_manifest.get("request")
    parent_response = parent_manifest.get("response")
    if not isinstance(parent_request, dict) or not isinstance(parent_response, dict):
        raise DeriveError("source capsule request or response metadata is missing")
    parent_capsule_id = parent_manifest.get("capsule_id")
    if not isinstance(parent_capsule_id, str) or not _CAPSULE_ID_RE.fullmatch(parent_capsule_id):
        raise DeriveError("source capsule ID is invalid")
    if parent_response.get("status") != 400:
        raise DeriveError("source capsule must contain a rejected HTTP 400 request")
    if parent_request.get("method") != "POST":
        raise DeriveError("source route must use POST")
    target = parent_request.get("target")
    if not isinstance(target, str) or urlsplit(target).path != "/v1/messages":
        raise DeriveError("source route must be /v1/messages")

    source_object = _load_json_object(parent_body)
    derived_body, diff = _derive_request_body(parent_body, effort_override=effort_override)
    semantics = source_object
    if semantics.get("model") != "claude-sonnet-5":
        raise DeriveError("source model does not match the bounded Sonnet capsule")
    if semantics.get("max_tokens") != 96:
        raise DeriveError("source output budget does not match the bounded 96-token capsule")

    parent_request_hash = _sha256(parent_body)
    derived_request_hash = _sha256(derived_body)
    if parent_request.get("body_sha256") != parent_request_hash:
        raise DeriveError("source request body hash does not match its manifest")
    h04a_body_hash: str | None = None
    h04a_capsule_id: str | None = None
    if effort_override is not None:
        h04a_body, _h04a_diff = _derive_request_body(parent_body)
        h04a_body_hash = _sha256(h04a_body)
        h04a_capsule_id = f"derived-{parent_capsule_id}-{h04a_body_hash[:12]}"

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source_path.parent.resolve()
    )
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_owner_only(destination, directory=True)
    capsule_id = f"derived-{parent_capsule_id}-{derived_request_hash[:12]}"
    final_path = destination / f"{capsule_id}.capsule"
    if final_path.exists():
        raise DeriveError("the deterministic derived capsule already exists")
    partial_path = destination / f".{capsule_id}.partial"
    try:
        partial_path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise DeriveError("a partial derived capsule already exists") from exc

    request_headers = parent_request.get("headers")
    if not isinstance(request_headers, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("name"), str)
        or not isinstance(item.get("value"), str)
        for item in request_headers
    ):
        shutil.rmtree(partial_path)
        raise DeriveError("source request headers are invalid")
    request_headers = copy.deepcopy(request_headers)
    derived_semantics = _request_semantics(derived_body, request_headers)
    output_budget = source_object.get("max_tokens")
    output_budget = output_budget if isinstance(output_budget, int) and not isinstance(output_budget, bool) else None
    request_identity = copy.deepcopy(parent_request.get("identity", {}))
    if not isinstance(request_identity, dict):
        request_identity = {}
    request_identity["capsule_id"] = capsule_id
    response = {
        "status": None,
        "reason": None,
        "headers": [],
        "content_type": None,
        "content_length": None,
        "transfer_encoding": None,
        "body_file": "response.body",
        "body_bytes": 0,
        "body_sha256": EMPTY_SHA256,
        "chunks": [],
        "stream": {
            "requested": derived_semantics["stream_requested"],
            "observed": False,
            "chunk_count": 0,
            "event_types": [],
            "terminal_event": None,
            "termination": "derived_not_replayed",
        },
        "identity": {},
    }
    derivation = {
        "parent_capsule_id": parent_capsule_id,
        "parent_raw_location": str(capsule),
        "parent_raw_request_sha256": parent_request_hash,
        "transformed_body_sha256": derived_request_hash,
        "diff": diff,
        "semantic_transform": True,
        "production_compatible": False,
    }
    if h04a_body_hash is not None and h04a_capsule_id is not None:
        derivation["h04a_capsule_id"] = h04a_capsule_id
        derivation["h04a_request_sha256"] = h04a_body_hash
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": capsule_id,
        "artifact_kind": "derived_request",
        "upstream_origin": parent_manifest.get("upstream_origin"),
        "cache_state": parent_manifest.get("cache_state", "unknown"),
        "request": {
            "method": parent_request["method"],
            "target": target,
            "headers": request_headers,
            "body_file": "request.body",
            "body_bytes": len(derived_body),
            "body_sha256": derived_request_hash,
            "semantics": derived_semantics,
            "identity": request_identity,
        },
        "response": response,
        "derivation": derivation,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    try:
        _write_owner_only(partial_path / "request.body", derived_body)
        _write_owner_only(partial_path / "response.body", b"")
        _write_owner_only(partial_path / "manifest.json", manifest_bytes)
        os.chmod(partial_path, 0o500)
        partial_path.rename(final_path)
    except Exception:
        try:
            os.chmod(partial_path, 0o700)
            shutil.rmtree(partial_path)
        except OSError:
            pass
        raise

    labels = {
        f"route={parent_request['method']} {target.split('?', 1)[0]}",
        f"stream={str(derived_semantics['stream_requested']).lower()}",
        f"cache_state={manifest['cache_state']}",
        "response_status=not_replayed",
        "semantic_transform=true",
        "production_compatible=false",
    }
    if derived_semantics.get("model"):
        labels.add(f"model={derived_semantics['model']}")
    if derived_semantics.get("cli_version"):
        labels.add(f"cli_version={derived_semantics['cli_version']}")
    descriptor_metadata = {
        "method": parent_request["method"],
        "request_target": target.split("?", 1)[0],
        "model": derived_semantics.get("model"),
        "cli_version": derived_semantics.get("cli_version"),
        "stream_requested": derived_semantics["stream_requested"],
        "cache_state": manifest["cache_state"],
        "prompt_cache_control_present": derived_semantics["prompt_cache_control_present"],
        "response_status": None,
        "response_sha256": EMPTY_SHA256,
        "parent_capsule_id": parent_capsule_id,
        "parent_raw_request_sha256": parent_request_hash,
        "transformed_body_sha256": derived_request_hash,
        "diff": diff,
        "semantic_transform": True,
        "production_compatible": False,
    }
    if h04a_body_hash is not None and h04a_capsule_id is not None:
        descriptor_metadata["h04a_capsule_id"] = h04a_capsule_id
        descriptor_metadata["h04a_request_sha256"] = h04a_body_hash
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": capsule_id,
        "capsule_path": str(final_path),
        "bucket_labels": sorted(labels),
        "request_sha256": derived_request_hash,
        "metadata": descriptor_metadata,
    }
    if output_budget is not None:
        descriptor["output_budget"] = output_budget
    descriptor_path = (
        Path(descriptor_jsonl).expanduser().resolve()
        if descriptor_jsonl is not None
        else destination / "descriptors.jsonl"
    )
    _append_descriptor(descriptor_path, descriptor)
    return descriptor


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_capsule", type=Path, help="Owner-only rejected capsule directory")
    parser.add_argument("--output-dir", type=Path, help="Owner-only derived capsule directory; defaults to the source directory")
    parser.add_argument("--descriptor-jsonl", type=Path, help="Sanitized descriptor JSONL; defaults to OUTPUT_DIR/descriptors.jsonl")
    parser.add_argument(
        "--effort-override",
        choices=(EFFORT_TRANSFORM_TO,),
        help="Also map a source output_config.effort of high to xhigh for a separate exploratory variant",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        descriptor = derive_capsule(
            args.source_capsule,
            output_dir=args.output_dir,
            descriptor_jsonl=args.descriptor_jsonl,
            effort_override=args.effort_override,
        )
    except (OSError, ValueError, ReplayError, DeriveError) as exc:
        parser.error(str(exc))
    print(json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
