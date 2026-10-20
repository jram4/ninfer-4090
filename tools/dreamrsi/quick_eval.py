#!/usr/bin/env python3
"""Deterministic sparse capsule sampling and paired Serve telemetry reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any


PLAN_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
SERVE_SCHEMA_VERSION = 21
SAMPLER_NAME = "observed_bucket_stratified"
MIN_PAIRS_FOR_CLASSIFICATION = 3
BOOTSTRAP_RESAMPLES = 4000
FINISH_REASONS = {"output_limit", "stop_token", "stop_string"}
TERMINAL_EVENTS = {"request_done", "request_error", "request_rejected"}

JsonObject = dict[str, Any]


class QuickEvalError(ValueError):
    """Raised when an input violates the quick-eval contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuickEvalError(f"{name} must be a nonempty string")
    return value


def _sha256_string(value: Any, name: str) -> str:
    digest = _required_string(value, name)
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise QuickEvalError(f"{name} must be a 64-digit hexadecimal SHA-256")
    return digest.lower()


def _transform_diff(value: Any, name: str) -> JsonObject | list[JsonObject]:
    def validate_entry(entry: Any, entry_name: str) -> JsonObject:
        if not isinstance(entry, dict) or set(entry) != {"path", "from", "to"}:
            raise QuickEvalError(f"{entry_name} must contain exactly path, from, and to")
        return {
            key: _required_string(entry.get(key), f"{entry_name}.{key}")
            for key in ("path", "from", "to")
        }

    if isinstance(value, dict):
        return validate_entry(value, name)
    if not isinstance(value, list) or len(value) != 2:
        raise QuickEvalError(f"{name} must be one diff object or an ordered two-entry array")
    entries = [validate_entry(entry, f"{name}[{index}]") for index, entry in enumerate(value)]
    if entries[0]["path"] == entries[1]["path"]:
        raise QuickEvalError(f"{name} must not repeat a transformed path")
    return entries


def _descriptor_metadata_value(
    row: JsonObject,
    metadata: JsonObject,
    name: str,
    aliases: tuple[str, ...] = (),
) -> Any:
    values = [
        source[key]
        for source in (row, metadata)
        for key in (name, *aliases)
        if key in source
    ]
    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        raise QuickEvalError(f"descriptor has conflicting {name} provenance values")
    return first


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QuickEvalError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QuickEvalError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuickEvalError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise QuickEvalError(f"{name} must be a finite number")
    return result


def _load_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QuickEvalError(
                        f"{path}:{line_number}: invalid JSON on line"
                    ) from exc
                if not isinstance(value, dict):
                    raise QuickEvalError(f"{path}:{line_number}: each line must be an object")
                rows.append(value)
    except OSError as exc:
        raise QuickEvalError(f"cannot read {path}: {exc.strerror or exc}") from exc
    return rows


def _load_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QuickEvalError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise QuickEvalError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise QuickEvalError(f"{path}: expected a JSON object")
    return value


def _validate_descriptor(row: JsonObject, line_number: int) -> JsonObject:
    prefix = f"descriptor line {line_number}"
    if row.get("schema_version") != 1:
        raise QuickEvalError(f"{prefix}: schema_version must be 1")
    capsule_id = _required_string(row.get("capsule_id"), f"{prefix} capsule_id")
    capsule_path = _required_string(row.get("capsule_path"), f"{prefix} capsule_path")
    labels_value = row.get("bucket_labels")
    if not isinstance(labels_value, list) or not labels_value:
        raise QuickEvalError(f"{prefix}: bucket_labels must be a nonempty array")
    labels = sorted({_required_string(value, f"{prefix} bucket label") for value in labels_value})
    if len(labels) != len(labels_value):
        raise QuickEvalError(f"{prefix}: bucket_labels must not contain duplicates")
    request_sha256 = _sha256_string(row.get("request_sha256"), f"{prefix} request_sha256")
    capture_budget = row.get("output_budget")
    if capture_budget is not None:
        capture_budget = _positive_int(capture_budget, f"{prefix} output_budget")
    metadata_value = row.get("metadata")
    if metadata_value is not None and not isinstance(metadata_value, dict):
        raise QuickEvalError(f"{prefix} metadata must be an object when present")
    metadata: JsonObject = metadata_value if isinstance(metadata_value, dict) else {}
    semantic_value = _descriptor_metadata_value(row, metadata, "semantic_transform")
    semantic_transform = False if semantic_value is None else semantic_value
    production_value = _descriptor_metadata_value(row, metadata, "production_compatible")
    production_compatible = not semantic_transform if production_value is None else production_value
    if not isinstance(semantic_transform, bool):
        raise QuickEvalError(f"{prefix} semantic_transform must be a boolean")
    if not isinstance(production_compatible, bool):
        raise QuickEvalError(f"{prefix} production_compatible must be a boolean")
    if semantic_transform and production_compatible:
        raise QuickEvalError(
            f"{prefix}: a semantic transform cannot be production-compatible"
        )
    parent_request_sha256 = _descriptor_metadata_value(
        row, metadata, "parent_request_sha256", ("parent_raw_request_sha256",)
    )
    derived_request_sha256 = _descriptor_metadata_value(
        row, metadata, "derived_request_sha256", ("transformed_body_sha256",)
    )
    diff_value = _descriptor_metadata_value(row, metadata, "transform_diff", ("diff",))
    transform_diff: JsonObject | list[JsonObject] | None = None
    if semantic_transform:
        parent_request_sha256 = _sha256_string(
            parent_request_sha256, f"{prefix} parent_request_sha256"
        )
        derived_request_sha256 = _sha256_string(
            derived_request_sha256, f"{prefix} derived_request_sha256"
        )
        if derived_request_sha256 != request_sha256:
            raise QuickEvalError(
                f"{prefix}: derived_request_sha256 differs from request_sha256"
            )
        transform_diff = _transform_diff(diff_value, f"{prefix} transform_diff")
    elif any(value is not None for value in (parent_request_sha256, derived_request_sha256, diff_value)):
        raise QuickEvalError(f"{prefix}: transform provenance requires semantic_transform=true")
    # Keep the plan free of request bodies and arbitrary metadata. The capsule itself stays external.
    result: JsonObject = {
        "capsule_id": capsule_id,
        "capsule_path": capsule_path,
        "bucket_labels": labels,
        "request_sha256": request_sha256.lower(),
        "semantic_transform": semantic_transform,
        "production_compatible": production_compatible,
    }
    if capture_budget is not None:
        result["capture_output_budget"] = capture_budget
    if semantic_transform:
        result.update(
            {
                "parent_request_sha256": parent_request_sha256,
                "derived_request_sha256": derived_request_sha256,
                "transform_diff": transform_diff,
            }
        )
    return result


def make_plan(
    descriptors: list[JsonObject],
    *,
    seed: int,
    per_bucket: int,
    cache_recipe_id: str,
    output_budget: int,
    bucket_labels: list[str] | None = None,
) -> JsonObject:
    """Select one deterministic, unique capsule subset from observed labels only."""
    per_bucket = _positive_int(per_bucket, "per_bucket")
    output_budget = _positive_int(output_budget, "output_budget")
    cache_recipe_id = _required_string(cache_recipe_id, "cache_recipe_id")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise QuickEvalError("seed must be an integer")

    validated: list[JsonObject] = []
    seen_ids: set[str] = set()
    for index, descriptor in enumerate(descriptors, 1):
        clean = _validate_descriptor(descriptor, index)
        capsule_id = clean["capsule_id"]
        if capsule_id in seen_ids:
            raise QuickEvalError(f"duplicate capsule_id: {capsule_id}")
        seen_ids.add(capsule_id)
        validated.append(clean)
    if not validated:
        raise QuickEvalError("capsule manifest is empty")

    # A replay should preserve the capsule's captured output budget. Rows with a known, different
    # budget are ineligible for this paired plan; rows whose budget could not be parsed remain
    # selectable, and the later Serve request check still enforces the planned budget.
    validated = [
        row
        for row in validated
        if row.get("capture_output_budget") is None or row["capture_output_budget"] == output_budget
    ]
    if not validated:
        raise QuickEvalError(f"no capsules have the requested output_budget {output_budget}")

    observed = sorted({label for row in validated for label in row["bucket_labels"]})
    selected_labels = sorted(set(bucket_labels if bucket_labels else observed))
    if not selected_labels:
        raise QuickEvalError("no bucket labels were selected")
    unknown = sorted(set(selected_labels) - set(observed))
    if unknown:
        raise QuickEvalError(f"bucket label(s) were not observed: {', '.join(unknown)}")

    def rank(label: str, capsule_id: str) -> str:
        return hashlib.sha256(f"{seed}\0{label}\0{capsule_id}".encode("utf-8")).hexdigest()

    counts = {label: 0 for label in selected_labels}
    selected: dict[str, JsonObject] = {}
    chosen_for: dict[str, str] = {}
    by_id = {row["capsule_id"]: row for row in validated}

    while True:
        underfilled = {label for label, count in counts.items() if count < per_bucket}
        if not underfilled:
            break
        choices: list[tuple[int, str, str, str]] = []
        for label in sorted(underfilled):
            candidates = [
                row
                for row in validated
                if label in row["bucket_labels"] and row["capsule_id"] not in selected
            ]
            for row in candidates:
                coverage = sum(
                    1
                    for candidate_label in row["bucket_labels"]
                    if candidate_label in underfilled
                )
                choices.append(
                    (-coverage, rank(label, row["capsule_id"]), label, row["capsule_id"])
                )
        if not choices:
            break
        _, _, stratum, capsule_id = min(choices)
        row = by_id[capsule_id]
        selected[capsule_id] = row
        chosen_for[capsule_id] = stratum
        # A selected capsule represents each of its still-underfilled observed labels once.
        for label in row["bucket_labels"]:
            if label in counts and counts[label] < per_bucket:
                counts[label] += 1

    ordered_ids = list(selected)
    # Selection order is deterministic; it also supplies a stable serial replay order.
    selected_rows = [
        {
            **selected[capsule_id],
            "stratified_for": chosen_for[capsule_id],
            "output_budget": output_budget,
        }
        for capsule_id in ordered_ids
    ]
    semantic_transform = any(row["semantic_transform"] for row in selected_rows)
    production_compatible = all(row["production_compatible"] for row in selected_rows)
    core: JsonObject = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "seed": seed,
        "sampler": {
            "name": SAMPLER_NAME,
            "version": 1,
            "per_bucket": per_bucket,
            "bucket_labels": selected_labels,
        },
        "cache_recipe_id": cache_recipe_id,
        "output_budget": output_budget,
        "semantic_transform": semantic_transform,
        "production_compatible": production_compatible and not semantic_transform,
        "bucket_counts": counts,
        "unfilled_buckets": sorted(label for label, count in counts.items() if count < per_bucket),
        "selected": selected_rows,
    }
    return {**core, "plan_id": _sha256(core)}


def _validate_plan(plan: JsonObject) -> list[JsonObject]:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise QuickEvalError("plan schema_version must be 1")
    selected = plan.get("selected")
    if not isinstance(selected, list) or not selected:
        raise QuickEvalError("plan selected must be a nonempty array")
    if _positive_int(plan.get("output_budget"), "plan output_budget") <= 0:
        raise QuickEvalError("plan output_budget must be positive")
    _required_string(plan.get("cache_recipe_id"), "plan cache_recipe_id")
    sampler = plan.get("sampler")
    if not isinstance(sampler, dict) or sampler.get("name") != SAMPLER_NAME:
        raise QuickEvalError("plan sampler is not supported")
    core = {key: value for key, value in plan.items() if key != "plan_id"}
    expected = _sha256(core)
    if plan.get("plan_id") != expected:
        raise QuickEvalError("plan_id does not match the plan contents")
    semantic_transform = plan.get("semantic_transform", False)
    production_compatible = plan.get("production_compatible", not semantic_transform)
    if not isinstance(semantic_transform, bool):
        raise QuickEvalError("plan semantic_transform must be a boolean")
    if not isinstance(production_compatible, bool):
        raise QuickEvalError("plan production_compatible must be a boolean")
    if semantic_transform and production_compatible:
        raise QuickEvalError("a semantically transformed plan cannot be production-compatible")
    seen: set[str] = set()
    selected_semantic_transform = False
    selected_production_compatible = True
    for row in selected:
        if not isinstance(row, dict):
            raise QuickEvalError("plan selected entries must be objects")
        capsule_id = _required_string(row.get("capsule_id"), "plan capsule_id")
        if capsule_id in seen:
            raise QuickEvalError(f"plan contains duplicate capsule_id: {capsule_id}")
        seen.add(capsule_id)
        if row.get("output_budget") != plan["output_budget"]:
            raise QuickEvalError(f"plan output budget mismatch for capsule {capsule_id}")
        row_transform = row.get("semantic_transform", False)
        row_production = row.get("production_compatible", not row_transform)
        if not isinstance(row_transform, bool) or not isinstance(row_production, bool):
            raise QuickEvalError(f"plan semantic compatibility flags must be booleans for {capsule_id}")
        if row_transform and row_production:
            raise QuickEvalError(f"semantically transformed capsule {capsule_id} cannot be production-compatible")
        parent_request_sha256 = row.get("parent_request_sha256")
        derived_request_sha256 = row.get("derived_request_sha256")
        transform_diff = row.get("transform_diff")
        if row_transform:
            _sha256_string(parent_request_sha256, f"plan {capsule_id} parent_request_sha256")
            derived_hash = _sha256_string(
                derived_request_sha256, f"plan {capsule_id} derived_request_sha256"
            )
            request_hash = _sha256_string(row.get("request_sha256"), f"plan {capsule_id} request_sha256")
            if derived_hash != request_hash:
                raise QuickEvalError(
                    f"plan derived_request_sha256 differs from request_sha256 for {capsule_id}"
                )
            _transform_diff(transform_diff, f"plan {capsule_id} transform_diff")
        elif any(
            value is not None
            for value in (parent_request_sha256, derived_request_sha256, transform_diff)
        ):
            raise QuickEvalError(
                f"plan transform provenance requires semantic_transform=true for {capsule_id}"
            )
        selected_semantic_transform |= row_transform
        selected_production_compatible &= row_production
    if "semantic_transform" in plan and semantic_transform != selected_semantic_transform:
        raise QuickEvalError("plan semantic_transform does not match its selected capsules")
    if "production_compatible" in plan and production_compatible != (
        selected_production_compatible and not selected_semantic_transform
    ):
        raise QuickEvalError("plan production_compatible does not match its selected capsules")
    return selected


def _object(value: Any, name: str, *, required: bool = False) -> JsonObject | None:
    if value is None and not required:
        return None
    if not isinstance(value, dict):
        raise QuickEvalError(f"{name} must be an object")
    return value


def _event_identity(event: JsonObject) -> tuple[str | None, Any]:
    server_id = event.get("server_instance_id")
    if server_id is not None and (not isinstance(server_id, str) or not server_id):
        raise QuickEvalError("Serve event server_instance_id must be a nonempty string")
    request = event.get("request")
    request_id = request.get("request_id") if isinstance(request, dict) else None
    return server_id, request_id


def _validate_row_events(
    row: JsonObject, plan: JsonObject, capsule_id: str
) -> tuple[JsonObject | None, JsonObject | None]:
    start = _object(row.get("request_start"), f"{capsule_id} request_start")
    terminal = _object(row.get("terminal"), f"{capsule_id} terminal")
    if start is not None:
        if start.get("schema_version") != SERVE_SCHEMA_VERSION or start.get("event") != "request_start":
            raise QuickEvalError(f"{capsule_id}: request_start must be a schema-v21 request_start event")
        start_server, start_id = _event_identity(start)
        if start_server is None or start_id is None:
            raise QuickEvalError(f"{capsule_id}: request_start lacks Serve request identity")
        request = start.get("request")
        if not isinstance(request, dict):
            raise QuickEvalError(f"{capsule_id}: request_start has no request object")
        if request.get("requested_output_tokens") != plan["output_budget"]:
            raise QuickEvalError(f"{capsule_id}: request output budget differs from the plan")
    if terminal is None:
        # The request may have failed before Serve wrote a terminal event. Keep it in the report as
        # censored work instead of turning the whole paired reduction into a parse failure.
        return start, None
    if terminal.get("schema_version") != SERVE_SCHEMA_VERSION:
        raise QuickEvalError(f"{capsule_id}: terminal Serve event must use schema-v21")
    if terminal.get("event") not in TERMINAL_EVENTS:
        raise QuickEvalError(f"{capsule_id}: terminal event must be request_done/error/rejected")
    if start is not None:
        terminal_server, terminal_id = _event_identity(terminal)
        start_server, start_id = _event_identity(start)
        if terminal_server is None or terminal_id is None:
            raise QuickEvalError(f"{capsule_id}: terminal event lacks Serve request identity")
        if start_server != terminal_server or start_id != terminal_id:
            raise QuickEvalError(f"{capsule_id}: request_start and terminal request identities differ")
    elif terminal.get("event") != "request_rejected":
        raise QuickEvalError(f"{capsule_id}: only request_rejected may lack request_start")
    elif terminal.get("request") is not None and not isinstance(terminal.get("request"), dict):
        raise QuickEvalError(f"{capsule_id}: rejected event request must be an object when present")
    return start, terminal


def _tool_parse(result: JsonObject) -> JsonObject | None:
    value = result.get("tool_call_parse")
    if not isinstance(value, dict):
        return None
    allowed = (
        "marker_seen",
        "structured_call_count",
        "empty_arguments_omitted",
        "schema_mismatch_arguments",
        "fallback_reason",
    )
    return {key: value[key] for key in allowed if key in value}


def _sanitize_client(client_value: Any) -> JsonObject:
    if client_value is None:
        client: JsonObject = {}
    elif isinstance(client_value, dict):
        client = client_value
    else:
        raise QuickEvalError("client must be an object")
    complete = client.get("complete") is True
    received = _nonnegative_int(client.get("received_completion_tokens"), "client received_completion_tokens")
    ttft = _finite_number(client.get("ttft_seconds"), "client ttft_seconds")
    wall = _finite_number(client.get("wall_seconds"), "client wall_seconds")
    if ttft is not None and ttft < 0 or wall is not None and wall < 0:
        raise QuickEvalError("client times must be nonnegative")
    finish_reason = client.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise QuickEvalError("client finish_reason must be a string when present")
    return {
        "complete": complete,
        "finish_reason": finish_reason,
        "received_completion_tokens": received,
        "ttft_seconds": ttft,
        "wall_seconds": wall,
        "error_present": client.get("error") not in (None, "", False),
    }


def _arm_capsule(row: JsonObject | None, plan: JsonObject, selected: JsonObject) -> JsonObject:
    capsule_id = selected["capsule_id"]
    semantic_transform = selected.get(
        "semantic_transform", plan.get("semantic_transform", False)
    )
    production_compatible = selected.get(
        "production_compatible", plan.get("production_compatible", not semantic_transform)
    )
    if row is None:
        return {
            "capsule_id": capsule_id,
            "semantic_transform": semantic_transform,
            "production_compatible": production_compatible,
            "parent_request_sha256": selected.get("parent_request_sha256"),
            "derived_request_sha256": selected.get("derived_request_sha256"),
            "transform_diff": selected.get("transform_diff"),
            "status": "missing",
            "censored_reasons": ["missing_result_row"],
            "completed": False,
            "metrics_available": False,
        }
    if row.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise QuickEvalError(f"{capsule_id}: result schema_version must be 1")
    if row.get("plan_id") != plan["plan_id"]:
        raise QuickEvalError(f"{capsule_id}: result plan_id differs from the paired plan")
    if row.get("capsule_id") != capsule_id:
        raise QuickEvalError(f"result row order/ID mismatch for capsule {capsule_id}")
    for key, expected in (
        ("semantic_transform", semantic_transform),
        ("production_compatible", production_compatible),
    ):
        if key in row:
            if not isinstance(row[key], bool):
                raise QuickEvalError(f"{capsule_id}: result {key} must be a boolean")
            if row[key] != expected:
                raise QuickEvalError(f"{capsule_id}: result {key} differs from the plan")

    start, terminal = _validate_row_events(row, plan, capsule_id)
    client = _sanitize_client(row.get("client"))
    event_name = terminal.get("event") if terminal is not None else "missing_terminal_event"
    serve_id, request_id = _event_identity(terminal) if terminal is not None else (None, None)
    if terminal is None and start is not None:
        serve_id, request_id = _event_identity(start)
    metrics: JsonObject = {
        "capsule_id": capsule_id,
        "server_instance_id": serve_id,
        "request_id": request_id,
        "semantic_transform": semantic_transform,
        "production_compatible": production_compatible,
        "parent_request_sha256": selected.get("parent_request_sha256"),
        "derived_request_sha256": selected.get("derived_request_sha256"),
        "transform_diff": selected.get("transform_diff"),
        "terminal_event": event_name,
        "finish_reason": None,
        "client_finish_reason": client["finish_reason"],
        "generated_tokens": None,
        "committed_tokens": None,
        "decode_committed_tokens": None,
        "rounds": None,
        "drafted_tokens": None,
        "accepted_tokens": None,
        "tokens_per_round": None,
        "server_ttft_seconds": None,
        "client_ttft_seconds": client["ttft_seconds"],
        "ttft_seconds": client["ttft_seconds"],
        "ttft_source": "client" if client["ttft_seconds"] is not None else None,
        "decode_seconds": None,
        "decode_tok_s": None,
        "wall_seconds": client["wall_seconds"],
        "server_total_seconds": None,
        "tool_call_parse": None,
        "tool_call_count": None,
        "vram_peak_bytes": _nonnegative_int(row.get("vram_peak_bytes"), "vram_peak_bytes"),
        "cache_recipe_id": row.get("cache_recipe_id") if isinstance(row.get("cache_recipe_id"), str) else None,
        "cache_recipe_matches_plan": row.get("cache_recipe_id") == plan.get("cache_recipe_id"),
    }
    if start is not None:
        start_request = start.get("request", {})
        sampling = start_request.get("sampling")
        if not isinstance(sampling, dict):
            raise QuickEvalError(f"{capsule_id}: request_start has no sampling identity")
        metrics["request_config"] = {
            "protocol": start_request.get("protocol"),
            "model": start_request.get("model"),
            "stream": start_request.get("stream"),
            "requested_output_tokens": start_request.get("requested_output_tokens"),
            "sampling": sampling,
            "enable_thinking": start_request.get("enable_thinking"),
            "thinking_budget": start_request.get("thinking_budget"),
            "requested_reasoning_effort": start_request.get("requested_reasoning_effort"),
            "preserve_thinking": start_request.get("preserve_thinking"),
            "tool_count": start_request.get("tool_count"),
            "tool_choice": start_request.get("tool_choice"),
            "has_tool_history": start_request.get("has_tool_history"),
        }
    else:
        metrics["request_config"] = None

    censored: list[str] = []
    if terminal is None:
        censored.append("missing_terminal_event")
    if event_name != "request_done":
        censored.append(event_name)
    if not client["complete"]:
        censored.append("client_incomplete")
    if client["error_present"]:
        censored.append("client_error")

    if event_name == "request_done":
        assert terminal is not None
        result = _object(terminal.get("result"), f"{capsule_id} result", required=True)
        timings = _object(terminal.get("timings_seconds"), f"{capsule_id} timings_seconds", required=True)
        speculative = _object(terminal.get("speculative"), f"{capsule_id} speculative", required=True)
        assert result is not None and timings is not None and speculative is not None
        generated = _nonnegative_int(result.get("completion_tokens"), "completion_tokens")
        if generated is None:
            raise QuickEvalError(f"{capsule_id}: request_done has no completion_tokens")
        decode_seconds = _finite_number(timings.get("decode"), "timings_seconds.decode")
        server_ttft = _finite_number(timings.get("ttft"), "timings_seconds.ttft")
        total_seconds = _finite_number(timings.get("total"), "timings_seconds.total")
        if any(value is not None and value < 0 for value in (decode_seconds, server_ttft, total_seconds)):
            raise QuickEvalError(f"{capsule_id}: Serve timings must be nonnegative")
        rounds = _nonnegative_int(speculative.get("rounds"), "speculative.rounds")
        drafted = _nonnegative_int(speculative.get("drafted_tokens"), "speculative.drafted_tokens")
        accepted = _nonnegative_int(speculative.get("accepted_tokens"), "speculative.accepted_tokens")
        if rounds is None or drafted is None or accepted is None:
            raise QuickEvalError(f"{capsule_id}: request_done has incomplete speculative counters")
        finish_reason = result.get("finish_reason")
        if not isinstance(finish_reason, str):
            raise QuickEvalError(f"{capsule_id}: request_done has no finish_reason")
        decoded = max(generated - 1, 0)
        metrics.update(
            {
                "finish_reason": finish_reason,
                "generated_tokens": generated,
                "committed_tokens": generated,
                "decode_committed_tokens": decoded,
                "rounds": rounds,
                "drafted_tokens": drafted,
                "accepted_tokens": accepted,
                "tokens_per_round": (1.0 + accepted / rounds) if rounds > 0 else None,
                "server_ttft_seconds": server_ttft,
                "decode_seconds": decode_seconds,
                "decode_tok_s": decoded / decode_seconds if decode_seconds and decode_seconds > 0 else None,
                "server_total_seconds": total_seconds,
                "tool_call_parse": _tool_parse(result),
                "tool_call_count": _nonnegative_int(result.get("tool_call_count"), "tool_call_count"),
            }
        )
        if metrics["ttft_seconds"] is None:
            metrics["ttft_seconds"] = server_ttft
            metrics["ttft_source"] = "serve"
        if finish_reason not in FINISH_REASONS:
            censored.append(f"finish_{finish_reason}")
        if client["received_completion_tokens"] is not None and client["received_completion_tokens"] != generated:
            censored.append("client_server_token_mismatch")

    metrics["completed"] = not censored and event_name == "request_done"
    metrics["metrics_available"] = (
        metrics["decode_tok_s"] is not None and metrics["ttft_seconds"] is not None
    )
    metrics["censored_reasons"] = sorted(set(censored))
    metrics["status"] = "complete" if metrics["completed"] else "censored"
    identity = _object(row.get("identity"), f"{capsule_id} identity", required=True)
    assert identity is not None
    if not identity:
        raise QuickEvalError(f"{capsule_id}: identity must include exact artifact/runtime identity")
    metrics["identity"] = identity
    return metrics


def _summary_stats(values: list[float]) -> JsonObject:
    if not values:
        return {"count": 0, "mean": None, "median": None, "sample_stddev": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
    }


def _arm_summary(rows: list[JsonObject]) -> JsonObject:
    complete_rows = [row for row in rows if row.get("completed")]
    finish_counts: dict[str, int] = {}
    tool_summary: dict[str, Any] = {
        "marker_seen": 0,
        "structured_call_count": 0,
        "empty_arguments_omitted": 0,
        "schema_mismatch_arguments": 0,
        "fallback_reasons": {},
    }
    for row in rows:
        finish = row.get("finish_reason")
        if isinstance(finish, str):
            finish_counts[finish] = finish_counts.get(finish, 0) + 1
        diagnostics = row.get("tool_call_parse")
        if not isinstance(diagnostics, dict):
            continue
        if diagnostics.get("marker_seen") is True:
            tool_summary["marker_seen"] += 1
        for field in ("structured_call_count", "empty_arguments_omitted", "schema_mismatch_arguments"):
            value = diagnostics.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                tool_summary[field] += value
        fallback = diagnostics.get("fallback_reason")
        if isinstance(fallback, str):
            counts = tool_summary["fallback_reasons"]
            counts[fallback] = counts.get(fallback, 0) + 1

    numeric_fields = {
        "ttft_seconds": "ttft_seconds",
        "server_ttft_seconds": "server_ttft_seconds",
        "client_ttft_seconds": "client_ttft_seconds",
        "decode_tok_s": "decode_tok_s",
        "wall_seconds": "wall_seconds",
        "server_total_seconds": "server_total_seconds",
        "tokens_per_round": "tokens_per_round",
    }
    stats = {
        name: _summary_stats(
            [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        )
        for name, field in numeric_fields.items()
    }
    sum_fields = ("generated_tokens", "committed_tokens", "decode_committed_tokens", "rounds", "drafted_tokens", "accepted_tokens", "tool_call_count")
    totals = {
        field: sum(row[field] for row in rows if isinstance(row.get(field), int) and not isinstance(row.get(field), bool))
        for field in sum_fields
    }
    if totals["rounds"] > 0:
        totals["pooled_tokens_per_round"] = 1.0 + totals["accepted_tokens"] / totals["rounds"]
    else:
        totals["pooled_tokens_per_round"] = None
    vram_values = [row["vram_peak_bytes"] for row in rows if isinstance(row.get("vram_peak_bytes"), int)]
    identities = []
    for row in rows:
        identity = row.get("identity")
        if isinstance(identity, dict) and identity not in identities:
            identities.append(identity)
    return {
        "capsules": len(rows),
        "completed": len(complete_rows),
        "censored_or_failed": len(rows) - len(complete_rows),
        "cache_recipe_mismatch_capsules": sum(
            1 for row in rows if not row.get("cache_recipe_matches_plan", False)
        ),
        "missing_identity_capsules": sum(1 for row in rows if not row.get("identity")),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "metrics": stats,
        "token_and_round_totals": totals,
        "tool_call_parse": tool_summary,
        "error_capsules": sum(1 for row in rows if row.get("censored_reasons")),
        "vram_peak_bytes": max(vram_values) if vram_values else None,
        "identities_observed": identities,
        "server_instance_ids": sorted(
            {row["server_instance_id"] for row in rows if isinstance(row.get("server_instance_id"), str)}
        ),
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ratio_uncertainty(ratios: list[float], seed: int, resamples: int) -> JsonObject:
    positive = [ratio for ratio in ratios if ratio > 0.0 and math.isfinite(ratio)]
    logs = [math.log(ratio) for ratio in positive]
    if not logs:
        return {"count": 0, "geometric_mean_ratio": None, "log_sample_stddev": None, "log_standard_error": None, "bootstrap_95_ci": None}
    mean_log = statistics.fmean(logs)
    sample_sd = statistics.stdev(logs) if len(logs) > 1 else None
    ci: list[float] | None = None
    if len(logs) > 1 and resamples > 0:
        rng = random.Random(seed)
        bootstrap_means = [
            statistics.fmean(rng.choice(logs) for _ in logs) for _ in range(resamples)
        ]
        ci = [
            math.exp(_percentile(bootstrap_means, 0.025)),
            math.exp(_percentile(bootstrap_means, 0.975)),
        ]
    return {
        "count": len(logs),
        "geometric_mean_ratio": math.exp(mean_log),
        "log_sample_stddev": sample_sd,
        "log_standard_error": sample_sd / math.sqrt(len(logs)) if sample_sd is not None else None,
        "bootstrap_95_ci": ci,
    }


def _classify(
    decode: JsonObject,
    ttft: JsonObject,
    *,
    comparable_pairs: int,
    win_blockers: list[str],
) -> tuple[str, str]:
    if comparable_pairs < MIN_PAIRS_FOR_CLASSIFICATION:
        decode_ratio = decode.get("geometric_mean_ratio")
        ttft_ratio = ttft.get("geometric_mean_ratio")
        severe_decode_regression = (
            isinstance(decode_ratio, (int, float))
            and decode_ratio < 0.70
            and (ttft_ratio is None or ttft_ratio >= 0.95)
        )
        severe_ttft_regression = (
            isinstance(ttft_ratio, (int, float))
            and ttft_ratio > 1.30
            and (decode_ratio is None or decode_ratio <= 1.05)
        )
        if comparable_pairs > 0 and (severe_decode_regression or severe_ttft_regression):
            return (
                "obvious regression",
                "measured effect exceeds 30% in the available pairs; sample count is too small for a bootstrap interval",
            )
        return "flat/noisy", f"only {comparable_pairs} complete paired samples; need {MIN_PAIRS_FOR_CLASSIFICATION}"
    decode_ci = decode.get("bootstrap_95_ci")
    ttft_ci = ttft.get("bootstrap_95_ci")
    if not isinstance(decode_ci, list) or not isinstance(ttft_ci, list):
        return "flat/noisy", "paired uncertainty interval is unavailable"

    decode_regression = decode_ci[1] < 0.95
    ttft_regression = ttft_ci[0] > 1.05
    decode_improvement = decode_ci[0] > 1.05
    ttft_improvement = ttft_ci[1] < 0.95
    if (decode_regression and not ttft_improvement) or (ttft_regression and not decode_improvement):
        return "obvious regression", "paired 95% bootstrap interval shows a material decode or TTFT regression"
    if win_blockers:
        return "flat/noisy", "a candidate failure, censored response, shorter output, or finish mismatch blocks a win"
    if (decode_improvement and not ttft_regression) or (ttft_improvement and not decode_regression):
        return "promising", "paired 95% bootstrap interval clears a 5% improvement threshold without a material tradeoff"
    return "flat/noisy", "paired differences do not clear the 5% screen thresholds"


def reduce_pair(
    plan: JsonObject,
    baseline_rows: list[JsonObject],
    candidate_rows: list[JsonObject],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonObject:
    selected = _validate_plan(plan)
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 0:
        raise QuickEvalError("bootstrap_resamples must be a nonnegative integer")

    def index_rows(rows: list[JsonObject], arm: str) -> dict[str, JsonObject]:
        indexed: dict[str, JsonObject] = {}
        known = {item["capsule_id"] for item in selected}
        for row in rows:
            capsule_id = row.get("capsule_id")
            if not isinstance(capsule_id, str) or capsule_id not in known:
                raise QuickEvalError(f"{arm} result contains an unplanned capsule_id")
            if capsule_id in indexed:
                raise QuickEvalError(f"{arm} result repeats capsule_id {capsule_id}")
            indexed[capsule_id] = row
        return indexed

    baseline_by_id = index_rows(baseline_rows, "baseline")
    candidate_by_id = index_rows(candidate_rows, "candidate")
    baseline_metrics = [
        _arm_capsule(baseline_by_id.get(item["capsule_id"]), plan, item) for item in selected
    ]
    candidate_metrics = [
        _arm_capsule(candidate_by_id.get(item["capsule_id"]), plan, item) for item in selected
    ]

    paired: list[JsonObject] = []
    decode_ratios: list[float] = []
    ttft_ratios: list[float] = []
    wall_ratios: list[float] = []
    win_blockers: list[str] = []
    complete_response_pairs = 0
    comparable_count = 0
    candidate_shorter_count = 0
    for base, candidate in zip(baseline_metrics, candidate_metrics, strict=True):
        pair_reasons: list[str] = []
        base_ttft: float | None = None
        candidate_ttft: float | None = None
        ttft_source: str | None = None
        responses_complete = bool(base.get("completed") and candidate.get("completed"))
        configuration_match = (
            base.get("cache_recipe_matches_plan") is True
            and candidate.get("cache_recipe_matches_plan") is True
            and isinstance(base.get("request_config"), dict)
            and base.get("request_config") == candidate.get("request_config")
        )
        metrics_valid = responses_complete and configuration_match
        if not responses_complete:
            pair_reasons.append("incomplete_or_censored")
        if not base.get("cache_recipe_matches_plan") or not candidate.get("cache_recipe_matches_plan"):
            pair_reasons.append("cache_recipe_mismatch")
        if base.get("request_config") != candidate.get("request_config"):
            pair_reasons.append("request_configuration_mismatch")
        if responses_complete:
            complete_response_pairs += 1
        if metrics_valid:
            comparable_count += 1
            base_tokens = base.get("generated_tokens")
            candidate_tokens = candidate.get("generated_tokens")
            if isinstance(base_tokens, int) and isinstance(candidate_tokens, int) and candidate_tokens < base_tokens:
                candidate_shorter_count += 1
                pair_reasons.append("candidate_shorter_output")
            if base.get("finish_reason") != candidate.get("finish_reason"):
                pair_reasons.append("finish_reason_mismatch")
            base_client_finish = base.get("client_finish_reason")
            candidate_client_finish = candidate.get("client_finish_reason")
            if base_client_finish != candidate_client_finish and (
                base_client_finish is not None or candidate_client_finish is not None
            ):
                # External protocol labels (for example, Anthropic `max_tokens`) need not
                # equal Serve's internal finish reason, but should agree between the arms.
                pair_reasons.append("client_finish_mismatch")
            base_decode = base.get("decode_tok_s")
            candidate_decode = candidate.get("decode_tok_s")
            if isinstance(base_decode, (int, float)) and base_decode > 0 and isinstance(candidate_decode, (int, float)) and candidate_decode > 0:
                decode_ratios.append(candidate_decode / base_decode)
            base_client_ttft = base.get("client_ttft_seconds")
            candidate_client_ttft = candidate.get("client_ttft_seconds")
            if (
                isinstance(base_client_ttft, (int, float))
                and base_client_ttft > 0
                and isinstance(candidate_client_ttft, (int, float))
                and candidate_client_ttft > 0
            ):
                base_ttft = base_client_ttft
                candidate_ttft = candidate_client_ttft
                ttft_source = "client"
            else:
                base_ttft = base.get("server_ttft_seconds")
                candidate_ttft = candidate.get("server_ttft_seconds")
                ttft_source = "serve"
            if isinstance(base_ttft, (int, float)) and base_ttft > 0 and isinstance(candidate_ttft, (int, float)) and candidate_ttft > 0:
                ttft_ratios.append(candidate_ttft / base_ttft)
            else:
                ttft_source = None
            base_wall = base.get("wall_seconds")
            candidate_wall = candidate.get("wall_seconds")
            if isinstance(base_wall, (int, float)) and base_wall > 0 and isinstance(candidate_wall, (int, float)) and candidate_wall > 0:
                wall_ratios.append(candidate_wall / base_wall)
        else:
            pair_reasons.extend(base.get("censored_reasons", []))
            pair_reasons.extend(candidate.get("censored_reasons", []))
        semantic_flags_match = (
            base.get("semantic_transform") == candidate.get("semantic_transform")
            and base.get("production_compatible") == candidate.get("production_compatible")
        )
        if not semantic_flags_match:
            pair_reasons.append("semantic_compatibility_mismatch")
        production_compatible_pair = (
            semantic_flags_match
            and base.get("production_compatible") is True
            and candidate.get("production_compatible") is True
        )
        if not production_compatible_pair:
            pair_reasons.append("not_production_compatible")
        exploratory_eligible = metrics_valid and not any(
            reason != "not_production_compatible" for reason in pair_reasons
        )
        eligible = exploratory_eligible and production_compatible_pair
        if not exploratory_eligible:
            win_blockers.append(base.get("capsule_id", "unknown"))
        paired.append(
            {
                "capsule_id": base.get("capsule_id"),
                "baseline": base,
                "candidate": candidate,
                "responses_complete": responses_complete,
                "metrics_comparable": metrics_valid,
                "candidate_win_eligible": eligible,
                "exploratory_candidate_win_eligible": exploratory_eligible,
                "semantic_transform": bool(
                    base.get("semantic_transform") or candidate.get("semantic_transform")
                ),
                "production_compatible": production_compatible_pair,
                "win_ineligibility_reasons": sorted(set(pair_reasons)),
                "decode_speed_ratio": (
                    candidate["decode_tok_s"] / base["decode_tok_s"]
                    if isinstance(candidate.get("decode_tok_s"), (int, float))
                    and isinstance(base.get("decode_tok_s"), (int, float))
                    and base["decode_tok_s"] > 0
                    else None
                ),
                "ttft_ratio": (candidate_ttft / base_ttft if ttft_source is not None else None),
                "ttft_source": ttft_source,
                "wall_ratio": (
                    candidate["wall_seconds"] / base["wall_seconds"]
                    if isinstance(candidate.get("wall_seconds"), (int, float))
                    and isinstance(base.get("wall_seconds"), (int, float))
                    and base["wall_seconds"] > 0
                    else None
                ),
            }
        )

    plan_seed = plan.get("seed", 0)
    seed_base = int(hashlib.sha256(f"{plan.get('plan_id')}:{plan_seed}".encode()).hexdigest()[:16], 16)
    decode_uncertainty = _ratio_uncertainty(decode_ratios, seed_base ^ 0xD3C0DE, bootstrap_resamples)
    ttft_uncertainty = _ratio_uncertainty(ttft_ratios, seed_base ^ 0x77F7, bootstrap_resamples)
    wall_uncertainty = _ratio_uncertainty(wall_ratios, seed_base ^ 0xA11, bootstrap_resamples)
    exploratory_classification, exploratory_classification_reason = _classify(
        decode_uncertainty,
        ttft_uncertainty,
        comparable_pairs=comparable_count,
        win_blockers=win_blockers,
    )
    semantic_transform = bool(plan.get("semantic_transform", False)) or any(
        row.get("semantic_transform") is True
        for row in selected
    )
    production_compatible = (
        plan.get("production_compatible", not semantic_transform) is True
        and not semantic_transform
    )
    if production_compatible:
        classification = exploratory_classification
        classification_reason = exploratory_classification_reason
        production_classification = exploratory_classification
    else:
        production_classification = "ineligible"
        if exploratory_classification == "promising":
            classification = "exploratory only"
            classification_reason = (
                "the paired statistical screen is promising, but this workload is not "
                "production-compatible"
            )
        else:
            classification = exploratory_classification
            classification_reason = exploratory_classification_reason
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "classification": classification,
        "classification_reason": classification_reason,
        "exploratory_classification": exploratory_classification,
        "exploratory_classification_reason": exploratory_classification_reason,
        "production_classification": production_classification,
        "semantic_transform": semantic_transform,
        "production_compatible": production_compatible,
        "thresholds": {
            "minimum_complete_pairs": MIN_PAIRS_FOR_CLASSIFICATION,
            "material_change_fraction": 0.05,
            "small_sample_obvious_regression_fraction": 0.30,
            "interval": "paired percentile bootstrap 95% CI of mean log ratio",
        },
        "identity": {
            "seed": plan.get("seed"),
            "sampler": plan.get("sampler"),
            "cache_recipe_id": plan.get("cache_recipe_id"),
            "output_budget": plan.get("output_budget"),
            "selected_capsules": [
                {
                    "capsule_id": item["capsule_id"],
                    "request_sha256": item.get("request_sha256"),
                    "capsule_path": item.get("capsule_path"),
                    "bucket_labels": item.get("bucket_labels"),
                    "stratified_for": item.get("stratified_for"),
                    "semantic_transform": item.get("semantic_transform", False),
                    "production_compatible": item.get(
                        "production_compatible", not item.get("semantic_transform", False)
                    ),
                    "parent_request_sha256": item.get("parent_request_sha256"),
                    "derived_request_sha256": item.get("derived_request_sha256"),
                    "transform_diff": item.get("transform_diff"),
                }
                for item in selected
            ],
            "semantic_transform_provenance": [
                {
                    "capsule_id": item["capsule_id"],
                    "parent_request_sha256": item.get("parent_request_sha256"),
                    "derived_request_sha256": item.get("derived_request_sha256"),
                    "transform_diff": item.get("transform_diff"),
                }
                for item in selected
                if item.get("semantic_transform") is True
            ],
        },
        "arms": {
            "baseline": _arm_summary(baseline_metrics),
            "candidate": _arm_summary(candidate_metrics),
        },
        "paired": {
            "planned": len(selected),
            "complete_responses": complete_response_pairs,
            "complete_comparable": comparable_count,
            "candidate_shorter_outputs": candidate_shorter_count,
            "candidate_win_blocked_capsules": win_blockers,
            "uncertainty": {
                "decode_speed_ratio_candidate_over_baseline": decode_uncertainty,
                "ttft_ratio_candidate_over_baseline": ttft_uncertainty,
                "wall_ratio_candidate_over_baseline": wall_uncertainty,
            },
            "capsules": paired,
        },
    }


def _write_json(path: Path, value: JsonObject) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise QuickEvalError(f"cannot write {path}: {exc.strerror or exc}") from exc


def _sample_command(args: argparse.Namespace) -> JsonObject:
    descriptors = _load_jsonl(Path(args.manifest))
    plan = make_plan(
        descriptors,
        seed=args.seed,
        per_bucket=args.per_bucket,
        cache_recipe_id=args.cache_recipe_id,
        output_budget=args.output_budget,
        bucket_labels=args.bucket_label,
    )
    _write_json(Path(args.output), plan)
    return plan


def _reduce_command(args: argparse.Namespace) -> JsonObject:
    plan = _load_json(Path(args.plan))
    report = reduce_pair(
        plan,
        _load_jsonl(Path(args.baseline_results)),
        _load_jsonl(Path(args.candidate_results)),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    _write_json(Path(args.output), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample", help="make a deterministic sparse capsule plan")
    sample.add_argument("--manifest", required=True, help="capture descriptor JSONL")
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--per-bucket", type=int, required=True)
    sample.add_argument("--bucket-label", action="append", help="restrict to an observed label; repeatable")
    sample.add_argument("--cache-recipe-id", required=True)
    sample.add_argument("--output-budget", type=int, required=True)
    sample.add_argument("--output", required=True, help="write the paired plan JSON")
    reduce = subparsers.add_parser("reduce", help="reduce paired per-capsule Serve result JSONL")
    reduce.add_argument("--plan", required=True)
    reduce.add_argument("--baseline-results", required=True)
    reduce.add_argument("--candidate-results", required=True)
    reduce.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    reduce.add_argument("--output", required=True, help="write the machine-readable report JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "sample":
            plan = _sample_command(args)
            print(json.dumps({"plan_id": plan["plan_id"], "selected": len(plan["selected"]), "output": args.output}))
        else:
            report = _reduce_command(args)
            print(
                json.dumps(
                    {
                        "plan_id": report["plan_id"],
                        "classification": report["classification"],
                        "report": args.output,
                    }
                )
            )
    except (QuickEvalError, OSError) as exc:
        print(f"quick_eval: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
