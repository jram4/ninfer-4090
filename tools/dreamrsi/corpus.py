#!/usr/bin/env python3
"""Create deterministic session-stratified replay plans from private descriptors.

Plans contain sanitized metadata and capsule paths only; capsule bodies remain
outside reports. Sessions, not individual requests, are assigned to disjoint
screen and holdout groups. A small or unbalanced corpus is explicitly marked
inconclusive rather than treated as a release gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MIN_SCREEN_SESSIONS = 8
MIN_HOLDOUT_SESSIONS = 4
MIN_SCREEN_PAIRS = 12
TRAFFIC_CLASSES = {"real_user", "agent_generated", "derived"}
_CAPSULE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
JsonObject = dict[str, Any]


class CorpusError(ValueError):
    """Raised when a descriptor or plan request violates the corpus contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, max_length: int = 1024) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise CorpusError(f"{name} must be a nonempty string of at most {max_length} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise CorpusError(f"{name} must not contain control characters")
    return value.strip()


def _sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    digest = _text(value, name, max_length=64)
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise CorpusError(f"{name} must be a 64-digit hexadecimal SHA-256")
    return digest.lower()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusError(f"{name} must be a nonnegative integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorpusError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise CorpusError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise CorpusError(f"{name} must be a finite positive number")
    return number


def _validate_descriptor(row: Any, line_number: int) -> JsonObject:
    prefix = f"descriptor line {line_number}"
    if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
        raise CorpusError(f"{prefix}: schema_version must be 1")
    capsule_id = _text(row.get("capsule_id"), f"{prefix} capsule_id", max_length=256)
    if not _CAPSULE_ID.fullmatch(capsule_id):
        raise CorpusError(f"{prefix}: capsule_id contains unsafe characters")
    capsule_path = _text(row.get("capsule_path"), f"{prefix} capsule_path")
    capsule_hash = _sha256(row.get("capsule_sha256"), f"{prefix} capsule_sha256")
    identity_scope = row.get("identity_scope")
    if not isinstance(identity_scope, str) or identity_scope not in {
        "exact_body_no_credentials",
        "redacted_no_body_identity",
    }:
        raise CorpusError(f"{prefix}: identity_scope is unsupported")
    request_hash = _sha256(row.get("request_sha256"), f"{prefix} request_sha256", optional=True)
    response_hash = _sha256(row.get("response_sha256"), f"{prefix} response_sha256", optional=True)
    if identity_scope == "exact_body_no_credentials" and request_hash is None:
        raise CorpusError(f"{prefix}: exact request identity requires request_sha256")
    if identity_scope == "redacted_no_body_identity" and request_hash is not None:
        raise CorpusError(f"{prefix}: redacted body identity must not retain a body hash")
    traffic_class = row.get("traffic_class")
    if not isinstance(traffic_class, str) or traffic_class not in TRAFFIC_CLASSES:
        raise CorpusError(f"{prefix}: traffic_class must distinguish real_user, agent_generated, or derived")
    derived_from = row.get("derived_from")
    if traffic_class == "derived":
        derived_from = _text(derived_from, f"{prefix} derived_from", max_length=256)
        if not _CAPSULE_ID.fullmatch(derived_from):
            raise CorpusError(f"{prefix}: derived_from is not a capsule ID")
    elif derived_from is not None:
        raise CorpusError(f"{prefix}: derived_from is only valid for derived traffic")
    labels_value = row.get("labels", [])
    if not isinstance(labels_value, list):
        raise CorpusError(f"{prefix}: labels must be an array")
    labels = sorted({_text(label, f"{prefix} label", max_length=200) for label in labels_value})
    if len(labels) != len(labels_value):
        raise CorpusError(f"{prefix}: labels must be unique")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CorpusError(f"{prefix}: metadata must be an object")
    output_budget = row.get("output_budget")
    if output_budget is not None:
        output_budget = _positive_int(output_budget, f"{prefix} output_budget")
    captured_at = row.get("captured_at_utc", metadata.get("captured_at_utc"))
    captured_at = _text(captured_at, f"{prefix} captured_at_utc", max_length=64)
    if not captured_at.endswith("Z"):
        raise CorpusError(f"{prefix}: captured_at_utc must use UTC Z notation")
    try:
        datetime.fromisoformat(captured_at[:-1] + "+00:00")
    except ValueError as exc:
        raise CorpusError(f"{prefix}: captured_at_utc is invalid") from exc
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": capsule_id,
        "sequence": _positive_int(row.get("sequence"), f"{prefix} sequence"),
        "capsule_path": capsule_path,
        "capsule_sha256": capsule_hash,
        "request_sha256": request_hash,
        "response_sha256": response_hash,
        "identity_scope": identity_scope,
        "session_id": _text(row.get("session_id"), f"{prefix} session_id", max_length=256),
        "task_label": _text(row.get("task_label"), f"{prefix} task_label", max_length=256),
        "run_label": _text(row.get("run_label"), f"{prefix} run_label", max_length=256),
        "traffic_class": traffic_class,
        "derived_from": derived_from,
        "stratum": _text(row.get("stratum"), f"{prefix} stratum", max_length=200),
        "labels": labels,
        "captured_at_utc": captured_at,
        "output_budget": output_budget,
    }


def read_descriptors(path: str | Path) -> list[JsonObject]:
    """Read and validate capture descriptor JSONL without loading capsule bodies."""
    descriptor_path = Path(path)
    rows: list[JsonObject] = []
    try:
        with descriptor_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorpusError(f"{descriptor_path}:{line_number}: invalid JSON") from exc
                rows.append(_validate_descriptor(value, line_number))
    except OSError as exc:
        raise CorpusError(f"cannot read {descriptor_path}: {exc.strerror or exc}") from exc
    return rows


def _descriptor_set(descriptors: list[JsonObject]) -> list[JsonObject]:
    rows = [_validate_descriptor(row, index) for index, row in enumerate(descriptors, 1)]
    if not rows:
        raise CorpusError("descriptor manifest is empty")
    ids: set[str] = set()
    by_session: dict[str, dict[str, str]] = {}
    session_sequences: dict[str, set[int]] = {}
    for row in rows:
        if row["capsule_id"] in ids:
            raise CorpusError(f"duplicate capsule_id: {row['capsule_id']}")
        ids.add(row["capsule_id"])
        session = row["session_id"]
        signature = {
            key: row[key]
            for key in ("task_label", "run_label", "traffic_class", "stratum")
        }
        previous = by_session.setdefault(session, signature)
        if previous != signature:
            raise CorpusError(f"session {session} has inconsistent task, run, traffic class, or stratum labels")
        sequences = session_sequences.setdefault(session, set())
        if row["sequence"] in sequences:
            raise CorpusError(f"session {session} repeats capture sequence {row['sequence']}")
        sequences.add(row["sequence"])
    return rows


def _rank(seed: int, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{purpose}\0{value}".encode("utf-8")).hexdigest()


def _capsule_reference(row: JsonObject, *, role: str) -> JsonObject:
    return {
        "sequence": row["sequence"],
        "role": role,
        "capsule_id": row["capsule_id"],
        "capsule_path": row["capsule_path"],
        "capsule_sha256": row["capsule_sha256"],
        "request_sha256": row["request_sha256"],
        "identity_scope": row["identity_scope"],
        "output_budget": row["output_budget"],
    }


def _session_bundle(
    session_rows: list[JsonObject],
    *,
    samples_per_session: int,
    warmup_requests: int,
    output_budget_filter: int | None,
) -> tuple[JsonObject | None, str | None]:
    """Build one replayable contiguous captured-session prefix bundle."""
    ordered = sorted(session_rows, key=lambda row: row["sequence"])
    candidates = [
        row
        for row in ordered
        if row["sequence"] > warmup_requests
        and (output_budget_filter is None or row["output_budget"] == output_budget_filter)
    ]
    if len(candidates) < samples_per_session:
        if output_budget_filter is not None and not candidates:
            return None, "no_matching_output_budget"
        return None, "insufficient_measured_requests"

    measured = candidates[:samples_per_session]
    last_sequence = measured[-1]["sequence"]
    prefix = [row for row in ordered if row["sequence"] <= last_sequence]
    if [row["sequence"] for row in prefix] != list(range(1, last_sequence + 1)):
        return None, "missing_captured_prefix_sequence"
    if any(row["identity_scope"] != "exact_body_no_credentials" for row in prefix):
        return None, "redacted_prefix_not_replayable"

    measured_ids = {row["capsule_id"] for row in measured}
    replay_sequence = [
        _capsule_reference(
            row,
            role="measured" if row["capsule_id"] in measured_ids else "warmup",
        )
        for row in prefix
    ]
    return (
        {
            "session_id": session_rows[0]["session_id"],
            "stratum": session_rows[0]["stratum"],
            "measured": measured,
            "measured_capsule_ids": [row["capsule_id"] for row in measured],
            "replay_sequence": replay_sequence,
            "warmup_requests": sum(item["role"] == "warmup" for item in replay_sequence),
            "request_cost": len(replay_sequence),
        },
        None,
    )


def make_plan(
    descriptors: list[JsonObject],
    *,
    seed: int,
    samples_per_session: int = 2,
    output_budget: int | None = None,
    request_budget: int = 64,
    cache_recipe_id: str,
    cache_state: str = "unknown",
    warmup_requests: int = 0,
    holdout_fraction: float = 0.25,
    stratum_weights: dict[str, float] | None = None,
    strata: list[str] | None = None,
) -> JsonObject:
    """Build a budget-bounded, session-stratified screen/holdout plan.

    Sessions, rather than every session's requests, are sampled under a per-arm
    total replay budget. Each selected session replays its captured sequence
    from sequence 1 through the last measured request; unmeasured requests in
    that prefix are explicit warmups. ``output_budget`` is only an optional
    cohort filter. ``strata`` optionally restricts the plan to a subset of the
    observed strata, so a screen can target the dominant traffic class
    without discarding the rest of the corpus. Captured request bodies and
    per-request budgets are never rewritten.
    """
    samples_per_session = _positive_int(samples_per_session, "samples_per_session")
    if output_budget is not None:
        output_budget = _positive_int(output_budget, "output_budget cohort filter")
    request_budget = _positive_int(request_budget, "request_budget")
    warmup_requests = _nonnegative_int(warmup_requests, "warmup_requests")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CorpusError("seed must be an integer")
    if not isinstance(cache_recipe_id, str) or not cache_recipe_id.strip():
        raise CorpusError("cache_recipe_id must be a nonempty string")
    if not isinstance(cache_state, str) or cache_state not in {"unknown", "cold", "warm"}:
        raise CorpusError("cache_state must be unknown, cold, or warm")
    if isinstance(holdout_fraction, bool) or not isinstance(holdout_fraction, (float, int)):
        raise CorpusError("holdout_fraction must be between zero and one")
    try:
        holdout_fraction = float(holdout_fraction)
    except OverflowError as exc:
        raise CorpusError("holdout_fraction must be between zero and one") from exc
    if not math.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
        raise CorpusError("holdout_fraction must be between zero and one")

    source_rows = _descriptor_set(descriptors)
    traffic_classes = {row["traffic_class"] for row in source_rows}
    if len(traffic_classes) != 1:
        raise CorpusError("one corpus plan may contain only one traffic_class; plan real, generated, or derived traffic separately")
    observed_strata = sorted({row["stratum"] for row in source_rows})
    if strata is None:
        selected_strata = observed_strata
    else:
        if not isinstance(strata, list) or not strata:
            raise CorpusError("strata must be a nonempty list when provided")
        unknown = sorted(set(strata) - set(observed_strata))
        if unknown:
            raise CorpusError("strata not observed in the corpus: " + ", ".join(unknown))
        selected_strata = sorted(set(strata))
    strata = selected_strata
    weights: dict[str, float] | None = None
    if stratum_weights is not None:
        if not isinstance(stratum_weights, dict) or not stratum_weights:
            raise CorpusError("stratum_weights must be a nonempty object when provided")
        weights = {}
        for label, weight in stratum_weights.items():
            normalized_label = _text(label, "stratum weight key", max_length=200)
            if normalized_label in weights:
                raise CorpusError("stratum_weights contains duplicate normalized keys")
            weights[normalized_label] = _finite_positive(weight, f"stratum weight {label}")
        if set(weights) != set(strata):
            raise CorpusError("stratum_weights must provide exactly one positive weight for every observed stratum")

    by_session: dict[str, list[JsonObject]] = {}
    for row in source_rows:
        by_session.setdefault(row["session_id"], []).append(row)
    corpus_identity = [
        {
            key: row[key]
            for key in (
                "capsule_id",
                "sequence",
                "capsule_sha256",
                "request_sha256",
                "response_sha256",
                "identity_scope",
                "session_id",
                "task_label",
                "run_label",
                "traffic_class",
                "derived_from",
                "stratum",
                "labels",
                "captured_at_utc",
                "output_budget",
            )
        }
        for row in sorted(source_rows, key=lambda item: item["capsule_id"])
    ]
    corpus_id = canonical_hash({"schema_version": SCHEMA_VERSION, "descriptors": corpus_identity})

    candidates_by_stratum: dict[str, list[JsonObject]] = {stratum: [] for stratum in strata}
    ineligible_by_stratum: dict[str, dict[str, int]] = {stratum: {} for stratum in strata}
    matching_sessions_by_stratum: dict[str, set[str]] = {stratum: set() for stratum in strata}
    for session_rows in by_session.values():
        stratum = session_rows[0]["stratum"]
        if stratum not in candidates_by_stratum:
            # A stratum the plan does not cover is left out entirely.
            continue
        if output_budget is None or any(row["output_budget"] == output_budget for row in session_rows):
            matching_sessions_by_stratum[stratum].add(session_rows[0]["session_id"])
        bundle, failure = _session_bundle(
            session_rows,
            samples_per_session=samples_per_session,
            warmup_requests=warmup_requests,
            output_budget_filter=output_budget,
        )
        if bundle is None:
            assert failure is not None
            failures = ineligible_by_stratum[stratum]
            failures[failure] = failures.get(failure, 0) + 1
        else:
            candidates_by_stratum[stratum].append(bundle)

    stratum_order = sorted(strata, key=lambda stratum: _rank(seed, "stratum-order", stratum))
    ranked_sessions = {
        stratum: sorted(
            candidates_by_stratum[stratum],
            key=lambda bundle: _rank(seed, f"session:{stratum}", bundle["session_id"]),
        )
        for stratum in strata
    }
    selected_by_stratum: dict[str, list[JsonObject]] = {stratum: [] for stratum in strata}
    selected_session_ids: set[str] = set()
    remaining_budget = request_budget

    def select_next(stratum: str) -> bool:
        nonlocal remaining_budget
        for bundle in ranked_sessions[stratum]:
            if bundle["session_id"] in selected_session_ids:
                continue
            if bundle["request_cost"] > remaining_budget:
                continue
            selected_session_ids.add(bundle["session_id"])
            selected_by_stratum[stratum].append(bundle)
            remaining_budget -= bundle["request_cost"]
            return True
        return False

    # First try to reserve one independent session for each holdout and screen
    # group where the source stratum has enough eligible sessions.
    minimum_sessions = {
        stratum: min(2, len(ranked_sessions[stratum])) for stratum in strata
    }
    while True:
        advanced = False
        for stratum in stratum_order:
            if len(selected_by_stratum[stratum]) < minimum_sessions[stratum]:
                advanced |= select_next(stratum)
        if not advanced:
            break

    # Spend remaining budget in deterministic stratified rounds. This samples
    # a subset of sessions for large corpora instead of failing after selecting
    # requests from every session.
    while True:
        advanced = False
        for stratum in stratum_order:
            advanced |= select_next(stratum)
        if not advanced:
            break

    split_bundles: dict[str, list[JsonObject]] = {"screen": [], "holdout": []}
    selected_by_split: dict[str, list[JsonObject]] = {"screen": [], "holdout": []}
    session_replay: dict[str, list[JsonObject]] = {"screen": [], "holdout": []}
    screen_sessions: list[str] = []
    holdout_sessions: list[str] = []
    for stratum in strata:
        bundles = selected_by_stratum[stratum]
        ranked_selected = sorted(
            bundles,
            key=lambda bundle: _rank(seed, f"holdout:{stratum}", bundle["session_id"]),
        )
        holdout_count = 0
        if len(ranked_selected) >= 2:
            holdout_count = min(
                len(ranked_selected) - 1,
                max(1, math.ceil(len(ranked_selected) * holdout_fraction)),
            )
        holdout_ids = {bundle["session_id"] for bundle in ranked_selected[:holdout_count]}
        for bundle in bundles:
            split_name = "holdout" if bundle["session_id"] in holdout_ids else "screen"
            split_bundles[split_name].append(bundle)
            session_replay[split_name].append(
                {
                    "session_id": bundle["session_id"],
                    "stratum": bundle["stratum"],
                    "warmup_requests": bundle["warmup_requests"],
                    "measured_capsule_ids": bundle["measured_capsule_ids"],
                    "replay_sequence": bundle["replay_sequence"],
                }
            )
            (holdout_sessions if split_name == "holdout" else screen_sessions).append(bundle["session_id"])
            for row in bundle["measured"]:
                planned = dict(row)
                planned["session_warmup_requests"] = bundle["warmup_requests"]
                selected_by_split[split_name].append(planned)

    for split_name in ("screen", "holdout"):
        split_bundles[split_name].sort(
            key=lambda bundle: _rank(seed, f"{split_name}-order", bundle["session_id"])
        )
        session_replay[split_name].sort(
            key=lambda session: _rank(seed, f"{split_name}-order", session["session_id"])
        )
        selected_by_split[split_name].sort(
            key=lambda row: (
                _rank(seed, f"{split_name}-order", row["session_id"]),
                row["sequence"],
            )
        )
    screen_sessions.sort(key=lambda value: _rank(seed, "screen-order", value))
    holdout_sessions.sort(key=lambda value: _rank(seed, "holdout-order", value))
    if set(screen_sessions) & set(holdout_sessions):
        raise AssertionError("session split overlap")

    coverage_strata: list[JsonObject] = []
    for stratum in strata:
        source_sessions = [
            session for session, session_rows in by_session.items()
            if session_rows[0]["stratum"] == stratum
        ]
        selected_bundles = selected_by_stratum[stratum]
        selected_sessions = {bundle["session_id"] for bundle in selected_bundles}
        screen_count = sum(bundle["session_id"] in set(screen_sessions) for bundle in selected_bundles)
        holdout_count = sum(bundle["session_id"] in set(holdout_sessions) for bundle in selected_bundles)
        eligible_count = len(ranked_sessions[stratum])
        if eligible_count == 0:
            reason_counts = ineligible_by_stratum[stratum]
            status = (
                "no_matching_output_budget"
                if output_budget is not None and not matching_sessions_by_stratum[stratum]
                else "no_replayable_contiguous_prefix"
            )
        elif len(source_sessions) < 2:
            status = "single_source_session_no_holdout"
        elif len(selected_sessions) < 2 or not screen_count or not holdout_count:
            status = "budget_limited_session_coverage"
        else:
            status = "covered"
        coverage_strata.append(
            {
                "stratum": stratum,
                "source_sessions": len(source_sessions),
                "cohort_sessions": len(matching_sessions_by_stratum[stratum]),
                "eligible_sessions": eligible_count,
                "selected_sessions": len(selected_sessions),
                "screen_sessions": screen_count,
                "holdout_sessions": holdout_count,
                "ineligible_reasons": ineligible_by_stratum[stratum],
                "status": status,
            }
        )
    coverage_complete = all(item["status"] == "covered" for item in coverage_strata)
    coverage = {
        "complete": coverage_complete,
        "missing_strata": [item["stratum"] for item in coverage_strata if item["status"] != "covered"],
        "strata": coverage_strata,
    }

    measured_count = sum(len(values) for values in selected_by_split.values())
    total_replay_count = sum(
        len(session["replay_sequence"])
        for sessions in session_replay.values()
        for session in sessions
    )
    warmup_count = total_replay_count - measured_count
    # A harness must run every unmeasured request in each selected session's
    # captured prefix, so the arm warmup budget is the worst session's count.
    max_session_warmup = max(
        (session["warmup_requests"] for sessions in session_replay.values() for session in sessions),
        default=0,
    )
    screen_pair_count = len(selected_by_split["screen"])
    holdout_pair_count = len(selected_by_split["holdout"])
    adequate = (
        coverage_complete
        and len(screen_sessions) >= MIN_SCREEN_SESSIONS
        and screen_pair_count >= MIN_SCREEN_PAIRS
        and len(holdout_sessions) >= MIN_HOLDOUT_SESSIONS
    )
    core: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "seed": seed,
        "sampling": {
            "name": "session_stratified_order_preserving",
            "version": 1,
            "samples_per_session": samples_per_session,
            "holdout_fraction": holdout_fraction,
        },
        "cache": {
            "cache_recipe_id": _text(cache_recipe_id, "cache_recipe_id", max_length=256),
            "cache_state": cache_state,
            "warmup_requests_per_arm": max_session_warmup,
            "minimum_head_warmup_skipped": warmup_requests,
        },
        "budgets": {
            # A null cohort filter means each captured request keeps its own
            # recorded output budget. Bodies are never rewritten.
            "output_tokens_per_request": output_budget,
            "per_request_budget_preserved": output_budget is None,
            "measured_requests_total": request_budget,
            "selected_requests_total": measured_count,
            "replay_requests_total": total_replay_count,
            "warmup_requests_per_arm": max_session_warmup,
            "total_warmup_requests": warmup_count,
        },
        "stratum_weights": weights,
        "observed_strata": observed_strata,
        "sessions": {
            "screen": screen_sessions,
            "holdout": holdout_sessions,
            "disjoint": True,
        },
        "adequacy": {
            "screen_independent_sessions": len(screen_sessions),
            "screen_planned_pairs": screen_pair_count,
            "holdout_independent_sessions": len(holdout_sessions),
            "holdout_planned_pairs": holdout_pair_count,
            "minimum_screen_sessions": MIN_SCREEN_SESSIONS,
            "minimum_holdout_sessions": MIN_HOLDOUT_SESSIONS,
            "minimum_screen_pairs": MIN_SCREEN_PAIRS,
            "status": "planned" if adequate else "inconclusive_insufficient_corpus",
        },
        "splits": selected_by_split,
        "session_replay": session_replay,
        "coverage": coverage,
    }
    return {**core, "plan_id": canonical_hash(core)}


def _load_manifest(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorpusError(f"{path}:{line_number}: invalid JSON") from exc
                rows.append(value)
    except OSError as exc:
        raise CorpusError(f"cannot read {path}: {exc.strerror or exc}") from exc
    return rows


def _write_json(path: Path, value: JsonObject) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.partial")
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    except OSError as exc:
        raise CorpusError(f"cannot write {path}: {exc.strerror or exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="make a deterministic session-stratified screen/holdout plan")
    plan.add_argument("--manifest", required=True, type=Path, help="sanitized capture descriptor JSONL")
    plan.add_argument("--seed", required=True, type=int)
    plan.add_argument(
        "--output-budget",
        type=int,
        help="optional cohort filter for one captured output-token budget; omitted keeps each request's own budget",
    )
    plan.add_argument("--request-budget", type=int, default=64, help="maximum measured capsules across both splits")
    plan.add_argument("--samples-per-session", type=int, default=2)
    plan.add_argument("--cache-recipe-id", required=True)
    plan.add_argument("--cache-state", choices=("unknown", "cold", "warm"), default="unknown")
    plan.add_argument("--warmup-requests", type=int, default=0, help="unmeasured warmups per arm")
    plan.add_argument("--holdout-fraction", type=float, default=0.25)
    plan.add_argument("--stratum-weights-json", type=Path, help="JSON object mapping every stratum to its production weight")
    plan.add_argument("--strata", help="comma-separated strata to plan; omitted plans every observed stratum")
    plan.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        weights = None
        if args.stratum_weights_json is not None:
            value = json.loads(args.stratum_weights_json.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise CorpusError("stratum weights file must contain a JSON object")
            weights = value
        plan = make_plan(
            _load_manifest(args.manifest),
            seed=args.seed,
            samples_per_session=args.samples_per_session,
            output_budget=args.output_budget,
            request_budget=args.request_budget,
            cache_recipe_id=args.cache_recipe_id,
            cache_state=args.cache_state,
            warmup_requests=args.warmup_requests,
            holdout_fraction=args.holdout_fraction,
            stratum_weights=weights,
            strata=args.strata.split(",") if args.strata else None,
        )
        _write_json(args.output, plan)
    except (CorpusError, OSError, json.JSONDecodeError) as exc:
        print(f"corpus: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"corpus_id": plan["corpus_id"], "plan_id": plan["plan_id"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
