#!/usr/bin/env python3
"""Reduce paired fixed-work replay measurements for an early performance screen.

This module evaluates request-replay performance only. It does not test task
quality, establish coding quality, qualify a candidate, or authorize release.
Statistical resampling clusters by independent session, and insufficient data
always remains inconclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MIN_COMPLETE_PAIRS = 12
MIN_INDEPENDENT_SESSIONS = 8
DEFAULT_BOOTSTRAP_RESAMPLES = 4000
IDENTITY_SCOPES = {"exact_body_no_credentials", "redacted_no_body_identity"}
_SAFE_IDENTITY_VALUE = re.compile(r"^[A-Za-z0-9_.:/+-]{1,256}$")
_SAFE_DEVICE_NAME = re.compile(r"^[A-Za-z0-9 ._()+/-]{1,128}$")
_SAFE_CAPSULE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
JsonObject = dict[str, Any]


class QuickEvalError(ValueError):
    """Raised when a plan or per-arm result violates the quick-eval contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise QuickEvalError(f"{name} must be a nonempty string of at most {max_length} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise QuickEvalError(f"{name} must not contain control characters")
    return value.strip()


def _number(value: Any, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuickEvalError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise QuickEvalError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise QuickEvalError(f"{name} must be a finite nonnegative number")
    return result


def _count(value: Any, name: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QuickEvalError(f"{name} must be a nonnegative integer")
    return value


def _object(value: Any, name: str, *, required: bool = True) -> JsonObject | None:
    if value is None and not required:
        return None
    if not isinstance(value, dict):
        raise QuickEvalError(f"{name} must be an object")
    return value


def _validate_plan(plan: JsonObject, split: str) -> list[JsonObject]:
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA_VERSION:
        raise QuickEvalError("plan schema_version must be 1")
    plan_id = plan.get("plan_id")
    core = {key: value for key, value in plan.items() if key != "plan_id"}
    if not isinstance(plan_id, str) or canonical_hash(core) != plan_id:
        raise QuickEvalError("plan_id does not match plan contents")
    if not isinstance(split, str) or split not in {"screen", "holdout"}:
        raise QuickEvalError("split must be screen or holdout")
    corpus_id = _text(plan.get("corpus_id"), "plan corpus_id")
    if len(corpus_id) != 64 or any(char not in "0123456789abcdef" for char in corpus_id):
        raise QuickEvalError("plan corpus_id must be a SHA-256")
    budgets = _object(plan.get("budgets"), "plan budgets")
    cache = _object(plan.get("cache"), "plan cache")
    assert budgets is not None and cache is not None
    output_budget = _count(budgets.get("output_tokens_per_request"), "plan output budget", optional=True)
    request_budget = _count(budgets.get("measured_requests_total"), "plan request budget")
    preserved = budgets.get("per_request_budget_preserved")
    if preserved is None:
        preserved = output_budget is None
    if type(preserved) is not bool:
        raise QuickEvalError("plan per_request_budget_preserved must be boolean")
    if preserved:
        # Each captured request keeps the output budget it was recorded with.
        if output_budget is not None:
            raise QuickEvalError("plan cannot both preserve per-request budgets and fix one output budget")
    elif output_budget is None or output_budget <= 0:
        raise QuickEvalError("plan output budget must be positive")
    if request_budget is None or request_budget <= 0:
        raise QuickEvalError("plan request budget must be positive")
    _text(cache.get("cache_recipe_id"), "plan cache_recipe_id")
    if not isinstance(cache.get("cache_state"), str) or cache.get("cache_state") not in {"unknown", "cold", "warm"}:
        raise QuickEvalError("plan cache_state is invalid")
    warmups = _count(cache.get("warmup_requests_per_arm"), "plan warmup count")
    if warmups is None or budgets.get("warmup_requests_per_arm") != warmups:
        raise QuickEvalError("plan warmup budget is inconsistent")
    splits = _object(plan.get("splits"), "plan splits")
    sessions = _object(plan.get("sessions"), "plan sessions")
    assert splits is not None and sessions is not None
    screen_sessions = sessions.get("screen")
    holdout_sessions = sessions.get("holdout")
    if not isinstance(screen_sessions, list) or not isinstance(holdout_sessions, list):
        raise QuickEvalError("plan session groups must be arrays")
    screen_sessions = [_text(session, "screen session_id") for session in screen_sessions]
    holdout_sessions = [_text(session, "holdout session_id") for session in holdout_sessions]
    if len(set(screen_sessions)) != len(screen_sessions) or len(set(holdout_sessions)) != len(holdout_sessions):
        raise QuickEvalError("plan session groups contain duplicates")
    if set(screen_sessions) & set(holdout_sessions):
        raise QuickEvalError("screen and holdout sessions overlap")
    if sessions.get("disjoint") is not True:
        raise QuickEvalError("plan does not assert disjoint session groups")
    all_rows: list[JsonObject] = []
    all_capsule_ids: set[str] = set()
    traffic_classes: set[str] = set()
    for split_name, group_sessions in (("screen", screen_sessions), ("holdout", holdout_sessions)):
        group_rows = splits.get(split_name)
        if not isinstance(group_rows, list):
            raise QuickEvalError(f"plan splits.{split_name} must be an array")
        group_sessions_set = set(group_sessions)
        previous_by_session: dict[str, int] = {}
        for item in group_rows:
            if not isinstance(item, dict):
                raise QuickEvalError(f"plan splits.{split_name} entries must be objects")
            capsule_id = _text(item.get("capsule_id"), "plan capsule_id")
            session_id = _text(item.get("session_id"), f"{capsule_id} session_id")
            sequence = _count(item.get("sequence"), f"{capsule_id} sequence")
            if not _SAFE_CAPSULE_ID.fullmatch(capsule_id) or sequence is None or sequence <= 0:
                raise QuickEvalError(f"plan capsule {capsule_id} has an invalid identity or sequence")
            if capsule_id in all_capsule_ids:
                raise QuickEvalError(f"plan repeats capsule_id {capsule_id} across splits")
            if session_id not in group_sessions_set:
                raise QuickEvalError(f"capsule {capsule_id} is outside its {split_name} session group")
            capsule_budget = _count(item.get("output_budget"), f"{capsule_id} output_budget")
            if capsule_budget is None or capsule_budget <= 0:
                raise QuickEvalError(f"capsule {capsule_id} output budget is invalid")
            if not preserved and capsule_budget != output_budget:
                raise QuickEvalError(f"capsule {capsule_id} output budget differs from plan")
            traffic_class = item.get("traffic_class")
            if not isinstance(traffic_class, str) or traffic_class not in {
                "real_user",
                "agent_generated",
                "derived",
            }:
                raise QuickEvalError(f"capsule {capsule_id} traffic_class is invalid")
            traffic_classes.add(traffic_class)
            previous_sequence = previous_by_session.get(session_id)
            if previous_sequence is not None and sequence <= previous_sequence:
                raise QuickEvalError(f"plan does not preserve capture order within session {session_id}")
            previous_by_session[session_id] = sequence
            all_capsule_ids.add(capsule_id)
            all_rows.append(item)
    if len(traffic_classes) > 1:
        raise QuickEvalError("one plan cannot mix traffic classes")
    selected_count = _count(budgets.get("selected_requests_total"), "plan selected request count")
    if selected_count != len(all_rows) or len(all_rows) > request_budget:
        raise QuickEvalError("plan selected requests do not match its declared total request budget")
    rows = splits[split]
    seen: set[str] = set()
    selected: list[JsonObject] = []
    allowed_sessions = set(screen_sessions if split == "screen" else holdout_sessions)
    for item in rows:
        if not isinstance(item, dict):
            raise QuickEvalError(f"plan splits.{split} entries must be objects")
        capsule_id = _text(item.get("capsule_id"), "plan capsule_id")
        if not _SAFE_CAPSULE_ID.fullmatch(capsule_id):
            raise QuickEvalError("plan capsule_id contains unsafe characters")
        session_id = _text(item.get("session_id"), f"{capsule_id} session_id")
        sequence = _count(item.get("sequence"), f"{capsule_id} sequence")
        assert sequence is not None
        if sequence <= 0:
            raise QuickEvalError(f"capsule {capsule_id} sequence must be positive")
        if capsule_id in seen:
            raise QuickEvalError(f"plan repeats capsule_id {capsule_id}")
        if session_id not in allowed_sessions:
            raise QuickEvalError(f"capsule {capsule_id} is outside the plan's {split} session group")
        capsule_budget = _count(item.get("output_budget"), f"{capsule_id} output_budget")
        if capsule_budget is None or capsule_budget <= 0:
            raise QuickEvalError(f"capsule {capsule_id} output budget is invalid")
        if not preserved and capsule_budget != output_budget:
            raise QuickEvalError(f"capsule {capsule_id} output budget differs from plan")
        scope = item.get("identity_scope")
        if not isinstance(scope, str) or scope not in IDENTITY_SCOPES:
            raise QuickEvalError(f"capsule {capsule_id} identity_scope is invalid")
        request_hash = item.get("request_sha256")
        if scope == "exact_body_no_credentials":
            if not isinstance(request_hash, str) or len(request_hash) != 64:
                raise QuickEvalError(f"capsule {capsule_id} exact body hash is missing")
            if any(char not in "0123456789abcdef" for char in request_hash):
                raise QuickEvalError(f"capsule {capsule_id} request hash is invalid")
        elif request_hash is not None:
            raise QuickEvalError(f"capsule {capsule_id} redacted identity must not retain request hash")
        capsule_hash = item.get("capsule_sha256")
        if (
            not isinstance(capsule_hash, str)
            or len(capsule_hash) != 64
            or any(char not in "0123456789abcdef" for char in capsule_hash)
        ):
            raise QuickEvalError(f"capsule {capsule_id} capsule hash is invalid")
        _text(item.get("stratum"), f"{capsule_id} stratum")
        _text(item.get("task_label"), f"{capsule_id} task_label")
        _text(item.get("run_label"), f"{capsule_id} run_label")
        if not isinstance(item.get("traffic_class"), str) or item.get("traffic_class") not in {
            "real_user",
            "agent_generated",
            "derived",
        }:
            raise QuickEvalError(f"capsule {capsule_id} traffic_class is invalid")
        labels = item.get("labels")
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise QuickEvalError(f"capsule {capsule_id} labels are invalid")
        for label in labels:
            _text(label, f"{capsule_id} label", max_length=200)
        seen.add(capsule_id)
        selected.append(item)
    if len(selected) > request_budget:
        raise QuickEvalError("plan selected requests exceed its declared request budget")
    weights = plan.get("stratum_weights")
    if weights is not None:
        if not isinstance(weights, dict) or not weights:
            raise QuickEvalError("plan stratum_weights must be a nonempty object or null")
        for stratum, weight in weights.items():
            _text(stratum, "plan stratum weight key")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or weight <= 0:
                raise QuickEvalError("plan stratum weights must be finite positive numbers")
    return selected


def _safe_runtime_identity(value: Any, name: str) -> JsonObject:
    identity = _object(value, name)
    assert identity is not None
    allowed = {
        "artifact_sha256",
        "model_id",
        "runtime_version",
        "source_commit",
        "precision",
        "capacity",
        "device_name",
    }
    if not identity or set(identity) - allowed:
        raise QuickEvalError(f"{name} must contain safe runtime identity fields only")
    clean: JsonObject = {}
    for key, item in identity.items():
        if isinstance(item, str):
            clean[key] = _text(item, f"{name}.{key}", max_length=256)
            pattern = _SAFE_DEVICE_NAME if key == "device_name" else _SAFE_IDENTITY_VALUE
            if not pattern.fullmatch(clean[key]):
                raise QuickEvalError(f"{name}.{key} contains unsafe identity text")
        elif isinstance(item, bool):
            clean[key] = item
        elif isinstance(item, int) and item >= 0:
            clean[key] = item
        else:
            raise QuickEvalError(f"{name}.{key} must be a safe string, boolean, or nonnegative integer")
    if "artifact_sha256" in clean:
        digest = clean["artifact_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise QuickEvalError(f"{name}.artifact_sha256 must be a SHA-256")
        clean["artifact_sha256"] = digest.lower()
    return clean


def _safe_output_work(value: Any, name: str) -> JsonObject:
    if value is None:
        return {}
    work = _object(value, name)
    assert work is not None
    cleaned: JsonObject = {}
    for key, item in work.items():
        if not isinstance(key, str) or not key or len(key) > 64 or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in key):
            raise QuickEvalError(f"{name} contains an invalid counter name")
        count = _count(item, f"{name}.{key}")
        assert count is not None
        cleaned[key] = count
    return cleaned


def _session_warmup(plan: JsonObject, session_id: str) -> int | None:
    """Return the planned warmup count for one session, if the plan declares it."""
    replay = plan.get("session_replay")
    if not isinstance(replay, dict):
        return None
    for split in ("screen", "holdout"):
        sessions = replay.get(split)
        if not isinstance(sessions, list):
            continue
        for session in sessions:
            if not isinstance(session, dict) or session.get("session_id") != session_id:
                continue
            count = _count(session.get("warmup_requests"), "session warmup count", optional=True)
            return count
    return None


def _sanitize_result(row: JsonObject, arm: str, selected: JsonObject, plan: JsonObject, split: str) -> JsonObject:
    capsule_id = selected["capsule_id"]
    if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
        raise QuickEvalError(f"{arm} {capsule_id}: result schema_version must be 1")
    if row.get("plan_id") != plan["plan_id"] or row.get("corpus_id") != plan["corpus_id"]:
        raise QuickEvalError(f"{arm} {capsule_id}: result plan/corpus identity differs")
    if row.get("split") != split or row.get("capsule_id") != capsule_id:
        raise QuickEvalError(f"{arm} result order/ID/split mismatch for {capsule_id}")
    if row.get("session_id") != selected["session_id"]:
        raise QuickEvalError(f"{arm} {capsule_id}: session_id differs from the plan")
    if row.get("identity_scope") != selected["identity_scope"]:
        raise QuickEvalError(f"{arm} {capsule_id}: identity_scope differs from the plan")
    if row.get("capsule_sha256") != selected["capsule_sha256"]:
        raise QuickEvalError(f"{arm} {capsule_id}: capsule hash differs from the plan")
    if row.get("request_sha256") != selected.get("request_sha256"):
        raise QuickEvalError(f"{arm} {capsule_id}: request identity differs from the plan")
    budgets = plan["budgets"]
    cache = plan["cache"]
    preserved = bool(budgets.get("per_request_budget_preserved"))
    if preserved:
        if row.get("output_budget") != selected.get("output_budget"):
            raise QuickEvalError(f"{arm} {capsule_id}: output budget differs from the captured request")
    elif row.get("output_budget") != budgets["output_tokens_per_request"]:
        raise QuickEvalError(f"{arm} {capsule_id}: output budget differs from the plan")
    if row.get("cache_recipe_id") != cache["cache_recipe_id"]:
        raise QuickEvalError(f"{arm} {capsule_id}: cache recipe differs from the plan")
    if row.get("cache_state") != cache["cache_state"]:
        raise QuickEvalError(f"{arm} {capsule_id}: cache state differs from the plan")
    expected_warmup = _session_warmup(plan, selected["session_id"])
    if expected_warmup is not None:
        if row.get("warmup_requests") != expected_warmup:
            raise QuickEvalError(f"{arm} {capsule_id}: warmup count differs from the session replay plan")
    elif row.get("warmup_requests") != cache["warmup_requests_per_arm"]:
        raise QuickEvalError(f"{arm} {capsule_id}: warmup count differs from the plan")

    complete = row.get("complete")
    error = row.get("error")
    truncated = row.get("truncated")
    if not isinstance(complete, bool) or not isinstance(error, bool) or not isinstance(truncated, bool):
        raise QuickEvalError(f"{arm} {capsule_id}: complete/error/truncated must be booleans")
    finish_reason = row.get("finish_reason")
    if finish_reason is not None:
        finish_reason = _text(finish_reason, f"{arm} {capsule_id} finish_reason", max_length=64)
        if any(not (char.isalnum() or char in "._-") for char in finish_reason):
            raise QuickEvalError(f"{arm} {capsule_id}: finish_reason contains unsafe characters")
    return {
        "capsule_id": capsule_id,
        "session_id": selected["session_id"],
        "stratum": selected["stratum"],
        "complete": complete,
        "error": error,
        "truncated": truncated,
        "finish_reason": finish_reason,
        "committed_tokens": _count(row.get("committed_tokens"), f"{arm} {capsule_id} committed_tokens", optional=True),
        "ttft_seconds": _number(row.get("ttft_seconds"), f"{arm} {capsule_id} ttft_seconds", optional=True),
        "wall_seconds": _number(row.get("wall_seconds"), f"{arm} {capsule_id} wall_seconds", optional=True),
        "output_work": _safe_output_work(row.get("output_work"), f"{arm} {capsule_id} output_work"),
        "runtime_identity": _safe_runtime_identity(row.get("runtime_identity"), f"{arm} {capsule_id} runtime_identity"),
    }


def _read_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QuickEvalError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise QuickEvalError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise QuickEvalError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QuickEvalError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise QuickEvalError(f"{path}:{line_number}: expected a JSON object")
                rows.append(row)
    except OSError as exc:
        raise QuickEvalError(f"cannot read {path}: {exc.strerror or exc}") from exc
    return rows


def _index_results(rows: list[JsonObject], arm: str, selected_ids: set[str]) -> dict[str, JsonObject]:
    indexed: dict[str, JsonObject] = {}
    for row in rows:
        capsule_id = row.get("capsule_id")
        if not isinstance(capsule_id, str) or capsule_id not in selected_ids:
            raise QuickEvalError(f"{arm} result contains an unplanned capsule_id")
        if capsule_id in indexed:
            raise QuickEvalError(f"{arm} result repeats capsule_id {capsule_id}")
        indexed[capsule_id] = row
    return indexed


def _numeric_summary(values: Any) -> JsonObject:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers) if numbers else None,
        "median": statistics.median(numbers) if numbers else None,
    }


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    maximum = max(float(value) for value in weights.values())
    scaled = {name: float(value) / maximum for name, value in weights.items()}
    total = math.fsum(scaled.values())
    return {name: value / total for name, value in scaled.items()}


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    try:
        result = numerator / denominator
    except OverflowError:
        return None
    return result if math.isfinite(result) and result > 0 else None


def _committed_rate_ratio(
    candidate_tokens: int,
    candidate_wall: float,
    baseline_tokens: int,
    baseline_wall: float,
) -> float | None:
    try:
        log_ratio = (
            math.log(candidate_tokens)
            - math.log(candidate_wall)
            - math.log(baseline_tokens)
            + math.log(baseline_wall)
        )
        result = math.exp(log_ratio)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _bootstrap_summary(
    pairs: list[JsonObject],
    metric: str,
    *,
    weights: dict[str, float] | None,
    seed: int,
    resamples: int,
) -> JsonObject:
    by_session: dict[str, dict[str, list[float]]] = {}
    for pair in pairs:
        ratio = pair.get(metric)
        if not isinstance(ratio, (float, int)) or ratio <= 0 or not math.isfinite(float(ratio)):
            continue
        session = pair["session_id"]
        stratum = pair["stratum"]
        by_session.setdefault(stratum, {}).setdefault(session, []).append(math.log(float(ratio)))
    session_means = {
        stratum: {session: statistics.fmean(logs) for session, logs in sessions.items()}
        for stratum, sessions in by_session.items()
    }
    session_count = sum(len(sessions) for sessions in session_means.values())
    pair_count = sum(len(logs) for sessions in by_session.values() for logs in sessions.values())
    if weights is None:
        values = [value for sessions in session_means.values() for value in sessions.values()]
        point_log = statistics.fmean(values) if values else None
    else:
        missing = sorted(stratum for stratum in weights if len(session_means.get(stratum, {})) < 2)
        available = {stratum: sessions for stratum, sessions in session_means.items() if stratum in weights}
        if missing:
            return {
                "pair_count": pair_count,
                "independent_sessions": session_count,
                "geometric_mean_ratio": None,
                "bootstrap_95_ci": None,
                "inadequate_weighted_strata": missing,
            }
        normalized_weights = _normalized_weights(weights)
        point_log = math.fsum(
            normalized_weights[stratum] * statistics.fmean(available[stratum].values())
            for stratum in weights
        )
    if point_log is None:
        return {
            "pair_count": pair_count,
            "independent_sessions": session_count,
            "geometric_mean_ratio": None,
            "bootstrap_95_ci": None,
            "inadequate_weighted_strata": [],
        }
    ci: list[float] | None = None
    if resamples > 0 and session_count >= 2:
        rng = random.Random(seed)
        draws: list[float] = []
        if weights is None:
            values = [value for sessions in session_means.values() for value in sessions.values()]
            for _ in range(resamples):
                draws.append(statistics.fmean(rng.choice(values) for _ in values))
        else:
            normalized_weights = _normalized_weights(weights)
            for _ in range(resamples):
                weighted_log = 0.0
                for stratum, weight in weights.items():
                    values = list(session_means[stratum].values())
                    sample = [rng.choice(values) for _ in values]
                    weighted_log += normalized_weights[stratum] * statistics.fmean(sample)
                draws.append(weighted_log)
        ordered = sorted(draws)
        lower = ordered[max(0, math.ceil(0.025 * len(ordered)) - 1)]
        upper = ordered[min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)]
        ci = [math.exp(lower), math.exp(upper)]
    return {
        "pair_count": pair_count,
        "independent_sessions": session_count,
        "geometric_mean_ratio": math.exp(point_log),
        "bootstrap_95_ci": ci,
        "session_log_means_by_stratum": {
            stratum: {session: value for session, value in sorted(sessions.items())}
            for stratum, sessions in sorted(session_means.items())
        },
    }


def _classify(
    rates: JsonObject,
    ttft: JsonObject,
    wall: JsonObject,
    *,
    complete_pairs: int,
    session_count: int,
    blockers: list[str],
    candidate_failures: int,
    minimum_pairs: int = MIN_COMPLETE_PAIRS,
    minimum_sessions: int = MIN_INDEPENDENT_SESSIONS,
) -> tuple[str, str]:
    if complete_pairs < minimum_pairs or session_count < minimum_sessions:
        return (
            "inconclusive",
            f"need at least {minimum_pairs} complete fixed-work pairs across "
            f"{minimum_sessions} independent sessions",
        )
    if candidate_failures:
        return "regression", "candidate has error, incomplete, or truncated responses"
    if blockers:
        return "inconclusive", "configuration, output-work, or finish-reason mismatch blocks a paired win"
    rate_ci = rates.get("bootstrap_95_ci")
    ttft_ci = ttft.get("bootstrap_95_ci")
    wall_ci = wall.get("bootstrap_95_ci")
    if not isinstance(rate_ci, list) or not isinstance(ttft_ci, list) or not isinstance(wall_ci, list):
        return "inconclusive", "one or more paired cluster-bootstrap intervals are unavailable"
    rate_regression = rate_ci[1] < 0.95
    latency_regression = ttft_ci[0] > 1.05 or wall_ci[0] > 1.05
    rate_improvement = rate_ci[0] > 1.05
    latency_improvement = ttft_ci[1] < 0.95 or wall_ci[1] < 0.95
    if rate_regression or latency_regression:
        return "regression", "cluster-bootstrap interval shows a material committed-token-rate or latency regression"
    if rate_improvement or latency_improvement:
        return "promising", "cluster-bootstrap interval clears a 5% performance threshold without a material tradeoff"
    return "flat/noisy", "paired session-cluster intervals do not clear the 5% screen thresholds"


def reduce_pair(
    plan: JsonObject,
    baseline_rows: list[JsonObject],
    candidate_rows: list[JsonObject],
    *,
    split: str = "screen",
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    minimum_pairs: int = MIN_COMPLETE_PAIRS,
    minimum_sessions: int = MIN_INDEPENDENT_SESSIONS,
) -> JsonObject:
    """Reduce matching baseline/candidate results and cluster uncertainty by session.

    ``minimum_pairs`` and ``minimum_sessions`` are the thresholds the trusted
    policy sets for this split. A holdout split is smaller than a screen by
    design, so it carries its own thresholds.
    """
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 0:
        raise QuickEvalError("bootstrap_resamples must be a nonnegative integer")
    selected = _validate_plan(plan, split)
    selected_ids = {row["capsule_id"] for row in selected}
    baseline_by_id = _index_results(baseline_rows, "baseline", selected_ids)
    candidate_by_id = _index_results(candidate_rows, "candidate", selected_ids)
    paired_rows: list[JsonObject] = []
    blocker_ids: list[str] = []
    candidate_failures = 0
    complete_pairs = 0
    candidate_shorter = 0

    for item in selected:
        capsule_id = item["capsule_id"]
        baseline_raw = baseline_by_id.get(capsule_id)
        candidate_raw = candidate_by_id.get(capsule_id)
        reasons: list[str] = []
        if item["identity_scope"] != "exact_body_no_credentials":
            reasons.append("redacted_request_identity")
        if baseline_raw is None or candidate_raw is None:
            reasons.append("missing_arm_result")
            baseline = None
            candidate = None
        else:
            baseline = _sanitize_result(baseline_raw, "baseline", item, plan, split)
            candidate = _sanitize_result(candidate_raw, "candidate", item, plan, split)
            if not baseline["complete"] or baseline["error"] or baseline["truncated"]:
                reasons.append("baseline_incomplete_or_error")
            if not candidate["complete"] or candidate["error"] or candidate["truncated"]:
                reasons.append("candidate_incomplete_or_error")
                candidate_failures += 1
            if baseline["finish_reason"] != candidate["finish_reason"]:
                reasons.append("finish_reason_mismatch")
            if baseline["finish_reason"] is None or candidate["finish_reason"] is None:
                reasons.append("missing_finish_reason")
            if baseline["committed_tokens"] != candidate["committed_tokens"]:
                if (
                    isinstance(candidate["committed_tokens"], int)
                    and isinstance(baseline["committed_tokens"], int)
                    and candidate["committed_tokens"] < baseline["committed_tokens"]
                ):
                    reasons.append("candidate_shorter_output")
                    candidate_shorter += 1
                else:
                    reasons.append("fixed_work_mismatch")
            if baseline["output_work"] != candidate["output_work"]:
                reasons.append("output_work_mismatch")
            if baseline["runtime_identity"] == {} or candidate["runtime_identity"] == {}:
                reasons.append("missing_runtime_identity")
        valid_pair = baseline is not None and candidate is not None
        committed = baseline.get("committed_tokens") if baseline is not None else None
        base_wall = baseline.get("wall_seconds") if baseline is not None else None
        cand_wall = candidate.get("wall_seconds") if candidate is not None else None
        base_ttft = baseline.get("ttft_seconds") if baseline is not None else None
        cand_ttft = candidate.get("ttft_seconds") if candidate is not None else None
        work_match = baseline is not None and candidate is not None and baseline["output_work"] == candidate["output_work"]
        metrics_eligible = (
            valid_pair
            and not reasons
            and committed is not None
            and committed > 0
            and base_wall is not None
            and base_wall > 0
            and cand_wall is not None
            and cand_wall > 0
            and base_ttft is not None
            and base_ttft > 0
            and cand_ttft is not None
            and cand_ttft > 0
            and work_match
        )
        base_tokens = baseline.get("committed_tokens") if baseline else None
        cand_tokens = candidate.get("committed_tokens") if candidate else None
        rate_ratio = None
        if (
            metrics_eligible
            and isinstance(base_tokens, int)
            and isinstance(cand_tokens, int)
            and base_wall
            and cand_wall
        ):
            rate_ratio = _committed_rate_ratio(cand_tokens, cand_wall, base_tokens, base_wall)
        ttft_ratio = _finite_ratio(cand_ttft, base_ttft) if metrics_eligible and base_ttft and cand_ttft else None
        wall_ratio = _finite_ratio(cand_wall, base_wall) if metrics_eligible and base_wall and cand_wall else None
        if metrics_eligible and any(value is None for value in (rate_ratio, ttft_ratio, wall_ratio)):
            reasons.append("nonfinite_metric_ratio")
            metrics_eligible = False
        elif valid_pair and not metrics_eligible and not reasons:
            reasons.append("missing_or_nonpositive_metrics")
        if metrics_eligible:
            complete_pairs += 1
        if reasons:
            blocker_ids.append(capsule_id)
        paired_rows.append(
            {
                "capsule_id": capsule_id,
                "session_id": item["session_id"],
                "stratum": item["stratum"],
                "baseline": baseline,
                "candidate": candidate,
                "candidate_win_eligible": bool(metrics_eligible),
                "ineligibility_reasons": sorted(set(reasons)),
                "committed_token_rate_ratio": rate_ratio,
                "ttft_ratio": ttft_ratio,
                "wall_ratio": wall_ratio,
            }
        )

    # Require a stable runtime identity within each arm, while allowing the two arms to differ.
    runtime_identity_variation: dict[str, list[str]] = {}
    for arm in ("baseline", "candidate"):
        identities = {
            canonical_json(pair[arm]["runtime_identity"])
            for pair in paired_rows
            if isinstance(pair.get(arm), dict)
        }
        if len(identities) > 1:
            runtime_identity_variation[arm] = sorted(identities)
            for pair in paired_rows:
                if pair.get(arm) is not None:
                    pair["candidate_win_eligible"] = False
                    pair["ineligibility_reasons"] = sorted(
                        set(pair["ineligibility_reasons"] + ["runtime_identity_varies_within_arm"])
                    )
                    pair["committed_token_rate_ratio"] = None
                    pair["ttft_ratio"] = None
                    pair["wall_ratio"] = None
                    blocker_ids.append(pair["capsule_id"])
    complete_pairs = sum(1 for pair in paired_rows if pair["candidate_win_eligible"])

    plan_seed = plan.get("seed", 0)
    seed_root = int(hashlib.sha256(f"{plan['plan_id']}:{plan_seed}:{split}".encode()).hexdigest()[:16], 16)
    weights = plan.get("stratum_weights")
    if weights is not None and not isinstance(weights, dict):
        raise QuickEvalError("plan stratum_weights must be an object or null")
    rate_summary = _bootstrap_summary(
        paired_rows,
        "committed_token_rate_ratio",
        weights=weights,
        seed=seed_root ^ 0xA17E,
        resamples=bootstrap_resamples,
    )
    ttft_summary = _bootstrap_summary(
        paired_rows,
        "ttft_ratio",
        weights=weights,
        seed=seed_root ^ 0x77F7,
        resamples=bootstrap_resamples,
    )
    wall_summary = _bootstrap_summary(
        paired_rows,
        "wall_ratio",
        weights=weights,
        seed=seed_root ^ 0xA11,
        resamples=bootstrap_resamples,
    )
    blocker_ids = sorted(set(blocker_ids))
    independent_sessions = len({row["session_id"] for row in paired_rows if row["candidate_win_eligible"]})
    classification, reason = _classify(
        rate_summary,
        ttft_summary,
        wall_summary,
        complete_pairs=complete_pairs,
        session_count=independent_sessions,
        blockers=blocker_ids,
        candidate_failures=candidate_failures,
        minimum_pairs=minimum_pairs,
        minimum_sessions=minimum_sessions,
    )
    if runtime_identity_variation and classification == "promising":
        classification = "inconclusive"
        reason = "runtime identity changed within an arm"
    output_work_totals: dict[str, dict[str, int]] = {"baseline": {}, "candidate": {}}
    arm_summaries: dict[str, JsonObject] = {}
    for arm in output_work_totals:
        rows = [pair.get(arm) for pair in paired_rows if isinstance(pair.get(arm), dict)]
        for pair in paired_rows:
            result = pair.get(arm)
            if not isinstance(result, dict):
                continue
            for name, count in result["output_work"].items():
                output_work_totals[arm][name] = output_work_totals[arm].get(name, 0) + count
        finish_counts: dict[str, int] = {}
        for row in rows:
            finish = row.get("finish_reason")
            if isinstance(finish, str):
                finish_counts[finish] = finish_counts.get(finish, 0) + 1
        arm_summaries[arm] = {
            "results": len(rows),
            "missing_results": len(selected) - len(rows),
            "complete": sum(1 for row in rows if row["complete"]),
            "errors": sum(1 for row in rows if row["error"]),
            "truncated": sum(1 for row in rows if row["truncated"]),
            "committed_tokens_total": sum(row["committed_tokens"] or 0 for row in rows),
            "finish_reason_counts": dict(sorted(finish_counts.items())),
            "ttft_seconds": _numeric_summary(row["ttft_seconds"] for row in rows),
            "wall_seconds": _numeric_summary(row["wall_seconds"] for row in rows),
            "runtime_identities": sorted({canonical_json(row["runtime_identity"]) for row in rows}),
            "output_work_totals": output_work_totals[arm],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "corpus_id": plan["corpus_id"],
        "split": split,
        "scope": "request_replay_performance_screen_only",
        "classification": classification,
        "classification_reason": reason,
        "qualification": "not_qualified",
        "task_quality": "not_evaluated",
        "insufficient_data_is_inconclusive": True,
        "thresholds": {
            "minimum_complete_pairs": minimum_pairs,
            "minimum_independent_sessions": minimum_sessions,
            "material_performance_change": 0.05,
            "interval": "95% percentile bootstrap of session-clustered mean log ratios",
        },
        "paired": {
            "planned_pairs": len(selected),
            "complete_fixed_work_pairs": complete_pairs,
            "independent_sessions_with_eligible_pairs": independent_sessions,
            "candidate_shorter_outputs": candidate_shorter,
            "candidate_error_or_truncated_pairs": candidate_failures,
            "blocked_capsules": blocker_ids,
            "runtime_identity_variation": runtime_identity_variation,
            "committed_token_rate_ratio_candidate_over_baseline": rate_summary,
            "ttft_ratio_candidate_over_baseline": ttft_summary,
            "wall_ratio_candidate_over_baseline": wall_summary,
            "output_work_totals": output_work_totals,
            "arms": arm_summaries,
            "capsules": paired_rows,
        },
        "identity": {
            "cache_recipe_id": plan["cache"]["cache_recipe_id"],
            "cache_state": plan["cache"]["cache_state"],
            "warmup_requests_per_arm": plan["cache"]["warmup_requests_per_arm"],
            "output_budget": plan["budgets"]["output_tokens_per_request"],
            "stratum_weights": weights,
            "selected_capsules": [
                {
                    "capsule_id": item["capsule_id"],
                    "sequence": item["sequence"],
                    "session_id": item["session_id"],
                    "task_label": item["task_label"],
                    "run_label": item["run_label"],
                    "traffic_class": item["traffic_class"],
                    "stratum": item["stratum"],
                    "labels": item["labels"],
                    "identity_scope": item["identity_scope"],
                    "request_sha256": item.get("request_sha256"),
                    "capsule_sha256": item["capsule_sha256"],
                }
                for item in selected
            ],
        },
    }


def _write_json(path: Path, value: JsonObject) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.partial")
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    except OSError as exc:
        raise QuickEvalError(f"cannot write {path}: {exc.strerror or exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reduce = commands.add_parser("reduce", help="reduce paired baseline/candidate replay rows")
    reduce.add_argument("--plan", required=True, type=Path)
    reduce.add_argument("--split", choices=("screen", "holdout"), default="screen")
    reduce.add_argument("--baseline-results", required=True, type=Path, help="sanitized result JSONL")
    reduce.add_argument("--candidate-results", required=True, type=Path, help="sanitized result JSONL")
    reduce.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    reduce.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = reduce_pair(
            _read_json(args.plan),
            _read_jsonl(args.baseline_results),
            _read_jsonl(args.candidate_results),
            split=args.split,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        _write_json(args.output, report)
    except (QuickEvalError, OSError) as exc:
        print(f"quick_eval: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "plan_id": report["plan_id"],
                "split": report["split"],
                "classification": report["classification"],
                "qualification": report["qualification"],
                "report": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
