#!/usr/bin/env python3
"""Summarize opt-in NInfer per-round decode research JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "ninfer.decode_research_round.v1"
BACKENDS = {"none", "mtp", "dflash", "dflash2"}
FINISH_REASONS = {
    "none",
    "output_limit",
    "context_capacity",
    "stop_token",
    "stop_string",
    "cancelled",
}
INTEGER_FIELDS = {
    "engine_session_id",
    "engine_round_id",
    "request_id",
    "request_round_index",
    "batch_size",
    "draft_window",
    "context_frontier",
    "draft_extent",
    "accepted_prefix",
    "committed_draft_prefix",
    "licensed_count",
    "committed_count",
    "round_elapsed_ns",
}
BOOLEAN_FIELDS = {"cuda_graph", "terminal", "truncated", "cancelled"}


def _validate_event(event: Any, path: Path, line_number: int) -> dict[str, Any]:
    label = f"{path}:{line_number}"
    if not isinstance(event, dict):
        raise ValueError(f"{label}: record must be a JSON object")
    if event.get("schema") != SCHEMA:
        raise ValueError(f"{label}: expected schema {SCHEMA!r}")
    for field in INTEGER_FIELDS:
        value = event.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}: {field} must be a non-negative integer")
    for field in BOOLEAN_FIELDS:
        if not isinstance(event.get(field), bool):
            raise ValueError(f"{label}: {field} must be boolean")
    if event.get("backend") not in BACKENDS:
        raise ValueError(f"{label}: invalid backend")
    if event.get("finish_reason") not in FINISH_REASONS:
        raise ValueError(f"{label}: invalid finish_reason")
    if event["engine_session_id"] == 0 or event["engine_round_id"] == 0:
        raise ValueError(f"{label}: engine session and round identifiers must be positive")
    if event["request_id"] == 0 or event["request_round_index"] == 0:
        raise ValueError(f"{label}: request and request-round identifiers must be positive")
    if event["batch_size"] == 0 or event["round_elapsed_ns"] == 0:
        raise ValueError(f"{label}: batch_size and round_elapsed_ns must be positive")
    if event["draft_extent"] > event["draft_window"]:
        raise ValueError(f"{label}: actual draft extent exceeds configured width")
    if event["accepted_prefix"] > event["draft_extent"]:
        raise ValueError(f"{label}: accepted prefix exceeds actual draft extent")
    if event["committed_count"] > event["licensed_count"]:
        raise ValueError(f"{label}: committed count exceeds licensed count")
    expected_committed_prefix = min(event["accepted_prefix"], event["committed_count"])
    if event["committed_draft_prefix"] != expected_committed_prefix:
        raise ValueError(f"{label}: committed draft prefix is inconsistent")
    expected_truncated = (
        not event["cancelled"] and event["committed_count"] < event["licensed_count"]
    )
    if event["truncated"] != expected_truncated:
        raise ValueError(f"{label}: truncation state is inconsistent")
    if event["terminal"] != (event["finish_reason"] != "none"):
        raise ValueError(f"{label}: terminal state disagrees with finish_reason")
    if event["cancelled"] != (event["finish_reason"] == "cancelled"):
        raise ValueError(f"{label}: cancellation state disagrees with finish_reason")
    if event["backend"] == "none" and event["draft_window"] != 0:
        raise ValueError(f"{label}: ordinary decoding must have draft_window=0")
    if event["backend"] == "none" and (
        event["draft_extent"] != 0
        or event["accepted_prefix"] != 0
        or event["licensed_count"] != 1
    ):
        raise ValueError(f"{label}: ordinary decode row has speculative metadata")
    if event["backend"] != "none" and event["licensed_count"] != event["accepted_prefix"] + 1:
        raise ValueError(f"{label}: speculative licensed count does not match accepted prefix")
    if event["backend"] != "none" and event["draft_window"] == 0:
        raise ValueError(f"{label}: speculative decoding must have a positive draft_window")
    return event


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            events.append(_validate_event(event, path, line_number))
    if not events:
        raise ValueError(f"{path}: no decode research records found")

    seen_request_rounds: set[tuple[int, int, int]] = set()
    engine_rounds: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    request_sequences: dict[tuple[int, int], list[int]] = defaultdict(list)
    for event in events:
        request_round = (
            event["engine_session_id"],
            event["request_id"],
            event["request_round_index"],
        )
        if request_round in seen_request_rounds:
            raise ValueError(f"{path}: duplicate request-round identity {request_round}")
        seen_request_rounds.add(request_round)
        request_sequences[(event["engine_session_id"], event["request_id"])].append(
            event["request_round_index"]
        )
        engine_rounds[(event["engine_session_id"], event["engine_round_id"])].append(event)

    shared_fields = ("batch_size", "round_elapsed_ns", "backend", "draft_window", "cuda_graph")
    for identity, rows in engine_rounds.items():
        first = rows[0]
        if len(rows) != first["batch_size"]:
            raise ValueError(
                f"{path}: engine round {identity} has {len(rows)} rows; "
                f"expected {first['batch_size']}"
            )
        if len({row["request_id"] for row in rows}) != len(rows):
            raise ValueError(f"{path}: engine round {identity} repeats a request row")
        if any(any(row[field] != first[field] for field in shared_fields) for row in rows[1:]):
            raise ValueError(f"{path}: shared timing/config differs within engine round {identity}")
    engine_round_sequences: dict[int, list[int]] = defaultdict(list)
    for session_id, round_id in engine_rounds:
        engine_round_sequences[session_id].append(round_id)
    for session_id, round_ids in engine_round_sequences.items():
        if sorted(round_ids) != list(range(1, max(round_ids) + 1)):
            raise ValueError(f"{path}: engine round sequence has a gap for session {session_id}")
    for identity, round_ids in request_sequences.items():
        if sorted(round_ids) != list(range(1, max(round_ids) + 1)):
            raise ValueError(f"{path}: request round sequence has a gap for identity {identity}")
    return events


def _summarize_group(events: list[dict[str, Any]]) -> dict[str, Any]:
    rounds: dict[tuple[int, int], dict[str, Any]] = {}
    for event in events:
        rounds.setdefault((event["engine_session_id"], event["engine_round_id"]), event)
    round_elapsed_ns = sum(event["round_elapsed_ns"] for event in rounds.values())
    committed = sum(event["committed_count"] for event in events)
    draft_tokens = sum(event["draft_extent"] for event in events)
    accepted = sum(event["accepted_prefix"] for event in events)
    context_values = [event["context_frontier"] for event in events]
    request_round_count = len(events)
    rate = committed * 1_000_000_000 / round_elapsed_ns if round_elapsed_ns else None

    max_width = max((event["draft_extent"] for event in events), default=0)
    marginal_width = []
    for position in range(1, max_width + 1):
        eligible = sum(event["draft_extent"] >= position for event in events)
        survived = sum(event["accepted_prefix"] >= position for event in events)
        marginal_width.append(
            {
                "position": position,
                "eligible_request_rounds": eligible,
                "accepted_request_rounds": survived,
                "survival": survived / eligible if eligible else None,
                "draft_work_per_request_round": eligible / request_round_count,
                "accepted_per_request_round": survived / request_round_count,
                "rejected_drafts_per_request_round": (eligible - survived) / request_round_count,
            }
        )

    return {
        "request_rounds": request_round_count,
        "engine_rounds": len(rounds),
        "committed_tokens": committed,
        "round_elapsed_seconds": round_elapsed_ns / 1_000_000_000,
        "committed_tokens_per_second": rate,
        "weighted_draft_acceptance": accepted / draft_tokens if draft_tokens else None,
        "drafted_tokens": draft_tokens,
        "accepted_draft_tokens": accepted,
        "terminal_request_rounds": sum(event["terminal"] for event in events),
        "truncated_request_rounds": sum(event["truncated"] for event in events),
        "cancelled_request_rounds": sum(event["cancelled"] for event in events),
        "context_frontier": {
            "min": min(context_values),
            "mean": math.fsum(context_values) / request_round_count,
            "max": max(context_values),
        },
        "marginal_width": marginal_width,
    }


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int, bool], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[(event["backend"], event["draft_window"], event["cuda_graph"])].append(event)
    summaries = []
    for (backend, draft_window, cuda_graph), group_events in sorted(groups.items()):
        summaries.append(
            {
                "backend": backend,
                "draft_window": draft_window,
                "cuda_graph": cuda_graph,
                **_summarize_group(group_events),
            }
        )
    return {"schema": "ninfer.decode_research_summary.v1", "groups": summaries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="per-round research JSONL written by ninfer")
    args = parser.parse_args(argv)
    try:
        result = summarize(load_events(args.jsonl))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
