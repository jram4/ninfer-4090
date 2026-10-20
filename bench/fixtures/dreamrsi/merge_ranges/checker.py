"""Independent behavioral checker for the public merge_ranges fixture."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any


FIXED_CASES = [
    ("empty", [], []),
    ("singleton", [(4, 4)], [(4, 4)]),
    ("unsorted-with-gaps", [(10, 12), (1, 2), (6, 8)], [(1, 2), (6, 8), (10, 12)]),
    ("overlap", [(1, 4), (3, 7)], [(1, 7)]),
    ("adjacent", [(1, 2), (3, 5)], [(1, 5)]),
    ("nested", [(1, 9), (3, 4), (5, 8)], [(1, 9)]),
    ("negative", [(-8, -5), (-4, -1), (2, 3)], [(-8, -1), (2, 3)]),
    ("preserve-gap", [(0, 1), (3, 4)], [(0, 1), (3, 4)]),
    ("list-input", [[7, 9], [1, 2], [3, 6]], [(1, 9)]),
]


def _load_function(candidate: Path):
    spec = importlib.util.spec_from_file_location("fixture_merge_ranges_candidate", candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate at {candidate}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "merge_ranges", None)
    if not callable(function):
        raise TypeError("candidate must define callable merge_ranges(ranges)")
    return function


def _coverage(intervals: list[tuple[int, int]]) -> set[int]:
    return {point for start, end in intervals for point in range(start, end + 1)}


def _validate_output(result: Any, original: Any) -> str | None:
    if type(result) is not list:
        return "result is not a list"
    if result is original:
        return "result reuses the input list"
    previous_end = None
    for index, interval in enumerate(result):
        if type(interval) is not tuple or len(interval) != 2:
            return f"result item {index} is not a 2-tuple"
        start, end = interval
        if type(start) is not int or type(end) is not int:
            return f"result item {index} does not contain exact ints"
        if start > end:
            return f"result item {index} has start greater than end"
        if previous_end is not None and start <= previous_end + 1:
            return "result is not sorted and fully merged"
        previous_end = end
    return None


def _validate_case_result(
    name: str,
    result: Any,
    original: Any,
    before: Any,
    expected: list[tuple[int, int]] | None,
    invocation: str,
) -> str | None:
    prefix = f"{name}: {invocation}"
    problem = _validate_output(result, original)
    if problem:
        return f"{prefix}: {problem}"
    if original != before:
        return f"{prefix}: mutated its input"

    if expected is not None and result != expected:
        return f"{prefix}: got {result!r}, expected {expected!r}"

    input_intervals = [(int(pair[0]), int(pair[1])) for pair in before]
    if input_intervals:
        lower = min(start for start, _ in input_intervals)
        upper = max(end for _, end in input_intervals)
        if any(start < lower or end > upper for start, end in result):
            return f"{prefix}: result extends beyond input bounds"
    elif result:
        return f"{prefix}: nonempty result for empty input"

    if _coverage(result) != _coverage(input_intervals):
        return f"{prefix}: result changes covered integer points"
    return None


def _check_one(function, name: str, supplied: Any, expected: list[tuple[int, int]] | None):
    before = copy.deepcopy(supplied)
    try:
        result = function(supplied)
    except Exception as exc:  # A candidate exception is a failed behavioral check.
        return f"{name}: raised {type(exc).__name__}: {exc}"
    problem = _validate_case_result(name, result, supplied, before, expected, "first invocation")
    if problem:
        return problem

    # A second invocation checks that each call returns a new top-level list.
    second_input = copy.deepcopy(before)
    try:
        second_result = function(second_input)
    except Exception as exc:
        return f"{name}: second invocation raised {type(exc).__name__}: {exc}"
    problem = _validate_case_result(name, second_result, second_input, before, expected, "second invocation")
    if problem:
        return problem
    if second_result is result:
        return f"{name}: reuses a result list across invocations"
    return None


def check_candidate(candidate: str | Path) -> dict[str, Any]:
    """Check fixed examples and every valid interval list of length at most 3."""
    candidate_path = Path(candidate)
    failures: list[str] = []
    checked = 0
    try:
        function = _load_function(candidate_path)
    except Exception as exc:
        return {
            "ok": False,
            "checked": 0,
            "failures": [f"load candidate: {type(exc).__name__}: {exc}"],
            "failure_count": 1,
        }

    def record(name: str, intervals: Any, expected=None):
        nonlocal checked
        checked += 1
        problem = _check_one(function, name, intervals, expected)
        if problem:
            failures.append(problem)

    for name, intervals, expected in FIXED_CASES:
        record(f"fixed/{name}", copy.deepcopy(intervals), expected)

    valid_intervals = [(start, end) for start in range(-2, 3) for end in range(start, 3)]
    for length in range(4):
        for index, sequence in enumerate(itertools.product(valid_intervals, repeat=length)):
            record(f"exhaustive/length-{length}/case-{index}", list(sequence))

    return {
        "ok": not failures,
        "checked": checked,
        "failure_count": len(failures),
        "failures": failures[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    report = check_candidate(args.candidate)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
