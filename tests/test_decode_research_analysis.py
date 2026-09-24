#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ANALYZER_PATH = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location("decode_research_analyzer", ANALYZER_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZER)


def event(
    *,
    engine_round_id,
    request_id,
    request_round_index,
    batch_size,
    round_elapsed_ns,
    context_frontier,
    draft_extent,
    accepted_prefix,
    licensed_count,
    committed_count,
    terminal=False,
    truncated=False,
    cancelled=False,
    finish_reason="none",
):
    return {
        "schema": ANALYZER.SCHEMA,
        "engine_session_id": 42,
        "engine_round_id": engine_round_id,
        "request_id": request_id,
        "request_round_index": request_round_index,
        "batch_size": batch_size,
        "backend": "mtp",
        "draft_window": 3,
        "cuda_graph": True,
        "context_frontier": context_frontier,
        "draft_extent": draft_extent,
        "accepted_prefix": accepted_prefix,
        "committed_draft_prefix": min(accepted_prefix, committed_count),
        "licensed_count": licensed_count,
        "committed_count": committed_count,
        "round_elapsed_ns": round_elapsed_ns,
        "terminal": terminal,
        "truncated": truncated,
        "cancelled": cancelled,
        "finish_reason": finish_reason,
    }


class DecodeResearchAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            event(
                engine_round_id=1,
                request_id=1,
                request_round_index=1,
                batch_size=2,
                round_elapsed_ns=1_000_000,
                context_frontier=1024,
                draft_extent=2,
                accepted_prefix=2,
                licensed_count=3,
                committed_count=3,
            ),
            event(
                engine_round_id=1,
                request_id=2,
                request_round_index=1,
                batch_size=2,
                round_elapsed_ns=1_000_000,
                context_frontier=2048,
                draft_extent=2,
                accepted_prefix=0,
                licensed_count=1,
                committed_count=1,
            ),
            event(
                engine_round_id=2,
                request_id=1,
                request_round_index=2,
                batch_size=1,
                round_elapsed_ns=3_000_000,
                context_frontier=1027,
                draft_extent=1,
                accepted_prefix=1,
                licensed_count=2,
                committed_count=1,
                terminal=True,
                truncated=True,
                finish_reason="output_limit",
            ),
        ]

    def test_ratio_of_sums_and_marginal_survival(self):
        summary = ANALYZER.summarize(self.rows)["groups"][0]
        # Batch rows share one elapsed engine-round cost. A naive row-time sum gives 1000 tok/s;
        # averaging the three row rates gives about 1444 tok/s. The correct ratio is 5 / .004.
        self.assertAlmostEqual(summary["committed_tokens_per_second"], 1250.0)
        self.assertAlmostEqual(summary["round_elapsed_seconds"], 0.004)
        self.assertEqual(summary["request_rounds"], 3)
        self.assertEqual(summary["engine_rounds"], 2)
        self.assertAlmostEqual(summary["weighted_draft_acceptance"], 0.6)
        self.assertEqual(summary["terminal_request_rounds"], 1)
        self.assertEqual(summary["truncated_request_rounds"], 1)
        self.assertEqual(
            summary["context_frontier"],
            {"min": 1024, "mean": 4099 / 3, "max": 2048},
        )
        self.assertEqual(
            summary["marginal_width"],
            [
                {
                    "position": 1,
                    "eligible_request_rounds": 3,
                    "accepted_request_rounds": 2,
                    "survival": 2 / 3,
                    "draft_work_per_request_round": 1.0,
                    "accepted_per_request_round": 2 / 3,
                    "rejected_drafts_per_request_round": 1 / 3,
                },
                {
                    "position": 2,
                    "eligible_request_rounds": 2,
                    "accepted_request_rounds": 1,
                    "survival": 0.5,
                    "draft_work_per_request_round": 2 / 3,
                    "accepted_per_request_round": 1 / 3,
                    "rejected_drafts_per_request_round": 1 / 3,
                },
            ],
        )

    def test_jsonl_parser_and_incomplete_batch_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
            self.assertEqual(len(ANALYZER.load_events(path)), 3)
            path.write_text(json.dumps(self.rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 2"):
                ANALYZER.load_events(path)

    def test_rejects_inconsistent_commit_and_terminal_state(self):
        broken_prefix = dict(self.rows[0], committed_draft_prefix=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(broken_prefix) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "committed draft prefix"):
                ANALYZER.load_events(path)

    def test_target_only_k0_has_no_speculative_survival_curve(self):
        ordinary = dict(self.rows[0])
        ordinary.update(
            {
                "batch_size": 1,
                "backend": "none",
                "draft_window": 0,
                "draft_extent": 0,
                "accepted_prefix": 0,
                "committed_draft_prefix": 0,
                "licensed_count": 1,
                "committed_count": 1,
                "engine_round_id": 1,
                "request_id": 1,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "k0.jsonl"
            path.write_text(json.dumps(ordinary) + "\n", encoding="utf-8")
            loaded = ANALYZER.load_events(path)
        summary = ANALYZER.summarize(loaded)["groups"][0]
        self.assertEqual(summary["committed_tokens"], 1)
        self.assertIsNone(summary["weighted_draft_acceptance"])
        self.assertEqual(summary["marginal_width"], [])


if __name__ == "__main__":
    unittest.main()
