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
                terminal=True,
                finish_reason="output_limit",
            ),
            event(
                engine_round_id=2,
                request_id=1,
                request_round_index=2,
                batch_size=1,
                round_elapsed_ns=3_000_000,
                context_frontier=1027,
                draft_extent=2,
                accepted_prefix=2,
                licensed_count=3,
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
        self.assertAlmostEqual(summary["weighted_draft_acceptance"], 2 / 3)
        self.assertAlmostEqual(summary["weighted_committed_draft_acceptance"], 0.5)
        self.assertEqual(summary["accepted_draft_tokens"], 4)
        self.assertEqual(summary["committed_accepted_draft_tokens"], 3)
        self.assertEqual(summary["terminal_request_rounds"], 2)
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
                    "committed_accepted_request_rounds": 2,
                    "committed_survival": 2 / 3,
                    "committed_accepted_per_request_round": 2 / 3,
                    "accepted_but_uncommitted_per_request_round": 0.0,
                },
                {
                    "position": 2,
                    "eligible_request_rounds": 3,
                    "accepted_request_rounds": 2,
                    "survival": 2 / 3,
                    "draft_work_per_request_round": 1.0,
                    "accepted_per_request_round": 2 / 3,
                    "rejected_drafts_per_request_round": 1 / 3,
                    "committed_accepted_request_rounds": 1,
                    "committed_survival": 1 / 3,
                    "committed_accepted_per_request_round": 1 / 3,
                    "accepted_but_uncommitted_per_request_round": 1 / 3,
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

            path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.rows[:2]) + '{"schema":',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                ANALYZER.load_events(path)

            duplicate_elapsed = json.dumps(self.rows[0])[:-1] + ',"round_elapsed_ns":1}'
            path.write_text(duplicate_elapsed + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key 'round_elapsed_ns'"):
                ANALYZER.load_events(path)

    def test_rejects_duplicate_rows_and_conflicting_shared_timing(self):
        duplicate = [dict(self.rows[0]), dict(self.rows[0])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-batch.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate request-round identity"):
                ANALYZER.load_events(path)

            conflicting_timing = [dict(row) for row in self.rows[:2]]
            conflicting_timing[1]["round_elapsed_ns"] += 1
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in conflicting_timing),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "shared timing/config differs"):
                ANALYZER.load_events(path)

    def test_rejects_inconsistent_commit_and_terminal_state(self):
        broken_prefix = dict(self.rows[0], committed_draft_prefix=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(broken_prefix) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "committed draft prefix"):
                ANALYZER.load_events(path)

    def test_rejects_out_of_range_width_and_uint64_overflow(self):
        too_wide = dict(self.rows[0], draft_window=1_000_000_000, draft_extent=1_000_000_000)
        with self.assertRaisesRegex(ValueError, "backend maximum"):
            ANALYZER._validate_event(too_wide, Path("wide.jsonl"), 1)

        overflow = dict(self.rows[0], round_elapsed_ns=1 << 64)
        with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
            ANALYZER._validate_event(overflow, Path("overflow.jsonl"), 1)

    def test_rejects_sparse_uint64_round_identity_without_allocating_by_id(self):
        sparse_round = dict(
            self.rows[0],
            engine_round_id=(1 << 64) - 1,
            batch_size=1,
            terminal=True,
            finish_reason="stop_token",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse-round.jsonl"
            path.write_text(json.dumps(sparse_round) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "engine round sequence has a gap"):
                ANALYZER.load_events(path)

    def test_rejects_impossible_cancel_and_nonterminal_truncation(self):
        cancelled_with_commit = dict(
            self.rows[0],
            batch_size=1,
            cancelled=True,
            terminal=True,
            finish_reason="cancelled",
            committed_count=1,
            committed_draft_prefix=1,
        )
        with self.assertRaisesRegex(ValueError, "cancelled rows cannot commit"):
            ANALYZER._validate_event(cancelled_with_commit, Path("cancel.jsonl"), 1)

        nonterminal_truncation = dict(
            self.rows[0],
            batch_size=1,
            terminal=False,
            truncated=True,
            finish_reason="none",
            committed_count=1,
            committed_draft_prefix=1,
        )
        with self.assertRaisesRegex(ValueError, "truncated row must be terminal"):
            ANALYZER._validate_event(nonterminal_truncation, Path("truncated.jsonl"), 1)

    def test_rejects_unfinished_requests_and_rounds_after_terminal(self):
        unfinished = dict(self.rows[0], batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unfinished.jsonl"
            path.write_text(json.dumps(unfinished) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "request sequence is incomplete"):
                ANALYZER.load_events(path)

            first = dict(unfinished, terminal=True, finish_reason="stop_token")
            second = dict(
                unfinished,
                engine_round_id=2,
                request_round_index=2,
                terminal=True,
                finish_reason="stop_token",
            )
            path.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "continues after terminal"):
                ANALYZER.load_events(path)

    def test_rejects_configuration_change_inside_engine_session(self):
        first = dict(self.rows[0], batch_size=1, terminal=True, finish_reason="stop_token")
        second = dict(
            first,
            engine_round_id=2,
            request_id=2,
            request_round_index=1,
            backend="dflash",
            draft_window=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configuration-change.jsonl"
            path.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "changed within session"):
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
                "terminal": True,
                "finish_reason": "output_limit",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "k0.jsonl"
            path.write_text(json.dumps(ordinary) + "\n", encoding="utf-8")
            loaded = ANALYZER.load_events(path)
        summary = ANALYZER.summarize(loaded)["groups"][0]
        self.assertEqual(summary["committed_tokens"], 1)
        self.assertIsNone(summary["weighted_draft_acceptance"])
        self.assertIsNone(summary["weighted_committed_draft_acceptance"])
        self.assertEqual(summary["marginal_width"], [])


if __name__ == "__main__":
    unittest.main()
