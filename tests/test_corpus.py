from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.dreamrsi.corpus import CorpusError, make_plan, read_descriptors


def _descriptor(
    session: int,
    request: int,
    *,
    stratum: str = "interactive",
    redacted: bool = False,
    output_budget: int = 16,
    traffic_class: str = "real_user",
) -> dict:
    capsule_id = f"s{session:02d}-r{request:02d}"
    return {
        "schema_version": 1,
        "capsule_id": capsule_id,
        "sequence": request + 1,
        "capsule_path": f"/private/{capsule_id}.capsule",
        "capsule_sha256": f"{session + 1:064x}",
        "request_sha256": None if redacted else f"{session * 100 + request + 1:064x}",
        "response_sha256": f"{session * 100 + request + 2:064x}",
        "session_id": f"session-{session:02d}",
        "task_label": "task-family-a",
        "run_label": f"capture-{session:02d}",
        "traffic_class": traffic_class,
        "stratum": stratum,
        "labels": ["stream=true", f"model=mock-{stratum}"],
        "identity_scope": "redacted_no_body_identity" if redacted else "exact_body_no_credentials",
        "output_budget": output_budget,
        "metadata": {"captured_at_utc": f"2026-10-20T00:00:{request:02d}Z"},
    }


def _make(descriptors: list[dict], **kwargs) -> dict:
    options = {
        "seed": 17,
        "samples_per_session": 2,
        "request_budget": 64,
        "cache_recipe_id": "cold-fixed-v1",
        "cache_state": "cold",
        "warmup_requests": 1,
    }
    options.update(kwargs)
    return make_plan(descriptors, **options)


class CorpusTests(unittest.TestCase):
    def test_session_split_is_disjoint_reproducible_and_order_preserving(self) -> None:
        descriptors = [_descriptor(session, request) for session in range(16) for request in (2, 0, 1)]
        plan = _make(descriptors)
        repeated = _make(list(reversed(descriptors)))
        self.assertEqual(plan["plan_id"], repeated["plan_id"])
        self.assertEqual(plan["corpus_id"], repeated["corpus_id"])
        screen_sessions = set(plan["sessions"]["screen"])
        holdout_sessions = set(plan["sessions"]["holdout"])
        self.assertTrue(screen_sessions.isdisjoint(holdout_sessions))
        self.assertEqual(screen_sessions | holdout_sessions, {f"session-{idx:02d}" for idx in range(16)})
        self.assertEqual(plan["adequacy"]["status"], "planned")
        for split in ("screen", "holdout"):
            seen_sessions = {row["session_id"] for row in plan["splits"][split]}
            self.assertEqual(seen_sessions, set(plan["sessions"][split]))
            for session in seen_sessions:
                sequences = [row["sequence"] for row in plan["splits"][split] if row["session_id"] == session]
                self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(plan["cache"]["cache_recipe_id"], "cold-fixed-v1")
        self.assertEqual(plan["cache"]["cache_state"], "cold")

    def test_tiny_corpus_is_marked_inconclusive_and_redacted_identity_is_labeled(self) -> None:
        plan = _make([_descriptor(0, 0, redacted=True), _descriptor(1, 0, redacted=True)])
        self.assertEqual(plan["adequacy"]["status"], "inconclusive_insufficient_corpus")
        self.assertFalse(set(plan["sessions"]["screen"]) & set(plan["sessions"]["holdout"]))
        self.assertTrue(
            all(row["request_sha256"] is None for split in plan["splits"].values() for row in split)
        )
        self.assertTrue(
            all(
                row["identity_scope"] == "redacted_no_body_identity"
                for split in plan["splits"].values()
                for row in split
            )
        )

    def test_weighted_strata_require_exact_complete_weights_and_bind_plan_id(self) -> None:
        descriptors = [_descriptor(session, 0, stratum="common" if session < 8 else "rare") for session in range(16)]
        plan = _make(descriptors, samples_per_session=1, request_budget=16, stratum_weights={"common": 0.8, "rare": 0.2})
        self.assertEqual(plan["stratum_weights"], {"common": 0.8, "rare": 0.2})
        with self.assertRaisesRegex(CorpusError, "exactly one positive weight"):
            _make(descriptors, samples_per_session=1, request_budget=16, stratum_weights={"common": 1.0})

    def test_captured_request_budgets_are_preserved_not_rewritten(self) -> None:
        descriptors = [_descriptor(0, 0, output_budget=8), _descriptor(1, 0, output_budget=4096)]
        plan = _make(descriptors, samples_per_session=1, request_budget=8, warmup_requests=0)
        self.assertTrue(plan["budgets"]["per_request_budget_preserved"])
        self.assertIsNone(plan["budgets"]["output_tokens_per_request"])
        budgets = {row["capsule_id"]: row["output_budget"] for split in plan["splits"].values() for row in split}
        self.assertEqual(budgets["s00-r00"], 8)
        self.assertEqual(budgets["s01-r00"], 4096)

    def test_output_budget_filter_selects_one_cohort_without_rewriting_bodies(self) -> None:
        descriptors = [_descriptor(0, 0, output_budget=8), _descriptor(1, 0, output_budget=4096)]
        plan = _make(descriptors, samples_per_session=1, request_budget=8, output_budget=8, warmup_requests=0)
        self.assertFalse(plan["budgets"]["per_request_budget_preserved"])
        self.assertEqual(plan["budgets"]["output_tokens_per_request"], 8)
        selected = [row for split in plan["splits"].values() for row in split]
        self.assertTrue(selected)
        self.assertTrue(all(row["output_budget"] == 8 for row in selected))
        self.assertEqual([row["capsule_id"] for row in selected], ["s00-r00"])

    def test_corpus_plan_does_not_mix_real_and_generated_traffic(self) -> None:
        descriptors = [_descriptor(0, 0), _descriptor(1, 0, traffic_class="agent_generated")]
        with self.assertRaisesRegex(CorpusError, "only one traffic_class"):
            _make(descriptors)

    def test_descriptor_jsonl_round_trips_into_stable_plan(self) -> None:
        with TemporaryDirectory() as temp_dir:
            rows = [_descriptor(session, 0) for session in range(12)]
            path = Path(temp_dir) / "descriptors.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            loaded = read_descriptors(path)
            plan_from_path = _make(loaded, samples_per_session=1, request_budget=12)
            plan_direct = _make(rows, samples_per_session=1, request_budget=12)
        self.assertEqual(plan_from_path["plan_id"], plan_direct["plan_id"])

    def test_large_corpus_selects_a_bounded_subset_of_sessions(self) -> None:
        descriptors = [_descriptor(session, request) for session in range(100) for request in range(2)]
        plan = _make(descriptors, samples_per_session=1, request_budget=20)
        measured = [row for split in plan["splits"].values() for row in split]
        self.assertLessEqual(len(measured), 20)
        self.assertLess(len(plan["sessions"]["screen"]) + len(plan["sessions"]["holdout"]), 100)
        self.assertTrue(plan["sessions"]["screen"])
        self.assertTrue(plan["sessions"]["holdout"])
        self.assertFalse(set(plan["sessions"]["screen"]) & set(plan["sessions"]["holdout"]))

    def test_bounded_subset_is_deterministic_for_the_same_seed(self) -> None:
        descriptors = [_descriptor(session, request) for session in range(60) for request in range(2)]
        first = _make(descriptors, samples_per_session=1, request_budget=16)
        second = _make(list(reversed(descriptors)), samples_per_session=1, request_budget=16)
        self.assertEqual(first["plan_id"], second["plan_id"])
        self.assertEqual(first["sessions"], second["sessions"])

    def test_session_replay_replays_a_contiguous_captured_prefix_with_explicit_warmups(self) -> None:
        descriptors = [_descriptor(0, request) for request in range(4)]
        plan = _make(descriptors, samples_per_session=1, request_budget=8, warmup_requests=0)
        session = plan["session_replay"]["screen"][0]
        sequences = [item["sequence"] for item in session["replay_sequence"]]
        self.assertEqual(sequences, list(range(1, sequences[-1] + 1)))
        roles = [item["role"] for item in session["replay_sequence"]]
        self.assertEqual(roles[-1], "measured")
        self.assertTrue(all(role == "warmup" for role in roles[:-1]))
        self.assertEqual(session["warmup_requests"], len(roles) - 1)
        self.assertEqual(plan["cache"]["warmup_requests_per_arm"], session["warmup_requests"])

    def test_missing_captured_prefix_makes_a_session_ineligible(self) -> None:
        descriptors = [_descriptor(0, 0), _descriptor(0, 2)]
        plan = _make(descriptors, samples_per_session=2, request_budget=8, warmup_requests=0)
        coverage = plan["coverage"]["strata"][0]
        self.assertEqual(coverage["status"], "no_replayable_contiguous_prefix")
        self.assertEqual(coverage["ineligible_reasons"]["missing_captured_prefix_sequence"], 1)
        self.assertFalse(plan["sessions"]["screen"] and plan["sessions"]["holdout"])

    def test_redacted_capture_is_not_replayable(self) -> None:
        descriptors = [_descriptor(0, 0, redacted=True), _descriptor(1, 0, redacted=True)]
        plan = _make(descriptors, samples_per_session=1, request_budget=8, warmup_requests=0)
        coverage = plan["coverage"]["strata"][0]
        self.assertEqual(coverage["status"], "no_replayable_contiguous_prefix")
        self.assertEqual(plan["adequacy"]["status"], "inconclusive_insufficient_corpus")


if __name__ == "__main__":
    unittest.main()


class StratumFilterTests(unittest.TestCase):
    def test_a_stratum_filter_restricts_the_plan_without_losing_the_rest(self) -> None:
        descriptors = [
            _descriptor(session, request, stratum="coding")
            for session in range(6) for request in (2, 0, 1)
        ] + [
            _descriptor(session, request, stratum="prose")
            for session in range(6, 9) for request in (2, 0, 1)
        ]
        plan = make_plan(descriptors, seed=5, samples_per_session=2, request_budget=64,
                         cache_recipe_id="cache", cache_state="cold", warmup_requests=0,
                         strata=["coding"])
        self.assertEqual(plan["observed_strata"], ["coding", "prose"])
        self.assertEqual([item["stratum"] for item in plan["coverage"]["strata"]], ["coding"])
        selected = [row for split in plan["splits"].values() for row in split]
        self.assertTrue(selected)
        self.assertTrue(all(row["stratum"] == "coding" for row in selected))

    def test_an_unobserved_stratum_is_refused(self) -> None:
        descriptors = [_descriptor(session, 0) for session in range(4)]
        with self.assertRaisesRegex(CorpusError, "not observed in the corpus"):
            make_plan(descriptors, seed=5, samples_per_session=1, request_budget=8,
                      cache_recipe_id="cache", cache_state="cold", warmup_requests=0,
                      strata=["coding"])
