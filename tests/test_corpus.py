from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.dreamrsi.corpus import CorpusError, make_plan, read_descriptors


def _descriptor(session: int, request: int, *, stratum: str = "interactive", redacted: bool = False) -> dict:
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
        "traffic_class": "real_user",
        "stratum": stratum,
        "labels": ["stream=true", f"model=mock-{stratum}"],
        "identity_scope": "redacted_no_body_identity" if redacted else "exact_body_no_credentials",
        "output_budget": 16,
        "metadata": {"captured_at_utc": f"2026-10-20T00:00:{request:02d}Z"},
    }


def _make(descriptors: list[dict], **kwargs) -> dict:
    options = {
        "seed": 17,
        "samples_per_session": 2,
        "output_budget": 16,
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
                timestamps = [row["captured_at_utc"] for row in plan["splits"][split] if row["session_id"] == session]
                self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(
            plan["cache"],
            {"cache_recipe_id": "cold-fixed-v1", "cache_state": "cold", "warmup_requests_per_arm": 1},
        )
        self.assertEqual(plan["budgets"]["output_tokens_per_request"], 16)

    def test_tiny_corpus_is_marked_inconclusive_and_redacted_identity_is_labeled(self) -> None:
        plan = _make([_descriptor(0, 0, redacted=True), _descriptor(1, 0, redacted=True)])
        self.assertEqual(plan["adequacy"]["status"], "inconclusive_insufficient_corpus")
        self.assertFalse(set(plan["sessions"]["screen"]) & set(plan["sessions"]["holdout"]))
        self.assertTrue(all(row["request_sha256"] is None for split in plan["splits"].values() for row in split))
        self.assertTrue(
            all(row["identity_scope"] == "redacted_no_body_identity" for split in plan["splits"].values() for row in split)
        )

    def test_weighted_strata_require_exact_complete_weights_and_bind_plan_id(self) -> None:
        descriptors = [_descriptor(session, 0, stratum="common" if session < 8 else "rare") for session in range(16)]
        plan = _make(descriptors, samples_per_session=1, request_budget=16, stratum_weights={"common": 0.8, "rare": 0.2})
        self.assertEqual(plan["stratum_weights"], {"common": 0.8, "rare": 0.2})
        with self.assertRaisesRegex(CorpusError, "exactly one positive weight"):
            _make(descriptors, samples_per_session=1, request_budget=16, stratum_weights={"common": 1.0})

    def test_captured_request_budget_is_not_silently_rewritten(self) -> None:
        descriptors = [_descriptor(0, 0)]
        descriptors[0]["output_budget"] = 8
        with self.assertRaisesRegex(CorpusError, "captured output budget differs"):
            _make(descriptors)

    def test_corpus_plan_does_not_mix_real_and_generated_traffic(self) -> None:
        descriptors = [_descriptor(0, 0), _descriptor(1, 0)]
        descriptors[1]["traffic_class"] = "agent_generated"
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
        self.assertEqual(plan_from_path["adequacy"]["status"], "inconclusive_insufficient_corpus")


if __name__ == "__main__":
    unittest.main()
