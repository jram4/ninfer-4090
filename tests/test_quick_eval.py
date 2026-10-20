from __future__ import annotations

import unittest

from tools.dreamrsi.corpus import make_plan
from tools.dreamrsi.quick_eval import QuickEvalError, reduce_pair


def _descriptor(session: int, request: int, *, stratum: str = "interactive") -> dict:
    capsule_id = f"s{session:02d}-r{request:02d}"
    return {
        "schema_version": 1,
        "capsule_id": capsule_id,
        "sequence": request + 1,
        "capsule_path": f"/private/{capsule_id}.capsule",
        "capsule_sha256": f"{session + 1:064x}",
        "request_sha256": f"{session * 100 + request + 1:064x}",
        "response_sha256": f"{session * 100 + request + 2:064x}",
        "session_id": f"session-{session:02d}",
        "task_label": "task-family-a",
        "run_label": f"capture-{session:02d}",
        "traffic_class": "agent_generated",
        "stratum": stratum,
        "labels": ["stream=true"],
        "identity_scope": "exact_body_no_credentials",
        "output_budget": 8,
        "metadata": {"captured_at_utc": f"2026-10-20T00:00:{request:02d}Z"},
    }


def _plan(session_count: int = 20, *, weighted: bool = False) -> dict:
    descriptors = [
        _descriptor(session, request, stratum=("common" if session < session_count // 2 else "rare") if weighted else "interactive")
        for session in range(session_count)
        for request in range(2)
    ]
    return make_plan(
        descriptors,
        seed=41,
        samples_per_session=2,
        request_budget=64,
        cache_recipe_id="fixed-cache-v1",
        cache_state="warm",
        warmup_requests=0,
        stratum_weights={"common": 0.8, "rare": 0.2} if weighted else None,
    )


def _result(plan: dict, selected: dict, *, arm: str, wall: float = 1.0, tokens: int = 8, error: bool = False, truncated: bool = False) -> dict:
    identity_byte = "a" if arm == "baseline" else "b"
    return {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "corpus_id": plan["corpus_id"],
        "split": "screen",
        "capsule_id": selected["capsule_id"],
        "capsule_sha256": selected["capsule_sha256"],
        "session_id": selected["session_id"],
        "request_sha256": selected["request_sha256"],
        "identity_scope": selected["identity_scope"],
        "output_budget": selected["output_budget"],
        "cache_recipe_id": plan["cache"]["cache_recipe_id"],
        "cache_state": plan["cache"]["cache_state"],
        "warmup_requests": plan["cache"]["warmup_requests_per_arm"],
        "complete": not error and not truncated,
        "error": error,
        "truncated": truncated,
        "finish_reason": "output_limit",
        "committed_tokens": tokens,
        "ttft_seconds": wall * 0.1,
        "wall_seconds": wall,
        "output_work": {"completion_tokens": tokens},
        "runtime_identity": {"artifact_sha256": identity_byte * 64, "runtime_version": "test-1"},
    }


def _arms(plan: dict, candidate_wall: float = 1.0, *, candidate_tokens: int = 8) -> tuple[list[dict], list[dict]]:
    screen = plan["splits"]["screen"]
    baseline = [_result(plan, row, arm="baseline") for row in screen]
    candidate = [_result(plan, row, arm="candidate", wall=candidate_wall, tokens=candidate_tokens) for row in screen]
    return baseline, candidate


class QuickEvalTests(unittest.TestCase):
    def test_paired_reduction_clusters_sessions_and_never_qualifies(self) -> None:
        plan = _plan()
        baseline, candidate = _arms(plan, candidate_wall=1 / 1.1)
        report = reduce_pair(plan, baseline, candidate, bootstrap_resamples=200)
        self.assertEqual(report["classification"], "promising")
        self.assertEqual(report["qualification"], "not_qualified")
        self.assertEqual(report["task_quality"], "not_evaluated")
        stats = report["paired"]["committed_token_rate_ratio_candidate_over_baseline"]
        self.assertEqual(stats["pair_count"], len(plan["splits"]["screen"]))
        self.assertEqual(stats["independent_sessions"], len(plan["sessions"]["screen"]))
        self.assertLess(stats["independent_sessions"], stats["pair_count"])

    def test_three_pairs_or_too_few_sessions_is_inconclusive_not_a_release_signal(self) -> None:
        plan = make_plan(
            [_descriptor(session, 0) for session in range(4)],
            seed=2,
            samples_per_session=1,
            output_budget=8,
            request_budget=4,
            cache_recipe_id="cache",
        )
        baseline = [_result(plan, row, arm="baseline") for row in plan["splits"]["screen"]]
        candidate = [_result(plan, row, arm="candidate", wall=0.1) for row in plan["splits"]["screen"]]
        report = reduce_pair(plan, baseline, candidate, bootstrap_resamples=50)
        self.assertLessEqual(report["paired"]["planned_pairs"], 3)
        self.assertEqual(report["classification"], "inconclusive")
        self.assertEqual(report["qualification"], "not_qualified")

    def test_noisy_session_clusters_do_not_turn_into_request_level_certainty(self) -> None:
        plan = _plan()
        screen = plan["splits"]["screen"]
        baseline = [_result(plan, row, arm="baseline") for row in screen]
        factors = {
            session: (0.5 if index % 2 == 0 else 2.0)
            for index, session in enumerate(plan["sessions"]["screen"])
        }
        candidate = [
            _result(plan, row, arm="candidate", wall=factors[row["session_id"]])
            for row in screen
        ]
        report = reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)
        self.assertEqual(report["classification"], "flat/noisy")
        interval = report["paired"]["committed_token_rate_ratio_candidate_over_baseline"]["bootstrap_95_ci"]
        self.assertLess(interval[0], 1.0)
        self.assertGreater(interval[1], 1.0)

    def test_shorter_outputs_and_finish_mismatch_block_a_performance_win(self) -> None:
        plan = _plan()
        baseline, candidate = _arms(plan, candidate_wall=0.01, candidate_tokens=7)
        report = reduce_pair(plan, baseline, candidate, bootstrap_resamples=100)
        self.assertEqual(report["classification"], "inconclusive")
        self.assertEqual(report["paired"]["candidate_shorter_outputs"], len(candidate))
        self.assertTrue(
            all("candidate_shorter_output" in row["ineligibility_reasons"] for row in report["paired"]["capsules"])
        )

        baseline, candidate = _arms(plan, candidate_wall=0.01)
        candidate[0]["finish_reason"] = "stop"
        report = reduce_pair(plan, baseline, candidate, bootstrap_resamples=100)
        self.assertEqual(report["classification"], "inconclusive")
        self.assertIn("finish_reason_mismatch", report["paired"]["capsules"][0]["ineligibility_reasons"])

    def test_candidate_errors_are_not_dropped_as_fast_samples(self) -> None:
        plan = _plan()
        baseline, candidate = _arms(plan, candidate_wall=0.01)
        candidate[0]["error"] = True
        candidate[0]["complete"] = False
        report = reduce_pair(plan, baseline, candidate, bootstrap_resamples=100)
        self.assertEqual(report["paired"]["candidate_error_or_truncated_pairs"], 1)
        self.assertIn(
            "candidate_incomplete_or_error",
            report["paired"]["capsules"][0]["ineligibility_reasons"],
        )
        self.assertEqual(report["classification"], "regression")

    def test_weighted_strata_are_reported_and_used_for_cluster_bootstrap(self) -> None:
        plan = _plan(weighted=True)
        baseline, candidate = _arms(plan, candidate_wall=1 / 1.1)
        report = reduce_pair(plan, baseline, candidate, bootstrap_resamples=100)
        stats = report["paired"]["committed_token_rate_ratio_candidate_over_baseline"]
        self.assertEqual(set(stats["session_log_means_by_stratum"]), {"common", "rare"})
        self.assertEqual(report["identity"]["stratum_weights"], {"common": 0.8, "rare": 0.2})
        self.assertEqual(report["classification"], "promising")

    def test_result_identity_mismatch_is_rejected_not_paired(self) -> None:
        plan = _plan()
        baseline, candidate = _arms(plan)
        candidate[0]["request_sha256"] = "f" * 64
        with self.assertRaisesRegex(QuickEvalError, "request identity differs"):
            reduce_pair(plan, baseline, candidate, bootstrap_resamples=20)

    def test_body_redaction_is_an_identity_change_and_cannot_claim_a_win(self) -> None:
        descriptors = [_descriptor(session, request) for session in range(20) for request in range(2)]
        plan = make_plan(
            descriptors,
            seed=43,
            samples_per_session=2,
            request_budget=64,
            cache_recipe_id="fixed-cache-v1",
        )
        baseline, candidate = _arms(plan, candidate_wall=0.01)
        # A redacted capture cannot be replayed, so a result that claims a
        # different identity scope is rejected instead of being paired.
        candidate[0]["identity_scope"] = "redacted_no_body_identity"
        with self.assertRaisesRegex(QuickEvalError, "identity_scope differs"):
            reduce_pair(plan, baseline, candidate, bootstrap_resamples=100)

    def test_output_does_not_echo_untrusted_prompt_or_completion_fields(self) -> None:
        plan = _plan()
        baseline, candidate = _arms(plan)
        baseline[0]["prompt"] = "do not publish private prompt"
        candidate[0]["completion"] = "do not publish model completion"
        report = reduce_pair(plan, baseline, candidate, bootstrap_resamples=10)
        encoded = str(report)
        self.assertNotIn("do not publish private prompt", encoded)
        self.assertNotIn("do not publish model completion", encoded)


if __name__ == "__main__":
    unittest.main()
