import copy
import hashlib
import json
import unittest

import quick_eval


def descriptor(
    capsule_id: str,
    label: str,
    output_budget: int | None = None,
    *,
    semantic_transform: bool = False,
    production_compatible: bool | None = None,
) -> dict:
    row = {
        "schema_version": 1,
        "capsule_id": capsule_id,
        "capsule_path": f"/capture/{capsule_id}",
        "bucket_labels": [label],
        "request_sha256": hashlib.sha256(capsule_id.encode()).hexdigest(),
        "metadata": {"prompt": "DO NOT COPY THIS PROMPT"},
    }
    if output_budget is not None:
        row["output_budget"] = output_budget
    if semantic_transform or production_compatible is not None:
        row["metadata"] = {
            "semantic_transform": semantic_transform,
            "production_compatible": (
                not semantic_transform if production_compatible is None else production_compatible
            ),
        }
        if semantic_transform:
            row["metadata"].update(
                {
                    "parent_raw_request_sha256": "b" * 64,
                    "transformed_body_sha256": row["request_sha256"],
                    "diff": {
                        "path": "/thinking/display",
                        "from": "omitted",
                        "to": "summarized",
                    },
                }
            )
    return row


def plan_with_three_capsules() -> dict:
    return quick_eval.make_plan(
        [descriptor("c1", "short"), descriptor("c2", "medium"), descriptor("c3", "long")],
        seed=91,
        per_bucket=1,
        cache_recipe_id="prefix-reuse-disabled-v1",
        output_budget=50,
    )


def serve_row(
    capsule_id: str,
    request_id: int,
    *,
    server: str,
    completion_tokens: int = 20,
    decode_seconds: float = 1.6,
    ttft_seconds: float = 0.25,
    finish_reason: str = "output_limit",
    event: str = "request_done",
    complete: bool = True,
    client_ttft: float = 0.28,
    vram_peak_bytes: int | None = 64_000_000,
) -> dict:
    request = {
        "request_id": request_id,
        "protocol": "anthropic_messages",
        "model": "qwen-test",
        "requested_output_tokens": 50,
        "enable_thinking": False,
        "sampling": {"seed": 7},
    }
    start = {
        "schema_version": 21,
        "event": "request_start",
        "server_instance_id": server,
        "request": request,
    }
    if event == "request_done":
        terminal = {
            "schema_version": 21,
            "event": event,
            "server_instance_id": server,
            "request": request,
            "result": {
                "finish_reason": finish_reason,
                "prompt_tokens": 31,
                "completion_tokens": completion_tokens,
                "tool_call_count": 1,
                "tool_call_parse": {
                    "marker_seen": True,
                    "structured_call_count": 1,
                    "empty_arguments_omitted": 0,
                    "schema_mismatch_arguments": 0,
                    "fallback_reason": "none",
                },
            },
            "timings_seconds": {
                "prepare": 0.03,
                "ttft": ttft_seconds,
                "vision": 0.0,
                "prefill": 0.20,
                "decode": decode_seconds,
                "total": 2.0,
            },
            "speculative": {
                "backend": "mtp",
                "draft_window": 4,
                "rounds": 12,
                "drafted_tokens": 36,
                "accepted_tokens": 18,
            },
        }
        received = completion_tokens
    else:
        terminal = {
            "schema_version": 21,
            "event": event,
            "server_instance_id": server,
            "request": request,
            "error": {"message": "omitted from report"},
        }
        received = None
    return {
        "schema_version": 1,
        "plan_id": None,
        "capsule_id": capsule_id,
        "request_start": start,
        "terminal": terminal,
        "client": {
            "complete": complete,
            "finish_reason": finish_reason,
            "received_completion_tokens": received,
            "ttft_seconds": client_ttft,
            "wall_seconds": 2.2,
            "error": None if complete else "transport failure detail should not escape",
        },
        "vram_peak_bytes": vram_peak_bytes,
        "identity": {"artifact_sha256": "a" * 64, "runtime_id": "fnn-test"},
    }


def arm_rows(plan: dict, *, server: str, **overrides) -> list[dict]:
    rows = []
    for index, selected in enumerate(plan["selected"], 1):
        row = serve_row(selected["capsule_id"], index, server=server, **overrides)
        row["plan_id"] = plan["plan_id"]
        row["cache_recipe_id"] = plan["cache_recipe_id"]
        row["semantic_transform"] = selected.get(
            "semantic_transform", plan.get("semantic_transform", False)
        )
        row["production_compatible"] = selected.get(
            "production_compatible",
            plan.get("production_compatible", not row["semantic_transform"]),
        )
        rows.append(row)
    return rows


class QuickEvalTests(unittest.TestCase):
    def test_sampler_is_seeded_stratified_and_does_not_copy_metadata(self) -> None:
        descriptors = [
            descriptor("s1", "short"),
            descriptor("s2", "short"),
            descriptor("m1", "medium"),
            descriptor("l1", "long"),
            descriptor("t1", "tool"),
        ]
        kwargs = {
            "seed": 17,
            "per_bucket": 1,
            "cache_recipe_id": "fixed-recipe",
            "output_budget": 80,
        }
        first = quick_eval.make_plan(descriptors, **kwargs)
        second = quick_eval.make_plan(descriptors, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["bucket_counts"], {"long": 1, "medium": 1, "short": 1, "tool": 1})
        selected_ids = {row["capsule_id"] for row in first["selected"]}
        self.assertTrue(selected_ids <= {row["capsule_id"] for row in descriptors})
        self.assertNotIn("DO NOT COPY THIS PROMPT", json.dumps(first))
        self.assertEqual(first["sampler"]["bucket_labels"], ["long", "medium", "short", "tool"])

    def test_sampler_rejects_unobserved_labels(self) -> None:
        with self.assertRaisesRegex(quick_eval.QuickEvalError, "not observed"):
            quick_eval.make_plan(
                [descriptor("s1", "short")],
                seed=1,
                per_bucket=1,
                cache_recipe_id="r1",
                output_budget=10,
                bucket_labels=["invented"],
            )

    def test_sampler_excludes_capsules_with_a_different_captured_budget(self) -> None:
        plan = quick_eval.make_plan(
            [descriptor("wrong-budget", "short", 40), descriptor("right-budget", "short", 50)],
            seed=1,
            per_bucket=1,
            cache_recipe_id="r1",
            output_budget=50,
        )
        self.assertEqual([item["capsule_id"] for item in plan["selected"]], ["right-budget"])

    def test_sampler_marks_a_transformed_workload_non_production_compatible(self) -> None:
        plan = quick_eval.make_plan(
            [
                descriptor("t1", "short", semantic_transform=True),
                descriptor("t2", "medium", semantic_transform=True),
                descriptor("t3", "long", semantic_transform=True),
            ],
            seed=2,
            per_bucket=1,
            cache_recipe_id="fixed-recipe",
            output_budget=50,
        )
        self.assertTrue(plan["semantic_transform"])
        self.assertFalse(plan["production_compatible"])
        self.assertTrue(all(row["semantic_transform"] for row in plan["selected"]))
        self.assertTrue(all(not row["production_compatible"] for row in plan["selected"]))
        self.assertEqual(plan["selected"][0]["parent_request_sha256"], "b" * 64)
        self.assertEqual(
            plan["selected"][0]["transform_diff"],
            {"path": "/thinking/display", "from": "omitted", "to": "summarized"},
        )

    def test_transformed_pair_can_only_receive_an_exploratory_screen(self) -> None:
        plan = quick_eval.make_plan(
            [
                descriptor("t1", "short", semantic_transform=True),
                descriptor("t2", "medium", semantic_transform=True),
                descriptor("t3", "long", semantic_transform=True),
            ],
            seed=2,
            per_bucket=1,
            cache_recipe_id="fixed-recipe",
            output_budget=50,
        )
        baseline = arm_rows(plan, server="serve-base", decode_seconds=2.0, ttft_seconds=0.32)
        candidate = arm_rows(plan, server="serve-candidate", decode_seconds=1.5, ttft_seconds=0.27)
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)

        self.assertTrue(report["semantic_transform"])
        self.assertFalse(report["production_compatible"])
        self.assertEqual(
            report["identity"]["semantic_transform_provenance"][0]["parent_request_sha256"],
            "b" * 64,
        )
        self.assertEqual(report["exploratory_classification"], "promising")
        self.assertEqual(report["classification"], "exploratory only")
        self.assertEqual(report["production_classification"], "ineligible")
        self.assertTrue(
            all(not row["candidate_win_eligible"] for row in report["paired"]["capsules"])
        )
        self.assertTrue(
            all(row["exploratory_candidate_win_eligible"] for row in report["paired"]["capsules"])
        )

    def test_result_transform_marker_must_match_the_plan(self) -> None:
        plan = quick_eval.make_plan(
            [
                descriptor("t1", "short", semantic_transform=True),
                descriptor("t2", "medium", semantic_transform=True),
                descriptor("t3", "long", semantic_transform=True),
            ],
            seed=2,
            per_bucket=1,
            cache_recipe_id="fixed-recipe",
            output_budget=50,
        )
        baseline = arm_rows(plan, server="serve-base")
        candidate = arm_rows(plan, server="serve-candidate")
        candidate[0]["semantic_transform"] = False
        with self.assertRaisesRegex(quick_eval.QuickEvalError, "semantic_transform differs from the plan"):
            quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=100)

    def test_semantic_transform_cannot_claim_production_compatibility(self) -> None:
        with self.assertRaisesRegex(quick_eval.QuickEvalError, "cannot be production-compatible"):
            quick_eval.make_plan(
                [descriptor("t1", "short", semantic_transform=True, production_compatible=True)],
                seed=1,
                per_bucket=1,
                cache_recipe_id="fixed-recipe",
                output_budget=50,
            )

    def test_derived_hash_must_match_descriptor_request_hash(self) -> None:
        row = descriptor("t1", "short", semantic_transform=True)
        row["metadata"]["transformed_body_sha256"] = "c" * 64
        with self.assertRaisesRegex(quick_eval.QuickEvalError, "differs from request_sha256"):
            quick_eval.make_plan(
                [row],
                seed=1,
                per_bucket=1,
                cache_recipe_id="fixed-recipe",
                output_budget=50,
            )

    def test_ordered_two_field_transform_diff_survives_plan_and_report(self) -> None:
        row = descriptor("t1", "short", semantic_transform=True)
        diff = [
            {"path": "$.thinking.display", "from": "omitted", "to": "summarized"},
            {"path": "$.output_config.effort", "from": "high", "to": "xhigh"},
        ]
        row["metadata"]["diff"] = diff
        plan = quick_eval.make_plan(
            [row],
            seed=3,
            per_bucket=1,
            cache_recipe_id="fixed-recipe",
            output_budget=50,
        )
        baseline = arm_rows(plan, server="serve-base")
        candidate = arm_rows(plan, server="serve-candidate")
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=100)

        self.assertEqual(plan["selected"][0]["transform_diff"], diff)
        self.assertEqual(
            report["identity"]["semantic_transform_provenance"][0]["transform_diff"],
            diff,
        )
        for pair in report["paired"]["capsules"]:
            self.assertEqual(pair["baseline"]["transform_diff"], diff)
            self.assertEqual(pair["candidate"]["transform_diff"], diff)
            self.assertTrue(pair["semantic_transform"])
            self.assertFalse(pair["production_compatible"])

    def test_transform_diff_array_must_have_two_distinct_paths(self) -> None:
        row = descriptor("t1", "short", semantic_transform=True)
        first = {"path": "$.thinking.display", "from": "omitted", "to": "summarized"}
        row["metadata"]["diff"] = [first]
        with self.assertRaisesRegex(quick_eval.QuickEvalError, "ordered two-entry array"):
            quick_eval.make_plan(
                [row], seed=1, per_bucket=1, cache_recipe_id="r", output_budget=50
            )

        row["metadata"]["diff"] = [first, {**first, "to": "visible"}]
        with self.assertRaisesRegex(quick_eval.QuickEvalError, "repeat a transformed path"):
            quick_eval.make_plan(
                [row], seed=1, per_bucket=1, cache_recipe_id="r", output_budget=50
            )

    def test_reducer_parses_schema_v21_and_reports_paired_metrics(self) -> None:
        plan = plan_with_three_capsules()
        baseline = arm_rows(plan, server="serve-base", decode_seconds=2.0, ttft_seconds=0.32, client_ttft=0.35)
        candidate = arm_rows(plan, server="serve-candidate", decode_seconds=1.5, ttft_seconds=0.27, client_ttft=0.29)
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)

        self.assertEqual(report["classification"], "promising")
        self.assertEqual(report["arms"]["candidate"]["token_and_round_totals"]["generated_tokens"], 60)
        self.assertEqual(report["arms"]["candidate"]["token_and_round_totals"]["committed_tokens"], 60)
        self.assertEqual(report["arms"]["candidate"]["token_and_round_totals"]["pooled_tokens_per_round"], 2.5)
        self.assertEqual(report["arms"]["candidate"]["tool_call_parse"]["structured_call_count"], 3)
        self.assertEqual(report["arms"]["candidate"]["vram_peak_bytes"], 64_000_000)
        self.assertEqual(report["paired"]["complete_comparable"], 3)
        self.assertEqual(
            report["paired"]["capsules"][0]["capsule_id"],
            plan["selected"][0]["capsule_id"],
        )
        self.assertAlmostEqual(
            report["paired"]["capsules"][0]["decode_speed_ratio"],
            (19 / 1.5) / (19 / 2.0),
        )
        self.assertIsNotNone(
            report["paired"]["uncertainty"]["decode_speed_ratio_candidate_over_baseline"]["bootstrap_95_ci"]
        )
        self.assertEqual(
            report["arms"]["candidate"]["identities_observed"],
            [{"artifact_sha256": "a" * 64, "runtime_id": "fnn-test"}],
        )

    def test_shorter_candidate_output_never_classifies_as_a_win(self) -> None:
        plan = plan_with_three_capsules()
        baseline = arm_rows(plan, server="serve-base", completion_tokens=20, decode_seconds=2.0)
        candidate = arm_rows(plan, server="serve-candidate", completion_tokens=19, decode_seconds=0.5)
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)

        self.assertEqual(report["classification"], "flat/noisy")
        self.assertEqual(report["paired"]["candidate_shorter_outputs"], 3)
        self.assertEqual(len(report["paired"]["candidate_win_blocked_capsules"]), 3)
        self.assertIn("shorter output", report["classification_reason"])

    def test_material_paired_slowdown_is_an_obvious_regression(self) -> None:
        plan = plan_with_three_capsules()
        baseline = arm_rows(plan, server="serve-base", decode_seconds=1.0, ttft_seconds=0.30, client_ttft=0.32)
        candidate = arm_rows(plan, server="serve-candidate", decode_seconds=2.0, ttft_seconds=0.30, client_ttft=0.32)
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)
        self.assertEqual(report["classification"], "obvious regression")

    def test_one_pair_can_reject_a_very_large_regression_but_has_no_interval(self) -> None:
        plan = quick_eval.make_plan(
            [descriptor("one", "short")],
            seed=9,
            per_bucket=1,
            cache_recipe_id="same-recipe",
            output_budget=50,
        )
        baseline = arm_rows(plan, server="serve-base", decode_seconds=1.0)
        candidate = arm_rows(plan, server="serve-candidate", decode_seconds=2.5)
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)
        self.assertEqual(report["classification"], "obvious regression")
        uncertainty = report["paired"]["uncertainty"]["decode_speed_ratio_candidate_over_baseline"]
        self.assertEqual(uncertainty["count"], 1)
        self.assertIsNone(uncertainty["bootstrap_95_ci"])

    def test_failed_or_incomplete_candidate_is_censored_not_a_win(self) -> None:
        plan = plan_with_three_capsules()
        baseline = arm_rows(plan, server="serve-base")
        candidate = arm_rows(plan, server="serve-candidate", event="request_error", complete=False)
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)

        self.assertNotEqual(report["classification"], "promising")
        self.assertEqual(report["arms"]["candidate"]["completed"], 0)
        self.assertEqual(report["arms"]["candidate"]["error_capsules"], 3)
        encoded = json.dumps(report)
        self.assertNotIn("transport failure detail should not escape", encoded)
        self.assertNotIn("omitted from report", encoded)

    def test_generation_sampler_or_cache_recipe_mismatch_blocks_a_win(self) -> None:
        plan = plan_with_three_capsules()
        baseline = arm_rows(plan, server="serve-base")
        candidate = arm_rows(plan, server="serve-candidate")
        for row in candidate:
            row["cache_recipe_id"] = "different-cache-recipe"
            row["request_start"]["request"]["sampling"]["seed"] = 99
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)
        self.assertEqual(report["paired"]["complete_responses"], 3)
        self.assertEqual(report["paired"]["complete_comparable"], 0)
        self.assertNotEqual(report["classification"], "promising")
        self.assertEqual(report["arms"]["candidate"]["cache_recipe_mismatch_capsules"], 3)

    def test_plan_tampering_and_wrong_serve_schema_are_rejected(self) -> None:
        plan = plan_with_three_capsules()
        tampered = copy.deepcopy(plan)
        tampered["output_budget"] = 51
        with self.assertRaisesRegex(quick_eval.QuickEvalError, "plan_id"):
            quick_eval.reduce_pair(tampered, [], [])

        baseline = arm_rows(plan, server="serve-base")
        candidate = arm_rows(plan, server="serve-candidate")
        candidate[0]["terminal"]["schema_version"] = 20
        with self.assertRaisesRegex(quick_eval.QuickEvalError, "schema-v21"):
            quick_eval.reduce_pair(plan, baseline, candidate)

    def test_missing_capsule_row_is_counted_as_censored(self) -> None:
        plan = plan_with_three_capsules()
        baseline = arm_rows(plan, server="serve-base")
        candidate = arm_rows(plan, server="serve-candidate")[:-1]
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)
        self.assertEqual(report["arms"]["candidate"]["censored_or_failed"], 1)
        self.assertNotEqual(report["classification"], "promising")

    def test_missing_terminal_telemetry_is_reported_as_censored(self) -> None:
        plan = plan_with_three_capsules()
        baseline = arm_rows(plan, server="serve-base")
        candidate = arm_rows(plan, server="serve-candidate")
        candidate[0]["terminal"] = None
        candidate[0]["client"]["complete"] = False
        candidate[0]["client"]["error"] = "not copied into output"
        report = quick_eval.reduce_pair(plan, baseline, candidate, bootstrap_resamples=500)
        first = report["paired"]["capsules"][0]["candidate"]
        self.assertEqual(first["terminal_event"], "missing_terminal_event")
        self.assertIn("missing_terminal_event", first["censored_reasons"])
        self.assertNotEqual(report["classification"], "promising")


if __name__ == "__main__":
    unittest.main()
