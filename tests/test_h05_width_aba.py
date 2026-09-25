#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from tools.bench import h05_width_aba as H05  # noqa: E402


def _gpu(temperature=65, clock=2500):
    return {
        "available": True,
        "uuid": "GPU-test",
        "temperature.gpu": temperature,
        "clocks.gr": clock,
        "memory.used": 23000,
    }


def _sampling_telemetry(seed):
    return {
        **H05.SAMPLING,
        "temperature": 0.6000000238418579,
        "top_p": 0.949999988079071,
        "seed": seed,
    }


def _rows_by_arm(*, k7_decode_seconds=0.82, k7_rounds=27, k3_decode_seconds=1.0,
                 k3_rounds=30, k7_b_decode_seconds=None, k7_b_rounds=None,
                 k7_completion_tokens=None, k7_b_completion_tokens=None):
    cases, _ = H05.load_cases()
    schedule = H05.request_schedule(cases)
    rows_by_arm = {}
    parameters = {
        "k7-a": (7, k7_decode_seconds, k7_rounds, k7_completion_tokens),
        "k3": (3, k3_decode_seconds, k3_rounds, None),
        "k7-b": (
            7,
            k7_decode_seconds if k7_b_decode_seconds is None else k7_b_decode_seconds,
            k7_rounds if k7_b_rounds is None else k7_b_rounds,
            k7_completion_tokens if k7_b_completion_tokens is None else k7_b_completion_tokens,
        ),
    }
    for arm, (width, seconds, rounds, completion_override) in parameters.items():
        rows = []
        for index, (case, seed) in enumerate(schedule, start=1):
            completion = completion_override or 101
            payload = H05.request_payload(case, seed)
            row = {
                "classification": "NON-QUALIFYING",
                "arm": arm,
                "sequence": index,
                "pair_id": f"{case.case_id}:{seed}",
                "case_id": case.case_id,
                "seed": seed,
                "request_body_sha256": H05.canonical_sha256(payload),
                "status": "ok",
                "configuration_verified": True,
                "quality_passed": True,
                "expected_tool_behavior": True,
                "prompt_tokens": 144,
                "completion_tokens": completion,
                "decode_tokens": completion - 1,
                "decode_seconds": seconds,
                "decode_tok_s": (completion - 1) / seconds,
                "rounds": rounds,
                "tokens_per_round": (completion - 1) / rounds,
                "ms_per_round": seconds * 1000.0 / rounds,
                "draft_window": width,
                "drafted_tokens": rounds * width,
                "accepted_tokens": rounds * (width // 2),
                "fallback_steps": rounds,
                "accepted_per_position": [rounds] * min(width, 7),
                "gpu_before": _gpu(),
                "gpu_after": _gpu(),
            }
            rows.append(row)
        rows_by_arm[arm] = rows
    return rows_by_arm


def _server_start(artifact_path, artifact_size, width, port=18082):
    engine = {
        **H05.ENGINE_CONFIG,
        "speculative_draft_window": width,
        "prefill_chunk": H05.ENGINE_CONFIG["prefill_chunk"],
        "log_stats_interval_ms": 0,
    }
    return {
        "artifact_type": H05.SERVER_LOG_ARTIFACT,
        "schema_version": H05.SERVER_LOG_SCHEMA,
        "event": "server_start",
        "server_instance_id": "serve-test",
        "artifact": {"path": str(artifact_path), "size_bytes": artifact_size},
        "server": {
            "public_model_id": H05.MODEL_ID,
            "host": "127.0.0.1",
            "port": port,
            "api_key_configured": False,
        },
        "engine": engine,
        "sampling_defaults": {
            "greedy": False,
            "server_overrides": _sampling_telemetry(None),
        },
        "environment": {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "compute_capability_major": 8,
            "compute_capability_minor": 9,
        },
    }


def _h03_event(*, request_id, engine_round_id, width):
    accepted = 2
    return {
        "schema": "ninfer.decode_research_round.v1",
        "engine_session_id": 91,
        "engine_round_id": engine_round_id,
        "request_id": request_id,
        "request_round_index": 1,
        "batch_size": 1,
        "backend": "mtp",
        "draft_window": width,
        "cuda_graph": True,
        "context_frontier": 128 + engine_round_id,
        "draft_extent": width,
        "accepted_prefix": accepted,
        "committed_draft_prefix": accepted,
        "licensed_count": accepted + 1,
        "committed_count": accepted + 1,
        "round_elapsed_ns": 30_000_000,
        "terminal": True,
        "truncated": False,
        "cancelled": False,
        "finish_reason": "stop_token",
    }


class H05FixtureTest(unittest.TestCase):
    def test_checked_in_public_cases_and_order_are_frozen(self):
        cases, manifest = H05.load_cases()
        self.assertEqual(manifest["classification"], "NON-QUALIFYING")
        self.assertEqual(
            [case.case_id for case in cases],
            ["async_gpu_job_runner", "cuda_toolchain_lookup"],
        )
        self.assertEqual(
            [(case.case_id, seed) for case, seed in H05.request_schedule(cases)],
            [
                ("async_gpu_job_runner", H05.SEEDS[0]),
                ("cuda_toolchain_lookup", H05.SEEDS[0]),
                ("cuda_toolchain_lookup", H05.SEEDS[1]),
                ("async_gpu_job_runner", H05.SEEDS[1]),
                ("async_gpu_job_runner", H05.SEEDS[2]),
                ("cuda_toolchain_lookup", H05.SEEDS[2]),
            ],
        )
        self.assertGreater(H05.SAMPLING["temperature"], 0)

    def test_body_hashes_fix_seed_tools_and_stop_omission(self):
        cases, _ = H05.load_cases()
        code, tool = cases
        first = H05.request_payload(code, H05.SEEDS[0])
        another_seed = H05.request_payload(code, H05.SEEDS[1])
        self.assertNotEqual(H05.canonical_sha256(first), H05.canonical_sha256(another_seed))
        self.assertNotIn("stop", first)
        tool_payload = H05.request_payload(tool, H05.SEEDS[0])
        self.assertEqual(tool_payload["tool_choice"], "auto")
        self.assertEqual(tool_payload["tools"][0]["function"]["name"], "lookup_public_environment_fact")
        self.assertNotIn("stop", tool_payload)

    def test_source_hash_mismatch_rejects_changed_public_source(self):
        cases, fixture = H05.load_cases()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            fixture["source_manifest_sha256"] = "0" * 64
            path.write_text(json.dumps(fixture), encoding="utf-8")
            with self.assertRaisesRegex(H05.HarnessError, "examples manifest hash changed"):
                H05.load_cases(path)


class H05ValidationTest(unittest.TestCase):
    def test_server_start_freezes_artifact_engine_sampling_and_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.ninfer"
            artifact.write_bytes(b"public fixture artifact marker")
            event = _server_start(artifact, artifact.stat().st_size, 7)
            result = H05.validate_server_start(
                event,
                artifact=artifact,
                artifact_size=artifact.stat().st_size,
                draft_tokens=7,
                port=18082,
            )
            self.assertEqual(result["engine"]["speculative_draft_window"], 7)
            event["engine"]["cuda_graph"] = False
            with self.assertRaisesRegex(H05.HarnessError, "Engine configuration mismatch"):
                H05.validate_server_start(
                    event,
                    artifact=artifact,
                    artifact_size=artifact.stat().st_size,
                    draft_tokens=7,
                    port=18082,
                )

    def test_server_start_accepts_exact_serialized_fp32_sampling_values(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.ninfer"
            artifact.write_bytes(b"public fixture artifact marker")
            event = _server_start(artifact, artifact.stat().st_size, 7)
            event["sampling_defaults"]["server_overrides"].update(
                temperature=0.6000000238418579,
                top_p=0.949999988079071,
            )

            result = H05.validate_server_start(
                event,
                artifact=artifact,
                artifact_size=artifact.stat().st_size,
                draft_tokens=7,
                port=18082,
            )

        self.assertEqual(
            result["sampling_defaults"]["server_overrides"]["temperature"],
            0.6000000238418579,
        )
        self.assertEqual(
            result["sampling_defaults"]["server_overrides"]["top_p"],
            0.949999988079071,
        )

    def test_server_start_rejects_sampling_drift_and_malformed_telemetry(self):
        invalid_overrides = (
            ("material temperature drift", {"temperature": 0.61}),
            ("unexplained close temperature value", {"temperature": 0.60000003}),
            ("material top-p drift", {"top_p": 0.96}),
            ("string temperature", {"temperature": "0.6"}),
            ("boolean temperature", {"temperature": True}),
            ("NaN temperature", {"temperature": float("nan")}),
            ("infinite top-p", {"top_p": float("inf")}),
            ("null top-p", {"top_p": None}),
            ("integer encoded as float", {"top_k": 20.0}),
            ("discrete top-k drift", {"top_k": 21}),
            ("unexpected seed", {"seed": 1}),
            ("unexpected field", {"extra": 0}),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.ninfer"
            artifact.write_bytes(b"public fixture artifact marker")
            for description, changes in invalid_overrides:
                with self.subTest(description=description):
                    event = _server_start(artifact, artifact.stat().st_size, 7)
                    event["sampling_defaults"]["server_overrides"].update(changes)
                    with self.assertRaises(H05.HarnessError):
                        H05.validate_server_start(
                            event,
                            artifact=artifact,
                            artifact_size=artifact.stat().st_size,
                            draft_tokens=7,
                            port=18082,
                        )

            for description, field in (
                ("missing temperature", "temperature"),
                ("missing discrete setting", "top_k"),
            ):
                with self.subTest(description=description):
                    event = _server_start(artifact, artifact.stat().st_size, 7)
                    del event["sampling_defaults"]["server_overrides"][field]
                    with self.assertRaises(H05.HarnessError):
                        H05.validate_server_start(
                            event,
                            artifact=artifact,
                            artifact_size=artifact.stat().st_size,
                            draft_tokens=7,
                            port=18082,
                        )

            event = _server_start(artifact, artifact.stat().st_size, 7)
            event["sampling_defaults"]["server_overrides"] = []
            with self.assertRaisesRegex(H05.HarnessError, "missing or malformed"):
                H05.validate_server_start(
                    event,
                    artifact=artifact,
                    artifact_size=artifact.stat().st_size,
                    draft_tokens=7,
                    port=18082,
                )

    def test_request_start_accepts_fp32_sampling_values_and_keeps_seed_exact(self):
        case = H05.load_cases()[0][0]
        payload = H05.request_payload(case, H05.SEEDS[0])
        event = {
            "artifact_type": H05.SERVER_LOG_ARTIFACT,
            "schema_version": H05.SERVER_LOG_SCHEMA,
            "event": "request_start",
            "request": {
                "protocol": "openai_chat_completions",
                "model": payload["model"],
                "stream": False,
                "requested_output_tokens": payload["max_completion_tokens"],
                "requested_output_tokens_source": "client",
                "enable_thinking": payload["enable_thinking"],
                "tool_count": len(payload["tools"]),
                "tool_choice": payload["tool_choice"],
                "sampling": _sampling_telemetry(payload["seed"]),
            },
        }

        H05.validate_request_start(event, payload)
        event["request"]["sampling"]["seed"] += 1
        with self.assertRaisesRegex(H05.HarnessError, "changed for seed"):
            H05.validate_request_start(event, payload)

    def test_tool_quality_checks_structured_call_without_retaining_arguments(self):
        cases, _ = H05.load_cases()
        tool = cases[1]
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "lookup_public_environment_fact",
                                    "arguments": '{"key":"cuda_toolkit"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }
        done = {
            "result": {
                "finish_reason": "stop_token",
                "tool_call_count": 1,
                "tool_call_parse": {"structured_call_count": 1, "fallback_reason": "none"},
            }
        }
        quality = H05._quality_facts(tool, response, done)
        self.assertTrue(quality["quality_passed"])
        self.assertTrue(quality["expected_tool_behavior"])
        self.assertNotIn("tool_arguments", quality)

        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            '{"key":"wrong"}'
        )
        self.assertFalse(H05._quality_facts(tool, response, done)["quality_passed"])

    def test_done_metrics_exclude_prefill_first_token(self):
        cases, _ = H05.load_cases()
        case = cases[0]
        payload = H05.request_payload(case, H05.SEEDS[0])
        start = {
            "artifact_type": H05.SERVER_LOG_ARTIFACT,
            "schema_version": H05.SERVER_LOG_SCHEMA,
            "event": "request_start",
            "request": {
                "request_id": 1,
                "protocol": "openai_chat_completions",
                "model": H05.MODEL_ID,
                "stream": False,
                "requested_output_tokens": 4096,
                "requested_output_tokens_source": "client",
                "enable_thinking": False,
                "tool_count": 0,
                "tool_choice": "none",
                "sampling": _sampling_telemetry(H05.SEEDS[0]),
            },
        }
        done = {
            "artifact_type": H05.SERVER_LOG_ARTIFACT,
            "schema_version": H05.SERVER_LOG_SCHEMA,
            "event": "request_done",
            "request": start["request"],
            "result": {
                "prompt_tokens": 144,
                "completion_tokens": 13,
                "finish_reason": "stop_token",
                "prefix_cache_hit_tokens": 0,
                "prefix_reuse_path": "root",
                "tool_call_count": 0,
                "tool_call_parse": {"structured_call_count": 0, "fallback_reason": "none"},
            },
            "timings_seconds": {"decode": 0.6},
            "speculative": {
                "backend": "mtp",
                "draft_window": 3,
                "rounds": 3,
                "drafted_tokens": 9,
                "accepted_tokens": 6,
                "fallback_steps": 3,
                "accepted_per_position": [3, 2, 1],
            },
        }
        response = {"usage": {"prompt_tokens": 144, "completion_tokens": 13}}
        metrics = H05._validate_done(start, done, payload, response)
        self.assertEqual(metrics["decode_tokens"], 12)
        self.assertEqual(metrics["tokens_per_round"], 4)
        self.assertEqual(metrics["ms_per_round"], 200)

    def test_transport_failure_returns_a_body_free_per_request_row(self):
        case = H05.load_cases()[0][0]

        class BrokenServer:
            def post(self, payload):
                raise H05.HarnessError("synthetic transport failure")

        with patch.object(H05, "gpu_sample", return_value=_gpu()):
            row, last_request_id = H05._run_attempt(
                BrokenServer(),
                case,
                H05.SEEDS[0],
                arm="k7-a",
                draft_tokens=7,
                sequence=1,
                last_request_id=0,
                expected_gpu_uuid="GPU-test",
            )
        self.assertEqual(last_request_id, 0)
        self.assertEqual(row["status"], "failed")
        self.assertIn("synthetic transport failure", row["failure"])
        self.assertEqual(row["pair_id"], f"{case.case_id}:{H05.SEEDS[0]}")
        self.assertNotIn("messages", row)
        self.assertNotIn("response", row)


class H05AnalysisTest(unittest.TestCase):
    def test_stable_aba_selects_acceptance_axis_from_yield_and_cost(self):
        rows = _rows_by_arm()
        expected = H05._expected_pair_ids(H05.load_cases()[0])
        report = H05.analyze_rows(rows, expected_pairs=expected)
        self.assertTrue(report["valid_for_exploratory_axis_selection"])
        self.assertEqual(report["k7_vs_k3"]["branch"], "acceptance-axis")
        self.assertEqual(report["h03_evidence"]["status"], "request_aggregate_fallback")
        self.assertIn("Exact extent-conditioned position survival is unavailable",
                      report["h03_evidence"]["note"])
        self.assertAlmostEqual(
            report["arm_metrics"]["k3"]["committed_decode_tok_s"], 100.0
        )

    def test_shorter_outputs_cannot_become_a_speedup(self):
        rows = _rows_by_arm(k7_completion_tokens=55, k7_b_completion_tokens=55,
                            k7_decode_seconds=0.25, k7_b_decode_seconds=0.25)
        report = H05.analyze_rows(rows, expected_pairs=H05._expected_pair_ids(H05.load_cases()[0]))
        self.assertFalse(report["valid_for_exploratory_axis_selection"])
        self.assertEqual(report["k7_vs_k3"]["branch"], "inconclusive")
        self.assertTrue(any("output_length" in reason for reason in report["invalid_reasons"]))

    def test_failures_missing_tool_behavior_and_body_config_drift_reject_run(self):
        expected = H05._expected_pair_ids(H05.load_cases()[0])
        rows = _rows_by_arm()
        failed = rows["k3"][0]
        failed["status"] = "failed"
        failed["failure"] = "http_500_request_error"
        rows["k7-a"][1]["quality_passed"] = False
        rows["k7-a"][1]["expected_tool_behavior"] = False
        rows["k7-b"][0]["request_body_sha256"] = "0" * 64
        report = H05.analyze_rows(rows, expected_pairs=expected)
        self.assertFalse(report["valid_for_exploratory_axis_selection"])
        self.assertTrue(any("request_failure" in reason for reason in report["invalid_reasons"]))
        self.assertTrue(any("quality_or_tool_behavior_failed" in reason
                            for reason in report["invalid_reasons"]))
        self.assertTrue(any("request_body_hash_drift" in reason
                            for reason in report["invalid_reasons"]))

    def test_k7_aba_and_thermal_drift_reject_axis_selection(self):
        expected = H05._expected_pair_ids(H05.load_cases()[0])
        rows = _rows_by_arm(k7_b_decode_seconds=1.2, k7_b_rounds=35)
        report = H05.analyze_rows(rows, expected_pairs=expected)
        self.assertFalse(report["valid_for_exploratory_axis_selection"])
        self.assertIn("k7_aba_or_thermal_drift_exceeds_gate", report["invalid_reasons"])

        rows = _rows_by_arm()
        for row in rows["k7-b"]:
            row["gpu_after"] = _gpu(temperature=76, clock=2500)
        report = H05.analyze_rows(rows, expected_pairs=expected)
        self.assertFalse(report["valid_for_exploratory_axis_selection"])
        self.assertIn("k7_aba_or_thermal_drift_exceeds_gate", report["invalid_reasons"])

    def test_one_large_paired_k7_drift_is_rejected_even_when_geomean_is_stable(self):
        rows = _rows_by_arm()
        rows["k7-b"][0]["decode_seconds"] *= 0.78
        rows["k7-b"][0]["decode_tok_s"] = (
            rows["k7-b"][0]["decode_tokens"] / rows["k7-b"][0]["decode_seconds"]
        )
        report = H05.analyze_rows(rows, expected_pairs=H05._expected_pair_ids(H05.load_cases()[0]))
        drift = report["paired_k7_a_to_k7_b"]
        self.assertLessEqual(
            abs(drift["geometric_mean_paired_decode_tok_s_ratio_b_over_a"] - 1),
            H05.MAX_K7_DRIFT,
        )
        self.assertGreater(
            drift["max_paired_decode_tok_s_absolute_delta"], H05.MAX_K7_PAIR_DRIFT
        )
        self.assertFalse(report["valid_for_exploratory_axis_selection"])
        self.assertIn("k7_aba_or_thermal_drift_exceeds_gate", report["invalid_reasons"])

    def test_missing_gpu_thermal_sample_is_inconclusive(self):
        rows = _rows_by_arm()
        rows["k3"][0]["gpu_after"] = {"available": False}
        report = H05.analyze_rows(rows, expected_pairs=H05._expected_pair_ids(H05.load_cases()[0]))
        self.assertFalse(report["valid_for_exploratory_axis_selection"])
        self.assertIn("missing_gpu_thermal_sample:k3", report["invalid_reasons"])

    def test_latency_axis_requires_stable_yield_and_higher_round_cost(self):
        rows = _rows_by_arm(
            k7_decode_seconds=1.2,
            k7_rounds=30,
            k7_b_decode_seconds=1.2,
            k7_b_rounds=30,
        )
        report = H05.analyze_rows(rows, expected_pairs=H05._expected_pair_ids(H05.load_cases()[0]))
        self.assertTrue(report["valid_for_exploratory_axis_selection"])
        self.assertEqual(report["k7_vs_k3"]["branch"], "latency-axis")

    def test_h03_jsonl_adds_extent_conditioned_survival(self):
        rows = _rows_by_arm(k7_decode_seconds=0.18, k7_rounds=6,
                            k3_decode_seconds=0.18, k3_rounds=6)
        for arm_rows in rows.values():
            for row in arm_rows:
                row["completion_tokens"] = 4
                row["decode_tokens"] = 3
                row["decode_seconds"] = 0.03
                row["decode_tok_s"] = 100
                row["rounds"] = 1
                row["tokens_per_round"] = 3
                row["ms_per_round"] = 30
        h03_by_arm = {}
        with tempfile.TemporaryDirectory() as directory:
            for arm, width in (("k7-a", 7), ("k3", 3), ("k7-b", 7)):
                path = Path(directory) / f"{arm}.jsonl"
                events = [
                    _h03_event(request_id=index, engine_round_id=index, width=width)
                    for index in range(1, len(rows[arm]) + 2)
                ]
                path.write_text(
                    "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
                )
                h03_by_arm[arm] = H05.summarize_h03(path, width, len(rows[arm]) + 1)
            report = H05.analyze_rows(
                rows,
                expected_pairs=H05._expected_pair_ids(H05.load_cases()[0]),
                h03_by_arm=h03_by_arm,
            )
        self.assertEqual(report["h03_evidence"]["status"], "present")
        self.assertEqual(report["h03_evidence"]["arms"]["k7-a"]["measurement_rounds"], 6)
        self.assertEqual(
            report["h03_evidence"]["arms"]["k7-a"]["position_survival"][0]["survival"],
            1.0,
        )
        self.assertFalse(any("h03_" in reason for reason in report["invalid_reasons"]))


if __name__ == "__main__":
    unittest.main()
