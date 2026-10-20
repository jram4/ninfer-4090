from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from tools.dreamrsi import stage


def _environment(root: Path, *, required: list[str], stage_name: str = "admission") -> dict[str, str]:
    values = {
        "NINFER_REPORT_SCHEMA": "1",
        "NINFER_SUBMISSION_ID": "s" * 64,
        "NINFER_POLICY_ID": "p" * 64,
        "NINFER_CONTRACT_ID": "c" * 64,
        "NINFER_SOURCE_SNAPSHOT_SHA256": "h" * 64,
        "NINFER_ATTEMPT_ID": "a" * 32,
        "NINFER_STAGE": stage_name,
        "NINFER_ATTEMPT_DIR": str(root),
        "NINFER_STAGE_RAW_DIR": str(root / "raw"),
        "NINFER_STAGE_EVIDENCE_PATH": str(root / "report.json"),
        "NINFER_STAGE_REQUIRED_EVIDENCE": json.dumps(required),
    }
    return values


class StageProgramTests(unittest.TestCase):
    def test_toolchain_stage_publishes_required_evidence(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.environ.update(_environment(root, required=["toolchain_evidence"]))
            try:
                result = stage.main(["toolchain"])
            finally:
                for key in list(os.environ):
                    if key.startswith("NINFER_"):
                        del os.environ[key]
            if result == 0:
                report = json.loads((root / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["schema_version"], 1)
                self.assertEqual(report["status"], "pass")
                self.assertIn("cmake", report["toolchain"])
                self.assertEqual(report["evidence"]["toolchain_evidence"]["path"],
                                 os.path.join("raw", "toolchain.json"))
            else:
                # A host without the verified toolchain must fail the stage
                # rather than report a toolchain it does not have.
                self.assertEqual(result, 2)

    def test_toolchain_versions_are_normalized(self) -> None:
        self.assertEqual(stage._cmake_version_for("cmake version 4.4.0\n"), "4.4.0")
        self.assertEqual(
            stage._cuda_version_for(
                "nvcc: NVIDIA (R) Cuda compiler driver\n"
                "Cuda compilation tools, release 13.0, V13.0.88\n"),
            "13.0",
        )
        self.assertEqual(stage._cuda_version_for("Cuda compilation tools, release 12.4, V12.4.99"),
                         "12.4")

    def test_missing_evidence_is_a_stage_failure(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.environ.update(_environment(root, required=["not_produced"]))
            try:
                self.assertEqual(stage.main(["toolchain"]), 2)
            finally:
                for key in list(os.environ):
                    if key.startswith("NINFER_"):
                        del os.environ[key]

    def test_contract_stage_requires_a_serve_log(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.environ.update(_environment(root, required=["contract_observation"],
                                          stage_name="smoke"))
            try:
                self.assertEqual(stage.main(["contract"]), 2)
            finally:
                for key in list(os.environ):
                    if key.startswith("NINFER_"):
                        del os.environ[key]

    def test_contract_stage_reports_the_observed_runtime(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / "requests.jsonl"
            log.write_text(json.dumps({
                "event": "server_start",
                "artifact": {"size_bytes": 100},
                "engine": {"kv_cache": "k8v4", "max_context": 188416, "kv_capacity": 188416},
                "environment": {"gpu_name": "NVIDIA GeForce RTX 4090"},
                "sampling_defaults": {"greedy": False},
            }) + "\n", encoding="utf-8")
            environment = _environment(root, required=["contract_observation"], stage_name="smoke")
            environment["NINFER_SERVE_LOG"] = str(log)
            environment["NINFER_MODEL_SHA256"] = "a" * 64
            os.environ.update(environment)
            try:
                self.assertEqual(stage.main(["contract"]), 0)
            finally:
                for key in list(os.environ):
                    if key.startswith("NINFER_"):
                        del os.environ[key]
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            observed = report["observed_contract"]
            self.assertEqual(observed["runtime"]["kv_dtype"], "k8v4")
            self.assertEqual(observed["runtime"]["max_context"], 188416)
            self.assertEqual(observed["model_artifact"]["sha256"], "a" * 64)
            self.assertEqual(observed["tokenizer"]["embedded_in_model_artifact_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
