from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.dreamrsi import contracts, policy


def _contract() -> dict:
    return {
        "model_artifact": {"sha256": "a" * 64, "size_bytes": 100},
        "runtime": {
            "kv_dtype": "k8v4",
            "max_context": 188416,
            "kv_capacity": 188416,
            "sampling_defaults": {
                "greedy": False,
                "omitted_seed": "random",
                "server_overrides": {},
                "thinking": {"frequency_penalty": 0.0, "min_p": 0.0, "presence_penalty": 0.0,
                             "temperature": 1.0, "top_k": 20, "top_p": 0.95},
                "non_thinking": {"frequency_penalty": 0.0, "min_p": 0.0, "presence_penalty": 1.5,
                                 "temperature": 0.7, "top_k": 20, "top_p": 0.8},
            },
        },
        "template": {"sha256": "b" * 64},
        "tokenizer": {"embedded_in_model_artifact_sha256": "a" * 64},
    }


def _specification() -> dict:
    return {
        "contract": _contract(),
        "baseline_id": "baseline-sampled-k7-70ebb12",
        "corpus_id": "corpus-screen-01",
        "toolchain_id": "fnn-cuda13-cmake44",
        "qualification_enabled": False,
        "stages": [
            {"name": name, "argv": ["/usr/bin/true"], "timeout_seconds": 60,
             "required_evidence": [f"{name}_evidence"],
             "criteria": {"equals": {"status": "pass"}}}
            for name in ("admission", "build", "correctness", "smoke", "screen", "full")
        ],
    }


class PolicyBuilderTests(unittest.TestCase):
    def test_build_computes_identities_and_validates(self) -> None:
        built = policy.build_policy(_specification())
        self.assertEqual(built["contract_id"], contracts.canonical_hash(_contract()))
        self.assertEqual(
            built["policy_id"],
            contracts.canonical_hash({k: v for k, v in built.items() if k != "policy_id"}),
        )
        self.assertFalse(built["qualification_enabled"])
        self.assertEqual([stage["name"] for stage in built["stages"]],
                         list(policy.STAGE_ORDER))

    def test_stage_order_and_duplicates_are_rejected(self) -> None:
        specification = _specification()
        specification["stages"][2], specification["stages"][3] = (
            specification["stages"][3], specification["stages"][2])
        with self.assertRaisesRegex(policy.PolicyError, "in that order"):
            policy.build_policy(specification)

        specification = _specification()
        specification["stages"].append(dict(specification["stages"][0]))
        with self.assertRaisesRegex(policy.PolicyError, "duplicate stage"):
            policy.build_policy(specification)

    def test_qualifying_policy_requires_content_locks(self) -> None:
        specification = _specification()
        specification["qualification_enabled"] = True
        specification["contract"]["model_artifact"].pop("size_bytes")
        with self.assertRaises(ValueError):
            policy.build_policy(specification)

    def test_cli_builds_a_policy_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(_specification()), encoding="utf-8")
            out_path = root / "policy.json"
            self.assertEqual(policy.main(["build", "--spec", str(spec_path),
                                          "--output", str(out_path)]), 0)
            built = contracts.validate_policy(contracts.read_json(out_path))
            self.assertEqual(built["contract_id"], contracts.canonical_hash(_contract()))


if __name__ == "__main__":
    unittest.main()
