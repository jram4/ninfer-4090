from __future__ import annotations

from pathlib import Path
import sys

import pytest

from tools.dreamrsi import contracts, evaluator, submit


REPO_ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict:
    policy = {
        "schema_version": 1,
        "policy_id": "0" * 64,
        "contract_id": "1" * 64,
        "contract": {
            "model_artifact": {"sha256": "a" * 64, "size_bytes": 1},
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
                "sampling_capabilities": {"mtp": True},
            },
            "template": {"sha256": "b" * 64},
            "tokenizer": {"embedded_in_model_artifact_sha256": "a" * 64},
        },
        "baseline_id": "baseline", "corpus_id": "corpus", "toolchain_id": "toolchain",
        "stages": [
            {"name": name, "argv": ["/usr/bin/true"], "timeout_seconds": 60,
             "required_evidence": [f"{name}_evidence"],
             "criteria": {"equals": {"status": "pass"}}}
            for name in ("admission", "build", "correctness", "smoke", "screen", "full")
        ],
        "qualification_enabled": False,
    }
    policy["policy_id"] = contracts.canonical_hash(
        {key: value for key, value in policy.items() if key != "policy_id"})
    return policy


def test_frozen_archive_extracts_to_exactly_the_frozen_manifest(tmp_path: Path) -> None:
    """A submission must extract to the exact tree it was frozen from."""
    policy = _policy()
    frozen = submit.freeze_submission(
        REPO_ROOT, tmp_path / "packages",
        epoch_id="epoch-roundtrip", worker_id="worker-roundtrip",
        hypothesis_id="hypothesis-roundtrip",
        contract_id=policy["contract_id"], policy_id=policy["policy_id"], ref="HEAD",
        include_dirty=True,
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    evaluator._safe_extract_archive(
        Path(frozen["source"]["archive_path"]), extracted, lambda: False)
    observed = evaluator._extracted_source_manifest(contracts, extracted)
    assert observed == frozen["source"]["manifest"]
    assert contracts.canonical_hash(observed) == frozen["source"]["snapshot_sha256"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
