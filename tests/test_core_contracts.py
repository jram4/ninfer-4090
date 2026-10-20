from __future__ import annotations

import json

import pytest

from tools.dreamrsi.contracts import (
    canonical_hash,
    compare_contract,
    validate_policy,
    validate_receipt,
    validate_submission,
)


def _submission() -> dict:
    manifest = [
        {"path": "src/models/core.cpp", "type": "file", "mode": 0o100644, "sha256": "1" * 64},
    ]
    source = {
        "archive_path": "/tmp/package/source.tar",
        "archive_sha256": "2" * 64,
        "snapshot_sha256": canonical_hash(manifest),
        "git_commit": "3" * 40,
        "dirty": False,
        "manifest": manifest,
    }
    value = {
        "schema_version": 1,
        "submission_id": "0" * 64,
        "epoch_id": "epoch-1",
        "worker_id": "worker-1",
        "hypothesis_id": "hypothesis-1",
        "base_commit": "4" * 40,
        "candidate_commit": "3" * 40,
        "source": source,
        "contract_id": "5" * 64,
        "policy_id": "6" * 64,
        "created_at": "2026-10-20T00:00:00+00:00",
        "idempotency_key": "key-1",
    }
    identity = dict(value)
    identity.pop("submission_id")
    identity.pop("created_at")
    identity["source"] = dict(source)
    identity["source"].pop("archive_path")
    value["submission_id"] = canonical_hash(identity)
    return value


def _policy() -> dict:
    contract = {
        "model_artifact": {"sha256": "a" * 64},
        "runtime": {"commit": "b" * 40},
        "template": {"sha256": "c" * 64},
        "tokenizer": {"sha256": "d" * 64},
    }
    policy = {
        "schema_version": 1,
        "policy_id": "0" * 64,
        "contract_id": canonical_hash(contract),
        "contract": contract,
        "baseline_id": "baseline-1",
        "corpus_id": "corpus-1",
        "toolchain_id": "toolchain-1",
        "stages": [
            {
                "name": name,
                "argv": ["/usr/bin/true"],
                "timeout_seconds": 60,
                "required_evidence": ["exit-status"],
                "criteria": {},
            }
            for name in ("admission", "build", "correctness", "smoke", "screen", "full")
        ],
        "qualification_enabled": False,
    }
    identity = dict(policy)
    identity.pop("policy_id")
    policy["policy_id"] = canonical_hash(identity)
    return policy


def test_canonical_identity_and_submission_mismatch_are_rejected() -> None:
    submission = _submission()
    assert validate_submission(submission)["submission_id"] == submission["submission_id"]
    submission["hypothesis_id"] = "different"
    with pytest.raises(ValueError, match="submission_id"):
        validate_submission(submission)


def test_submission_manifest_rejects_traversal_and_symlink_escape() -> None:
    submission = _submission()
    entry = submission["source"]["manifest"][0]
    entry["path"] = "../outside.cpp"
    submission["source"]["snapshot_sha256"] = canonical_hash(submission["source"]["manifest"])
    identity = dict(submission)
    identity.pop("submission_id")
    identity.pop("created_at")
    identity["source"] = dict(submission["source"])
    identity["source"].pop("archive_path")
    submission["submission_id"] = canonical_hash(identity)
    with pytest.raises(ValueError, match="unsafe source manifest path"):
        validate_submission(submission)

    submission = _submission()
    entry = {"path": "escape", "type": "symlink", "mode": 0o120000, "target": "../../outside"}
    submission["source"]["manifest"] = [entry]
    submission["source"]["snapshot_sha256"] = canonical_hash([entry])
    identity = dict(submission)
    identity.pop("submission_id")
    identity.pop("created_at")
    identity["source"] = dict(submission["source"])
    identity["source"].pop("archive_path")
    submission["submission_id"] = canonical_hash(identity)
    with pytest.raises(ValueError, match="symlink escapes"):
        validate_submission(submission)


def test_policy_identity_locks_contract_and_requires_ordered_stages() -> None:
    policy = _policy()
    assert validate_policy(policy)["qualification_enabled"] is False
    policy["contract"]["runtime"]["commit"] = "e" * 40
    with pytest.raises(ValueError, match="contract_id"):
        validate_policy(policy)

    policy = _policy()
    policy["stages"].reverse()
    identity = dict(policy)
    identity.pop("policy_id")
    policy["policy_id"] = canonical_hash(identity)
    with pytest.raises(ValueError, match="stages"):
        validate_policy(policy)


def test_contract_comparison_returns_stable_field_paths() -> None:
    assert compare_contract(
        {"model": {"sha256": "a", "dtype": "bf16"}, "layers": [1, 2]},
        {"model": {"sha256": "b", "dtype": "bf16"}, "layers": [1, 3]},
    ) == ["layers[1]", "model.sha256"]


def test_receipt_validates_evidence_hashes() -> None:
    receipt = {
        "schema_version": 1,
        "submission_id": "1" * 64,
        "policy_id": "2" * 64,
        "contract_id": "3" * 64,
        "source_snapshot_sha256": "4" * 64,
        "verdict": "inconclusive",
        "stage": "screen",
        "failure_class": None,
        "evidence": [{"path": "artifacts/report.json", "sha256": "5" * 64}],
        "created_at": "2026-10-20T00:00:00+00:00",
    }
    assert validate_receipt(receipt)["verdict"] == "inconclusive"
    receipt["evidence"][0]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="sha256"):
        validate_receipt(receipt)
