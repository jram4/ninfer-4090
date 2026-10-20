from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from tools.dreamrsi.contracts import canonical_hash, write_json
from tools.dreamrsi.promotion import PromotionError, plan_promotion, promote
from tools.dreamrsi.submit import freeze_submission


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Bench Test")
    _git(path, "config", "user.email", "bench@example.invalid")
    (path / "src").mkdir()
    (path / "src" / "core.cpp").write_text("int core;\n", encoding="utf-8")
    _git(path, "add", "src/core.cpp")
    _git(path, "commit", "-qm", "base")
    base = _git(path, "rev-parse", "HEAD")
    _git(path, "checkout", "-qb", "candidate")
    (path / "candidate.cpp").write_text("int candidate;\n", encoding="utf-8")
    _git(path, "add", "candidate.cpp")
    _git(path, "commit", "-qm", "candidate")
    return path, base


def _freeze(repo: Path, packages: Path, ref: str = "candidate", *, policy_id: str = "b" * 64,
            contract_id: str = "a" * 64, base_ref: str | None = None) -> dict:
    return freeze_submission(
        repo,
        packages,
        epoch_id="epoch-1",
        worker_id="worker-1",
        hypothesis_id="hypothesis-1",
        contract_id=contract_id,
        policy_id=policy_id,
        ref=ref,
        base_ref=base_ref,
    )


def _receipt(submission: dict, base: str) -> dict:
    return {
        "schema_version": 1,
        "submission_id": submission["submission_id"],
        "policy_id": submission["policy_id"],
        "contract_id": submission["contract_id"],
        "source_snapshot_sha256": submission["source"]["snapshot_sha256"],
        "verdict": "qualified",
        "stage": "full",
        "failure_class": None,
        "evidence": [{"path": "immutable/full-gate.json", "sha256": "c" * 64}],
        "created_at": "2026-10-20T00:00:00+00:00",
        "candidate_commit": submission["candidate_commit"],
        "base_commit": base,
    }


def _policy(enabled: bool = True) -> dict:
    contract = {
        "model_artifact": {
            "sha256": "b7c6693244cf7ab08554fff41c4f3c67695b6ec388c99eed6f12e0e71046fda8",
            "size_bytes": 18210704128,
        },
        "runtime": {
            "binary": {
                "sha256": "f9e27c1c162f8767e055c71a9b38c03217bcbcb77437ce20c749454fe99b365a",
                "size_bytes": 320457072,
            },
            "release_source_commit": "70ebb1290dc7abe246c20696a24d21f77faee8d4",
            "kv_dtype": "k8v4",
            "max_context": 188416,
            "kv_capacity": 188416,
            "max_concurrency": 1,
            "speculative_backend": "mtp",
            "draft_tokens": 7,
            "proposal_head": "optimized",
            "prefix_reuse": True,
            "cuda_graph": True,
            "prefill_chunk": 1024,
            "preserve_thinking": True,
            "sampling_defaults": {
                "greedy": False,
                "omitted_seed": "random",
                "server_overrides": {"frequency_penalty": None, "min_p": None,
                                     "presence_penalty": None, "seed": None,
                                     "temperature": None, "top_k": None, "top_p": None},
                "thinking": {"frequency_penalty": 0.0, "min_p": 0.0, "presence_penalty": 0.0,
                             "temperature": 1.0, "top_k": 20, "top_p": 0.949999988079071},
                "non_thinking": {"frequency_penalty": 0.0, "min_p": 0.0, "presence_penalty": 1.5,
                                 "temperature": 0.699999988079071, "top_k": 20,
                                 "top_p": 0.800000011920929},
            },
        },
        "template": {
            "sha256": "ab3f998ebacb74ea1b4f7e5ca2755b0def835f4c47ddf3cff44bc5d4a1a1f3c5",
            "size_bytes": 9858,
        },
        "tokenizer": {"sha256": "c" * 64},
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
            {"name": stage, "argv": ["/usr/bin/true"], "timeout_seconds": 10,
             "required_evidence": ["exit-status"], "criteria": {}}
            for stage in ("admission", "build", "correctness", "smoke", "screen", "full")
        ],
        "qualification_enabled": enabled,
    }
    identity = dict(policy)
    identity.pop("policy_id")
    policy["policy_id"] = canonical_hash(identity)
    return policy


def _trusted_files(root: Path, receipt: dict, policy: dict) -> tuple[Path, Path, Path]:
    root.mkdir()
    artifact = root / "immutable" / "full-gate.json"
    artifact.parent.mkdir()
    artifact.write_bytes(b"trusted full gate\n")
    receipt["evidence"] = [{"path": "immutable/full-gate.json", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]
    receipt_path = root / "receipt.json"
    write_json(receipt_path, receipt)
    policy_path = root.parent / "trusted-policy.json"
    write_json(policy_path, policy)
    return root, receipt_path, policy_path


def test_shadow_plan_does_not_mutate_and_stale_cas_is_denied(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    submission = _freeze(repo, tmp_path / "packages")
    receipt = _receipt(submission, base)

    _git(repo, "checkout", "main")
    (repo / "main-only.cpp").write_text("int main_only;\n", encoding="utf-8")
    _git(repo, "add", "main-only.cpp")
    _git(repo, "commit", "-qm", "advance main")
    current = _git(repo, "rev-parse", "refs/heads/main")

    plan = plan_promotion(repo, candidate, base, receipt)
    assert plan["shadow"] is True
    assert plan["ready"] is False
    assert any("stale" in reason for reason in plan["reasons"])
    assert any("trusted_receipt_root" in reason for reason in plan["reasons"])
    result = promote(repo, candidate, base, receipt)
    assert result["mutation"] == "none"
    assert _git(repo, "rev-parse", "refs/heads/main") == current
    with pytest.raises(PromotionError, match="denied"):
        promote(repo, candidate, base, receipt, shadow=False)
    assert _git(repo, "rev-parse", "refs/heads/main") == current


def test_unrelated_candidate_is_rejected_even_with_qualified_receipt(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    submission = _freeze(repo, tmp_path / "packages")
    receipt = _receipt(submission, base)
    _git(repo, "checkout", "--orphan", "unrelated")
    _git(repo, "rm", "-rf", ".")
    (repo / "alien.cpp").write_text("int alien;\n", encoding="utf-8")
    _git(repo, "add", "alien.cpp")
    _git(repo, "commit", "-qm", "unrelated root")
    unrelated = _git(repo, "rev-parse", "HEAD")
    plan = plan_promotion(repo, unrelated, base, receipt)
    assert not plan["ready"]
    assert any("unrelated" in reason for reason in plan["reasons"])


def test_actual_promotion_requires_qualified_receipt_and_bare_target(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    policy = _policy()
    submission = _freeze(
        repo, tmp_path / "packages", policy_id=policy["policy_id"], contract_id=policy["contract_id"]
    )
    receipt = _receipt(submission, base)
    bare = tmp_path / "target.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert _git(bare, "rev-parse", "refs/heads/main") == base

    trusted_root, receipt_path, policy_path = _trusted_files(tmp_path / "judge", receipt, policy)
    result = promote(
        bare, candidate, base, receipt, shadow=False,
        receipt_path=receipt_path, trusted_receipt_root=trusted_root,
        trusted_policy_path=policy_path, authorize=True,
    )
    assert result["mutation"] == "updated-main-ref"
    assert _git(bare, "rev-parse", "refs/heads/main") == candidate

    repo_plan = plan_promotion(repo, candidate, base, receipt, shadow=False)
    assert not repo_plan["ready"]
    assert any("bare target" in reason for reason in repo_plan["reasons"])


def test_self_reported_authorization_does_not_grant_promotion(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    submission = _freeze(repo, tmp_path / "packages")
    receipt = _receipt(submission, base)
    receipt["qualification_enabled"] = True
    receipt["promotion_authorized"] = True
    bare = tmp_path / "target.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with pytest.raises(PromotionError, match="authorize=True"):
        promote(bare, candidate, base, receipt, shadow=False)


def test_actual_promotion_requires_receipt_inside_trusted_judge_root(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    policy = _policy()
    submission = _freeze(
        repo, tmp_path / "packages", policy_id=policy["policy_id"], contract_id=policy["contract_id"]
    )
    receipt = _receipt(submission, base)
    trusted_root, receipt_path, policy_path = _trusted_files(tmp_path / "judge", receipt, policy)
    outside_receipt = tmp_path / "outside-receipt.json"
    outside_receipt.write_bytes(receipt_path.read_bytes())
    bare = tmp_path / "target.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with pytest.raises(PromotionError, match="receipt_path is outside trusted_receipt_root"):
        promote(
            bare, candidate, base, receipt, shadow=False,
            receipt_path=outside_receipt, trusted_receipt_root=trusted_root,
            trusted_policy_path=policy_path, authorize=True,
        )


@pytest.mark.parametrize("tamper", ["missing", "changed"])
def test_actual_promotion_rejects_missing_or_changed_evidence(tmp_path: Path, tamper: str) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    policy = _policy()
    submission = _freeze(
        repo, tmp_path / "packages", policy_id=policy["policy_id"], contract_id=policy["contract_id"]
    )
    receipt = _receipt(submission, base)
    trusted_root, receipt_path, policy_path = _trusted_files(tmp_path / "judge", receipt, policy)
    artifact = trusted_root / "immutable/full-gate.json"
    if tamper == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"changed after receipt\n")
    bare = tmp_path / "target.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with pytest.raises(PromotionError, match="trusted receipt/evidence verification failed"):
        promote(
            bare, candidate, base, receipt, shadow=False,
            receipt_path=receipt_path, trusted_receipt_root=trusted_root,
            trusted_policy_path=policy_path, authorize=True,
        )


@pytest.mark.parametrize("failure", ["mismatch", "disabled", "contract"])
def test_actual_promotion_rejects_policy_mismatch_and_disabled_policy(tmp_path: Path, failure: str) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    policy = _policy(enabled=failure != "disabled")
    submission = _freeze(
        repo, tmp_path / "packages", policy_id=policy["policy_id"], contract_id=policy["contract_id"]
    )
    receipt = _receipt(submission, base)
    trusted_policy = _policy(enabled=False) if failure == "mismatch" else policy
    if failure == "contract":
        receipt["contract_id"] = canonical_hash("different contract")
    trusted_root, receipt_path, policy_path = _trusted_files(tmp_path / "judge", receipt, trusted_policy)
    bare = tmp_path / "target.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with pytest.raises(PromotionError, match="trusted policy verification failed"):
        promote(
            bare, candidate, base, receipt, shadow=False,
            receipt_path=receipt_path, trusted_receipt_root=trusted_root,
            trusted_policy_path=policy_path, authorize=True,
        )


def test_receipt_for_different_source_snapshot_cannot_promote(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    submission = _freeze(repo, tmp_path / "packages")
    receipt = _receipt(submission, base)
    receipt["source_snapshot_sha256"] = canonical_hash(["not the tested source"])
    result = plan_promotion(repo, candidate, base, receipt)
    assert not result["ready"]
    assert any("source_snapshot" in reason for reason in result["reasons"])
