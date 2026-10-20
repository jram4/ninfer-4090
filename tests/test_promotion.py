from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tools.dreamrsi.contracts import canonical_hash
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


def _freeze(repo: Path, packages: Path, ref: str = "candidate") -> dict:
    return freeze_submission(
        repo,
        packages,
        epoch_id="epoch-1",
        worker_id="worker-1",
        hypothesis_id="hypothesis-1",
        contract_id="a" * 64,
        policy_id="b" * 64,
        ref=ref,
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
        "qualification_enabled": True,
        "promotion_authorized": True,
    }


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
    submission = _freeze(repo, tmp_path / "packages")
    receipt = _receipt(submission, base)
    bare = tmp_path / "target.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert _git(bare, "rev-parse", "refs/heads/main") == base

    result = promote(bare, candidate, base, receipt, shadow=False)
    assert result["mutation"] == "updated-main-ref"
    assert _git(bare, "rev-parse", "refs/heads/main") == candidate

    repo_plan = plan_promotion(repo, candidate, base, receipt, shadow=False)
    assert not repo_plan["ready"]
    assert any("bare target" in reason for reason in repo_plan["reasons"])


def test_receipt_for_different_source_snapshot_cannot_promote(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path / "repo")
    candidate = _git(repo, "rev-parse", "candidate")
    submission = _freeze(repo, tmp_path / "packages")
    receipt = _receipt(submission, base)
    receipt["source_snapshot_sha256"] = canonical_hash(["not the tested source"])
    result = plan_promotion(repo, candidate, base, receipt)
    assert not result["ready"]
    assert any("source_snapshot" in reason for reason in result["reasons"])
