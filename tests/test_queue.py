from __future__ import annotations

from pathlib import Path
import subprocess
import time

import pytest

from tools.dreamrsi.queue import (
    ActiveWorkerError,
    Queue,
    QueueFullError,
    StaleClaimError,
)
from tools.dreamrsi.submit import freeze_submission


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Bench Test")
    _git(path, "config", "user.email", "bench@example.invalid")
    (path / "src").mkdir()
    (path / "src" / "engine.cpp").write_text("int engine;\n", encoding="utf-8")
    _git(path, "add", "src/engine.cpp")
    _git(path, "commit", "-qm", "base")
    return path


def _submission(repo: Path, packages: Path, hypothesis: str = "idea-1", worker: str = "worker-1") -> dict:
    return freeze_submission(
        repo,
        packages,
        epoch_id="epoch-1",
        worker_id=worker,
        hypothesis_id=hypothesis,
        contract_id="a" * 64,
        policy_id="b" * 64,
    )


def _receipt(submission: dict) -> dict:
    return {
        "schema_version": 1,
        "submission_id": submission["submission_id"],
        "policy_id": submission["policy_id"],
        "contract_id": submission["contract_id"],
        "source_snapshot_sha256": submission["source"]["snapshot_sha256"],
        "verdict": "promising",
        "stage": "screen",
        "failure_class": None,
        "evidence": [{"path": "artifacts/screen.json", "sha256": "c" * 64}],
        "created_at": "2026-10-20T00:00:00+00:00",
        "candidate_commit": submission["candidate_commit"],
        "base_commit": submission["base_commit"],
    }


def test_duplicate_enqueue_is_idempotent_and_worker_has_one_active_job(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    queue = Queue(tmp_path / "queue.sqlite", max_pending=4)
    first = _submission(repo, tmp_path / "packages")
    assert queue.enqueue(first) == first["submission_id"]
    assert queue.enqueue(first) == first["submission_id"]
    second = _submission(repo, tmp_path / "packages", "idea-2")
    with pytest.raises(ActiveWorkerError):
        queue.enqueue(second)
    claim = queue.claim("worker-1")
    assert claim is not None
    assert claim["submission_id"] == first["submission_id"]
    assert queue.claim("worker-1") is None


def test_central_evaluator_claims_global_fifo_across_submission_authors(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    queue = Queue(tmp_path / "queue.sqlite", max_pending=4)
    first = _submission(repo, tmp_path / "packages-a", "idea-1", "opt1")
    second = _submission(repo, tmp_path / "packages-b", "idea-2", "opt2")
    queue.enqueue(first)
    queue.enqueue(second)

    first_claim = queue.claim("fnn-gpu0")
    assert first_claim is not None
    assert first_claim["submission"]["worker_id"] == "opt1"
    assert first_claim["submission_id"] == first["submission_id"]
    queue.finish(first["submission_id"], first_claim["claim_token"], _receipt(first))

    second_claim = queue.claim("fnn-gpu0")
    assert second_claim is not None
    assert second_claim["submission"]["worker_id"] == "opt2"
    assert second_claim["submission_id"] == second["submission_id"]


def test_fenced_lease_expiry_rejects_stale_finish_and_allows_retry(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    queue = Queue(tmp_path / "queue.sqlite")
    submission = _submission(repo, tmp_path / "packages")
    queue.enqueue(submission)
    stale = queue.claim("worker-1", lease_seconds=0.02)
    assert stale is not None
    time.sleep(0.04)
    fresh = queue.claim("worker-1", lease_seconds=10)
    assert fresh is not None
    assert fresh["claim_token"] != stale["claim_token"]
    with pytest.raises(StaleClaimError):
        queue.finish(submission["submission_id"], stale["claim_token"], _receipt(submission))
    completed = queue.finish(submission["submission_id"], fresh["claim_token"], _receipt(submission))
    assert completed["status"] == "completed"
    assert completed["attempt"] == 2
    assert completed["receipt"]["verdict"] == "promising"


def test_recover_after_process_crash_fences_old_token(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    queue = Queue(tmp_path / "queue.sqlite")
    submission = _submission(repo, tmp_path / "packages")
    queue.enqueue(submission)
    abandoned = queue.claim("worker-1", lease_seconds=60)
    assert abandoned is not None
    recovered = queue.recover(now=abandoned["lease_expires_at"] + 1)
    assert recovered == [submission["submission_id"]]
    assert queue.get(submission["submission_id"])["status"] == "queued"
    with pytest.raises(StaleClaimError):
        queue.finish(submission["submission_id"], abandoned["claim_token"], _receipt(submission))
    retry = queue.claim("worker-1", lease_seconds=60)
    assert retry is not None
    assert retry["submission_id"] == submission["submission_id"]
    assert queue.get(submission["submission_id"])["attempt"] == 2


def test_cancellation_wins_over_late_success(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "candidate")
    queue = Queue(tmp_path / "queue.sqlite")
    submission = _submission(repo, tmp_path / "packages")
    queue.enqueue(submission)
    claim = queue.claim("worker-1", lease_seconds=60)
    assert claim is not None
    assert queue.cancel(submission["submission_id"], "operator stopped this run")["status"] == "cancel_requested"
    with pytest.raises(StaleClaimError):
        queue.heartbeat(submission["submission_id"], claim["claim_token"])
    result = queue.finish(submission["submission_id"], claim["claim_token"], _receipt(submission))
    assert result["status"] == "cancelled"
    assert result["receipt"] is None


def test_backpressure_is_bounded(tmp_path: Path) -> None:
    first_repo = _repo(tmp_path / "candidate-1")
    second_repo = _repo(tmp_path / "candidate-2")
    queue = Queue(tmp_path / "queue.sqlite", max_pending=1)
    first = _submission(first_repo, tmp_path / "packages-1")
    second = _submission(second_repo, tmp_path / "packages-2", "idea-2")
    queue.enqueue(first)
    with pytest.raises(QueueFullError):
        queue.enqueue(second)
