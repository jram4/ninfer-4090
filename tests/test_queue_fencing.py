from __future__ import annotations

from pathlib import Path
import subprocess
import time
import unittest

from tools.dreamrsi.queue import Queue, StaleClaimError
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


class StaleClaimTests(unittest.TestCase):
    """A worker that lost its lease must not act on the item any more."""

    def setUp(self) -> None:
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue = Queue(self.root / "queue.sqlite3")
        self.repo = _repo(self.root / "repo")
        self.submission = freeze_submission(
            self.repo, self.root / "packages",
            epoch_id="epoch-1", worker_id="worker-1", hypothesis_id="hypothesis-1",
            contract_id="a" * 64, policy_id="b" * 64,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_a_stale_claim_cannot_cancel_a_new_owners_claim(self) -> None:
        submission_id = self.queue.enqueue(self.submission)
        first = self.queue.claim("worker-a", lease_seconds=60)
        self.assertIsNotNone(first)
        assert first is not None
        # The first worker stalls until its lease expires and another reclaims it.
        self.queue.recover(now=time.time() + 3600)
        second = self.queue.claim("worker-b", lease_seconds=60)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertNotEqual(second["claim_token"], first["claim_token"])

        with self.assertRaises(StaleClaimError):
            self.queue.cancel(submission_id, "stale worker gave up",
                              claim_token=first["claim_token"])

        state = self.queue.get(submission_id)
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["claim_token"], second["claim_token"])

    def test_the_current_holder_can_cancel_its_own_claim(self) -> None:
        submission_id = self.queue.enqueue(self.submission)
        claim = self.queue.claim("worker-a", lease_seconds=60)
        assert claim is not None
        state = self.queue.cancel(submission_id, "operator asked",
                                  claim_token=claim["claim_token"])
        self.assertEqual(state["status"], "cancel_requested")

    def test_a_repeated_cancellation_stays_idempotent(self) -> None:
        submission_id = self.queue.enqueue(self.submission)
        claim = self.queue.claim("worker-a", lease_seconds=60)
        assert claim is not None
        self.queue.cancel(submission_id, "operator asked", claim_token=claim["claim_token"])
        state = self.queue.cancel(submission_id, "again", claim_token=claim["claim_token"])
        self.assertEqual(state["status"], "cancel_requested")

    def test_the_record_exposes_the_claim_identity(self) -> None:
        submission_id = self.queue.enqueue(self.submission)
        claim = self.queue.claim("worker-a", lease_seconds=60)
        assert claim is not None
        state = self.queue.get(submission_id)
        self.assertEqual(state["claim_token"], claim["claim_token"])
        self.assertEqual(state["claim_worker"], "worker-a")


if __name__ == "__main__":
    unittest.main()
