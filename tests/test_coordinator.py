from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest

from tools.dreamrsi import contracts, coordinator, transport
from tools.dreamrsi.epochs import EpochStore
from tools.dreamrsi.queue import Queue


def _manifest() -> list[dict]:
    return [{"path": "src/main.cpp", "type": "file", "mode": 0o100644, "sha256": "1" * 64}]


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


def _policy() -> dict:
    contract = _contract()
    policy = {
        "schema_version": 1,
        "policy_id": "0" * 64,
        "contract_id": contracts.canonical_hash(contract),
        "contract": contract,
        "baseline_id": "baseline-test",
        "corpus_id": "corpus-test",
        "toolchain_id": "toolchain-test",
        "stages": [
            {"name": name, "argv": ["/usr/bin/true"], "timeout_seconds": 60,
             "required_evidence": [f"{name}_evidence"], "criteria": {"equals": {"status": "pass"}}}
            for name in ("admission", "build", "correctness", "smoke", "screen", "full")
        ],
        "qualification_enabled": False,
    }
    policy["policy_id"] = contracts.canonical_hash({k: v for k, v in policy.items() if k != "policy_id"})
    return policy


def _submission(policy: dict, *, worker: str = "opt-1", epoch: str = "epoch-1",
                archive: Path | None = None) -> dict:
    manifest = _manifest()
    archive_path = archive or Path("/private/source.tar")
    if archive is not None:
        archive_sha256 = contracts.sha256_file(archive)
    else:
        archive_sha256 = "2" * 64
    source = {
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "snapshot_sha256": contracts.canonical_hash(manifest),
        "git_commit": "3" * 40,
        "dirty": False,
        "manifest": manifest,
    }
    submission = {
        "schema_version": 1,
        "submission_id": "0" * 64,
        "epoch_id": epoch,
        "worker_id": worker,
        "hypothesis_id": "hypothesis-1",
        "base_commit": "4" * 40,
        "candidate_commit": "3" * 40,
        "source": source,
        "contract_id": policy["contract_id"],
        "policy_id": policy["policy_id"],
        "created_at": "2026-10-20T00:00:00+00:00",
        "idempotency_key": f"key-{worker}",
    }
    identity = dict(submission)
    identity.pop("submission_id")
    identity.pop("created_at")
    identity["source"] = dict(source)
    identity["source"].pop("archive_path")
    submission["submission_id"] = contracts.canonical_hash(identity)
    return submission


class CoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue_db = self.root / "queue.sqlite3"
        self.epoch_db = self.root / "epochs.sqlite3"
        self.run_root = self.root / "runs"
        self.policy_path = self.root / "policy.json"
        self.policy_path.write_text(json.dumps(_policy()), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _queue(self) -> Queue:
        return Queue(self.queue_db)

    def _epochs(self) -> EpochStore:
        store = EpochStore(self.epoch_db)
        store.create_epoch("epoch-1", {
            "max_workers": 4, "max_hypotheses": 10, "max_assignments": 10,
            "max_prompts": 100, "max_tokens": 1000, "max_wall_seconds": 3600,
            "max_cost_usd": "1.00",
        })
        return store

    def _enqueue(self, policy: dict | None = None, *, worker: str = "opt-1") -> str:
        policy = policy or _policy()
        archive = self.root / "source.tar"
        if not archive.exists():
            archive.write_bytes(b"frozen-candidate-source\n")
        submission = _submission(policy, worker=worker, archive=archive)
        return self._queue().enqueue(submission)

    def test_no_work_returns_none(self) -> None:
        self.assertIsNone(coordinator.run_once(
            self.queue_db, self.policy_path, self.run_root, evaluator_id="fnn-gpu0"))

    def test_claim_evaluate_finish_records_a_verdict(self) -> None:
        self._enqueue()
        calls: list[str] = []

        def executor(submission, policy, run_root, *, cancelled=lambda: False):
            calls.append(submission["submission_id"])
            return {
                "schema_version": 1,
                "submission_id": submission["submission_id"],
                "policy_id": submission["policy_id"],
                "contract_id": submission["contract_id"],
                "source_snapshot_sha256": submission["source"]["snapshot_sha256"],
                "verdict": "promising",
                "stage": "screen",
                "failure_class": None,
                "evidence": [{"path": "artifacts/report.json", "sha256": "4" * 64}],
                "created_at": "2026-10-20T00:00:00+00:00",
                "candidate_commit": "3" * 40,
                "base_commit": "4" * 40,
            }

        record = coordinator.run_once(
            self.queue_db, self.policy_path, self.run_root,
            evaluator_id="fnn-gpu0", executor=executor)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["receipt"]["verdict"], "promising")
        self.assertEqual(len(calls), 1)

    def test_policy_mismatch_is_rejected_without_running_the_evaluator(self) -> None:
        other = _policy()
        other["corpus_id"] = "other-corpus"
        other["policy_id"] = contracts.canonical_hash(
            {k: v for k, v in other.items() if k != "policy_id"})
        self._enqueue(other)

        def executor(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("the evaluator must not run for a mismatched policy")

        record = coordinator.run_once(
            self.queue_db, self.policy_path, self.run_root,
            evaluator_id="fnn-gpu0", executor=executor)
        assert record is not None
        self.assertEqual(record["receipt"]["verdict"], "rejected")
        self.assertEqual(record["receipt"]["failure_class"], "candidate")

    def test_epoch_cancellation_defers_and_cancels_the_item(self) -> None:
        store = self._epochs()
        self._enqueue()
        store.transition("epoch-1", "cancel", "operator stopped the epoch")
        record = coordinator.run_once(
            self.queue_db, self.policy_path, self.run_root,
            evaluator_id="fnn-gpu0", epoch_db=self.epoch_db,
            executor=lambda *a, **k: {})
        assert record is not None
        self.assertEqual(record["status"], "cancelled")
        self.assertIn("cancelled", record["deferred"])

    def test_evaluator_failure_is_infrastructure_not_candidate_evidence(self) -> None:
        self._enqueue()

        def executor(*args, **kwargs):
            raise RuntimeError("harness exploded")

        record = coordinator.run_once(
            self.queue_db, self.policy_path, self.run_root,
            evaluator_id="fnn-gpu0", executor=executor)
        assert record is not None
        self.assertEqual(record["receipt"]["verdict"], "blocked")
        self.assertEqual(record["receipt"]["failure_class"], "infrastructure")

    def test_result_is_recorded_as_an_idempotent_epoch_event(self) -> None:
        store = self._epochs()
        self._enqueue()
        archive = self.root / "source.tar"
        receipt = {
            "schema_version": 1,
            "submission_id": _submission(_policy(), archive=archive)["submission_id"],
            "policy_id": _policy()["policy_id"],
            "contract_id": _policy()["contract_id"],
            "source_snapshot_sha256": contracts.canonical_hash(_manifest()),
            "verdict": "promising",
            "stage": "screen",
            "failure_class": None,
            "evidence": [{"path": "artifacts/report.json", "sha256": "4" * 64}],
            "created_at": "2026-10-20T00:00:00+00:00",
            "candidate_commit": "3" * 40,
            "base_commit": "4" * 40,
        }
        coordinator.run_once(
            self.queue_db, self.policy_path, self.run_root,
            evaluator_id="fnn-gpu0", epoch_db=self.epoch_db,
            executor=lambda submission, policy, run_root, cancelled=lambda: False: receipt)
        events = [event for event in store.list_events("epoch-1")
                  if event["event_type"].startswith("coordinator.")]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "coordinator.evaluation_finished")
        self.assertEqual(events[0]["payload"]["verdict"], "promising")

    def test_queue_and_epoch_state_must_not_share_one_database(self) -> None:
        with self.assertRaises(coordinator.CoordinatorError):
            coordinator.run_once(
                self.queue_db, self.policy_path, self.run_root,
                evaluator_id="fnn-gpu0", epoch_db=self.queue_db)


class TransportTests(unittest.TestCase):
    def test_remote_root_must_be_inside_the_private_research_root(self) -> None:
        with self.assertRaises(ValueError):
            transport._safe_remote_root("/tmp/elsewhere")
        with self.assertRaises(ValueError):
            transport._safe_remote_root("/home/aramirezfamily/josh/cinference-4090")
        self.assertEqual(
            str(transport._safe_remote_root("/home/aramirezfamily/.local/share/ninfer-bench/runs")),
            "/home/aramirezfamily/.local/share/ninfer-bench/runs",
        )

    def test_installation_plan_is_side_effect_free_and_explicit(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = root / "harness"
            harness.mkdir()
            (harness / "evaluator.py").write_text("x = 1\n", encoding="utf-8")
            (harness / "lifecycle.py").write_text("y = 2\n", encoding="utf-8")
            (harness / "contracts.py").write_text("z = 3\n", encoding="utf-8")
            executor = transport.SSHExecutor(
                remote_root=transport.PRIVATE_RESEARCH_ROOT / "runs",
                trusted_harness=harness,
                alias="fnn",
            )
            plan = executor.installation_plan()
        self.assertIn("fnn", json.dumps(plan))
        self.assertNotIn("push", json.dumps(plan).lower())

    def test_ssh_executor_uses_argv_and_never_shell_text(self) -> None:
        import tempfile

        recorded: list[list[str]] = []

        def fake_runner(argv, *, input=None, timeout):
            recorded.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout=b"{}", stderr=b"")

        with tempfile.TemporaryDirectory() as temp:
            harness = Path(temp)
            for name in ("evaluator.py", "lifecycle.py", "contracts.py"):
                (harness / name).write_text("x = 1\n", encoding="utf-8")
            executor = transport.SSHExecutor(
                remote_root=transport.PRIVATE_RESEARCH_ROOT / "runs",
                trusted_harness=harness,
                alias="fnn",
                runner=fake_runner,
            )
            executor._ssh(["echo", "hello; rm -rf /"], timeout=5)
        self.assertTrue(recorded)
        self.assertEqual(recorded[0][0], "ssh")
        self.assertEqual(recorded[0][1], "-T")
        # The remote command is one argument, so a remote shell cannot split it.
        self.assertEqual(len(recorded[0]), 4)
        self.assertIn(";", recorded[0][-1])


if __name__ == "__main__":
    unittest.main()
