import subprocess

import pytest

from tools.dreamrsi.epochs import EpochError, EpochStore
from tools.dreamrsi.workers import (
    WorkerError,
    acknowledge_callback,
    build_launch_plan,
    continue_assignment,
    create_assignment,
    drain_callbacks,
    launch_assignment,
    list_workers,
    opencode_permission_rules,
    record_worker_callback,
    recover_launches,
    register_worker,
)


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo(tmp_path):
    root = tmp_path / "worker-tree"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _git(root, "config", "user.name", "Bench Test")
    _git(root, "config", "user.email", "bench-test@example.invalid")
    (root / "src").mkdir()
    (root / "src" / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "src/kernel.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    return root, _git(root, "rev-parse", "HEAD")


def _start(store, epoch_id="epoch-a"):
    store.create_epoch(epoch_id, {
        "max_workers": 2,
        "max_hypotheses": 3,
        "max_assignments": 4,
        "max_prompts": 4,
        "max_tokens": 500,
        "max_cost_usd": "5.000000",
        "max_wall_seconds": 600,
    })


def _register(store, root, base, epoch_id="epoch-a", worker_id="worker-a", allowed=None):
    allowed = allowed or ["src/kernel.py"]
    return register_worker(
        store,
        epoch_id=epoch_id,
        worker_id=worker_id,
        worktree=root,
        base_commit=base,
        model="fnn/qwen3.8-27b#medium",
        role="build",
        allowed_files=allowed,
        permissions={"read_paths": allowed, "write_paths": allowed, "shell": False, "network": False},
        stop_conditions=["one bounded task is complete", "permission or model identity differs"],
        callback={"mode": "manual", "recipient": "coordinator"},
    )


def _assign(store, worker_id="worker-a", assignment_id="task-a", epoch_id="epoch-a", path="src/kernel.py"):
    store.add_hypothesis(epoch_id, "hypothesis-a", "the target path can be changed safely")
    return create_assignment(
        store, epoch_id=epoch_id, assignment_id=assignment_id, worker_id=worker_id,
        hypothesis_id="hypothesis-a", idempotency_key=f"assignment:{assignment_id}",
        instructions="Inspect and propose one narrowly scoped change.", read_paths=[path], write_paths=[path],
        token_budget=100, cost_budget_usd="1.000000", wall_budget_seconds=60, prompt_budget=2,
    )


class FakeAdapter:
    def __init__(self):
        self.created = []
        self.prompts = []

    def create_session(self, body):
        self.created.append(body)
        return "ses_fake_worker"

    def send_prompt(self, session_id, body):
        self.prompts.append((session_id, body))
        return {"accepted": True}


def test_worker_path_permissions_dry_run_and_fake_launch(tmp_path):
    root, base = _repo(tmp_path)
    store = EpochStore(tmp_path / "epochs.sqlite3")
    _start(store)
    _register(store, root, base)
    assignment = _assign(store)

    reserved = store.get_epoch("epoch-a")["reserved"]
    store.transition("epoch-a", "pause")
    assert store.transition("epoch-a", "resume")["reserved"] == reserved

    plan = build_launch_plan(store, assignment["assignment_id"], opencode="opencode-test")
    body = plan["session_create"]["body"]
    assert body["location"] == {"directory": str(root)}
    assert "parentID" not in body
    assert body["model"] == {"providerID": "fnn", "id": "qwen3.8-27b", "variant": "medium"}
    assert body["permissions"] == opencode_permission_rules(["src/kernel.py"], ["src/kernel.py"])
    assert body["permissions"][0] == {"action": "*", "resource": "*", "effect": "deny"}
    assert body["permissions"][-2:] == [
        {"action": "read", "resource": "src/kernel.py", "effect": "allow"},
        {"action": "edit", "resource": "src/kernel.py", "effect": "allow"},
    ]
    assert plan["dry_run"] is True and plan["host_wake_supported"] is False
    assert "accounting_only" in plan["budget_enforcement"]
    assert store.get_epoch("epoch-a")["counts"]["active_assignments"] == 1

    fake = FakeAdapter()
    dry_run = launch_assignment(store, assignment["assignment_id"], adapter=fake)
    assert dry_run["dry_run"] is True
    assert fake.created == [] and fake.prompts == []
    result = launch_assignment(store, assignment["assignment_id"], adapter=fake, execute=True, confirm_launch=True)
    assert result["status"] == "launched"
    assert result["prompts_used"] == 1
    assert len(fake.created) == 1 and len(fake.prompts) == 1
    assert fake.prompts[0][0] == "ses_fake_worker"
    assert "src/kernel.py" in fake.prompts[0][1]["text"]

    preview = continue_assignment(store, assignment["assignment_id"], "Continue within this scope.")
    assert preview["new_session"] is False and preview["new_branch"] is False
    assert preview["location"] == str(root)
    continued = continue_assignment(store, assignment["assignment_id"], "One last bounded turn.",
                                    adapter=fake, execute=True, confirm_launch=True)
    assert continued["session_id"] == "ses_fake_worker"
    assert continued["prompts_used"] == 2
    with pytest.raises(WorkerError, match="prompt budget"):
        continue_assignment(store, assignment["assignment_id"], "No third prompt.",
                            adapter=fake, execute=True, confirm_launch=True)


def test_cancellation_between_session_creation_and_prompt_is_fenced(tmp_path):
    root, base = _repo(tmp_path)
    store = EpochStore(tmp_path / "epochs.sqlite3")
    _start(store)
    _register(store, root, base)
    assignment = _assign(store)

    class CancelOnCreate(FakeAdapter):
        def create_session(self, body):
            result = super().create_session(body)
            store.transition("epoch-a", "cancel", "operator stop")
            return result

        def send_prompt(self, session_id, body):
            pytest.fail("no model prompt may be sent after cancellation")

    result = launch_assignment(store, assignment["assignment_id"], adapter=CancelOnCreate(),
                               execute=True, confirm_launch=True)
    assert result["status"] == "cancelled_before_prompt"
    assert store.get_epoch("epoch-a")["status"] == "cancelled"
    assert store.list_events("epoch-a")[-1]["event_type"] == "epoch.cancelled"


def test_callbacks_are_idempotent_manual_and_acknowledged_once(tmp_path):
    root, base = _repo(tmp_path)
    db = tmp_path / "epochs.sqlite3"
    store = EpochStore(db)
    _start(store)
    _register(store, root, base)
    assignment = _assign(store)
    fake = FakeAdapter()
    launch_assignment(store, assignment["assignment_id"], adapter=fake, execute=True, confirm_launch=True)

    event = record_worker_callback(store, epoch_id="epoch-a", worker_id="worker-a", assignment_id="task-a",
                                   idempotency_key="callback:task-a:done", status="done", summary="Finished.")
    duplicate = record_worker_callback(store, epoch_id="epoch-a", worker_id="worker-a", assignment_id="task-a",
                                       idempotency_key="callback:task-a:done", status="done", summary="Finished.")
    assert duplicate["event_id"] == event["event_id"]
    assert list_workers(store, "epoch-a")["assignments"][0]["status"] == "completed"

    packets = drain_callbacks(EpochStore(db), "epoch-a")
    again = drain_callbacks(EpochStore(db), "epoch-a")
    assert packets == again
    assert packets[0]["packet_id"] == f"epoch-a:{event['event_id']}:coordinator"
    assert packets[0]["event"]["payload"]["summary"] == "Finished."
    assert acknowledge_callback(store, "epoch-a", event["event_id"]) is True
    assert acknowledge_callback(store, "epoch-a", event["event_id"]) is False
    assert drain_callbacks(store, "epoch-a") == []


def test_worktree_and_scope_checks_fail_closed(tmp_path):
    root, base = _repo(tmp_path)
    store = EpochStore(tmp_path / "epochs.sqlite3")
    _start(store)
    with pytest.raises(WorkerError, match="absolute path"):
        _register(store, "relative/path", base)
    with pytest.raises(WorkerError, match="parent path"):
        _register(store, root, base, allowed=["../outside"])
    with pytest.raises(WorkerError):
        _register(store, root, "0" * 40)
    (root / "src" / "outside.py").symlink_to(tmp_path / "secret.py")
    subprocess.run(["git", "-C", str(root), "add", "src/outside.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "add symlink fixture"], check=True)
    symlink_base = _git(root, "rev-parse", "HEAD")
    with pytest.raises(WorkerError, match="outside|symlink"):
        _register(store, root, symlink_base, allowed=["src/outside.py"])


def test_assignments_cannot_escape_permissions_or_reserve_past_budget(tmp_path):
    root, base = _repo(tmp_path)
    store = EpochStore(tmp_path / "epochs.sqlite3")
    _start(store)
    _register(store, root, base)
    store.add_hypothesis("epoch-a", "hypothesis-a", "one bounded experiment")

    with pytest.raises(WorkerError, match="subsets"):
        create_assignment(
            store, epoch_id="epoch-a", assignment_id="outside-task", worker_id="worker-a",
            hypothesis_id="hypothesis-a", idempotency_key="assignment:outside-task",
            instructions="Try an unregistered path.", read_paths=["src/**"], write_paths=[],
            token_budget=100, cost_budget_usd="1", wall_budget_seconds=60, prompt_budget=1,
        )
    with pytest.raises(WorkerError, match="token reservation"):
        create_assignment(
            store, epoch_id="epoch-a", assignment_id="too-large-task", worker_id="worker-a",
            hypothesis_id="hypothesis-a", idempotency_key="assignment:too-large-task",
            instructions="Exceed the epoch token envelope.", read_paths=["src/kernel.py"],
            write_paths=["src/kernel.py"], token_budget=501, cost_budget_usd="1",
            wall_budget_seconds=60, prompt_budget=1,
        )
    assert store.get_epoch("epoch-a")["counts"]["assignments"] == 0


def test_crash_recovery_marks_uncertain_without_relaunch_or_budget_refund(tmp_path):
    root, base = _repo(tmp_path)
    db = tmp_path / "epochs.sqlite3"
    store = EpochStore(db)
    _start(store)
    _register(store, root, base)
    assignment = _assign(store)

    class CrashAfterSessionCreate(FakeAdapter):
        def create_session(self, body):
            super().create_session(body)
            raise SystemExit("simulated process death")

    with pytest.raises(SystemExit, match="process death"):
        launch_assignment(store, assignment["assignment_id"], adapter=CrashAfterSessionCreate(),
                          execute=True, confirm_launch=True)
    reserved = EpochStore(db).get_epoch("epoch-a")["reserved"]
    assert recover_launches(EpochStore(db), "epoch-a") == [
        {"assignment_id": "task-a", "status": "launch_uncertain", "session_id": None}
    ]
    assert recover_launches(EpochStore(db), "epoch-a") == []
    after = EpochStore(db).get_epoch("epoch-a")
    assert after["reserved"] == reserved
    assert list_workers(EpochStore(db), "epoch-a")["assignments"][0]["status"] == "launch_uncertain"
    with pytest.raises(WorkerError, match="only planned"):
        launch_assignment(EpochStore(db), "task-a", adapter=FakeAdapter(), execute=True, confirm_launch=True)
