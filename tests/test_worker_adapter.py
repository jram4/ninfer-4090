import json
import subprocess

import pytest

from tools.dreamrsi.epochs import EpochError, EpochStore
from tools.dreamrsi.workers import (
    OpenCodeAdapter,
    WorkerError,
    acknowledge_callback,
    build_launch_plan,
    continue_assignment,
    collect_terminal_reports,
    create_assignment,
    dispatch_callback_outbox,
    drain_callbacks,
    launch_assignment,
    list_workers,
    opencode_permission_rules,
    record_worker_callback,
    recover_launches,
    register_worker,
    watch_terminal_reports,
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


class FakeCallbackAdapter:
    def __init__(self):
        self.messages = {}
        self.sent = []

    def get_message(self, session_id, message_id):
        return self.messages.get((session_id, message_id))

    def send_prompt(self, session_id, body):
        self.sent.append((session_id, dict(body)))
        self.messages[(session_id, body["id"])] = {"id": body["id"], "type": "user", "text": body["text"]}
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


def test_opencode_callback_uses_prompt_and_message_reconciliation_routes(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[-1] == "session.prompt":
            return subprocess.CompletedProcess(argv, 0, '{"data":{"id":"inbox_1"}}', "")
        if argv[-1] == "session.message.get":
            return subprocess.CompletedProcess(
                argv, 0, '{"data":{"id":"msg_stable","type":"user","text":"packet"}}', ""
            )
        raise AssertionError(f"unexpected OpenCode operation: {argv[-1]}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = OpenCodeAdapter("opencode-fake", "http://127.0.0.1:1")
    body = {"id": "msg_stable", "text": "packet", "resume": True}
    adapter.send_prompt("ses_parent123456", body)
    assert adapter.get_message("ses_parent123456", "msg_stable")["text"] == "packet"
    assert calls[0][-1] == "session.prompt"
    assert "sessionID=ses_parent123456" in calls[0]
    assert calls[0][calls[0].index("--data") + 1] == json.dumps(body, separators=(",", ":"),
                                                                 ensure_ascii=False, sort_keys=True)
    assert calls[1][-1] == "session.message.get"
    assert "messageID=msg_stable" in calls[1]
    assert "sessionID=ses_parent123456" in calls[1]

    def missing_message(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 1, '{"_tag":"MessageNotFoundError","sessionID":"ses_parent123456",'
            '"messageID":"msg_missing","message":"not found"}', "HTTP 404 Not Found",
        )

    monkeypatch.setattr(subprocess, "run", missing_message)
    assert adapter.get_message("ses_parent123456", "msg_missing") is None


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


def test_terminal_report_collector_dedupes_and_wakes_only_through_fake_outbox(tmp_path):
    root, base = _repo(tmp_path)
    (root / "reports").mkdir()
    db = tmp_path / "epochs.sqlite3"
    store = EpochStore(db)
    _start(store)
    report_path = "reports/terminal.json"
    register_worker(
        store, epoch_id="epoch-a", worker_id="worker-auto", worktree=root, base_commit=base,
        model="fnn/qwen3.8-27b#medium", role="build",
        allowed_files=["src/kernel.py", report_path],
        permissions={"read_paths": ["src/kernel.py", report_path],
                     "write_paths": ["src/kernel.py", report_path], "shell": False, "network": False},
        stop_conditions=["one bounded task is complete"],
        callback={"mode": "opencode_session", "recipient": "ses_parent123456", "report_path": report_path},
    )
    store.add_hypothesis("epoch-a", "hypothesis-auto", "report a bounded result")
    assignment = create_assignment(
        store, epoch_id="epoch-a", assignment_id="task-auto", worker_id="worker-auto",
        hypothesis_id="hypothesis-auto", idempotency_key="assignment:task-auto",
        instructions="Make a narrow change and report the result.",
        read_paths=["src/kernel.py", report_path], write_paths=["src/kernel.py", report_path],
        token_budget=100, cost_budget_usd="1", wall_budget_seconds=60, prompt_budget=1,
    )
    plan = build_launch_plan(store, assignment["assignment_id"])
    assert report_path in plan["session_prompt_template"]["body"]["text"]
    assert {"action": "edit", "resource": report_path, "effect": "allow"} in plan["session_create"]["body"]["permissions"]
    launch_assignment(store, assignment["assignment_id"], adapter=FakeAdapter(), execute=True, confirm_launch=True)

    report = {"schema_version": 1, "epoch_id": "epoch-a", "worker_id": "worker-auto",
              "assignment_id": "task-auto", "status": "done", "summary": "Finished one scoped change.",
              "evidence": ["src/kernel.py changed"]}
    (root / report_path).write_text(json.dumps(report), encoding="utf-8")
    first = collect_terminal_reports(EpochStore(db), "epoch-a")
    second = collect_terminal_reports(EpochStore(db), "epoch-a")
    assert first["reports"][0]["status"] == "collected"
    assert second["reports"][0]["event_id"] == first["reports"][0]["event_id"]

    fake = FakeCallbackAdapter()
    preview = dispatch_callback_outbox(EpochStore(db), "epoch-a", adapter=fake)
    assert preview["dry_run"] is True and fake.sent == []
    watched = watch_terminal_reports(EpochStore(db), "epoch-a", max_runs=1, dispatch=True,
                                     adapter=fake, execute=True, confirm_delivery=True)
    assert watched["runs"][0]["delivery"]["delivered"] == 1
    assert len(fake.sent) == 1
    recipient, body = fake.sent[0]
    assert recipient == "ses_parent123456"
    assert body["resume"] is True
    assert body["id"].startswith("msg_")
    assert "Finished one scoped change." in body["text"]
    assert "src/kernel.py changed" in body["text"]
    assert drain_callbacks(EpochStore(db), "epoch-a", "ses_parent123456") == []
    with pytest.raises(EpochError, match="only after API admission"):
        acknowledge_callback(EpochStore(db), "epoch-a", first["reports"][0]["event_id"], "ses_parent123456")
    assert dispatch_callback_outbox(EpochStore(db), "epoch-a", adapter=fake,
                                    execute=True, confirm_delivery=True)["delivered"] == 0
    assert len(fake.sent) == 1


def test_callback_outbox_recovers_admitted_but_unacked_and_cancel_fences_new_send(tmp_path):
    root, base = _repo(tmp_path)
    (root / "reports").mkdir()
    db = tmp_path / "epochs.sqlite3"
    store = EpochStore(db)
    _start(store)
    report_path = "reports/terminal.json"
    register_worker(
        store, epoch_id="epoch-a", worker_id="worker-auto", worktree=root, base_commit=base,
        model="fnn/qwen3.8-27b#medium", role="build",
        allowed_files=["src/kernel.py", report_path],
        permissions={"read_paths": ["src/kernel.py", report_path],
                     "write_paths": ["src/kernel.py", report_path], "shell": False, "network": False},
        stop_conditions=["one bounded task is complete"],
        callback={"mode": "opencode_session", "recipient": "ses_parent123456", "report_path": report_path},
    )
    store.add_hypothesis("epoch-a", "hypothesis-auto", "report a bounded result")
    create_assignment(
        store, epoch_id="epoch-a", assignment_id="task-auto", worker_id="worker-auto",
        hypothesis_id="hypothesis-auto", idempotency_key="assignment:task-auto",
        instructions="Work and report.", read_paths=["src/kernel.py", report_path],
        write_paths=["src/kernel.py", report_path], token_budget=100, cost_budget_usd="1",
        wall_budget_seconds=60, prompt_budget=1,
    )
    # A terminal callback requires a launched assignment.
    launch_assignment(store, "task-auto", adapter=FakeAdapter(), execute=True, confirm_launch=True)
    (root / report_path).write_text(json.dumps({
        "schema_version": 1, "epoch_id": "epoch-a", "worker_id": "worker-auto",
        "assignment_id": "task-auto", "status": "done", "summary": "Complete.", "evidence": [],
    }), encoding="utf-8")
    collect_terminal_reports(store, "epoch-a")

    fake = FakeCallbackAdapter()

    class LostAdmissionAck(FakeCallbackAdapter):
        def send_prompt(self, session_id, body):
            super().send_prompt(session_id, body)
            raise SystemExit("crash after server admission")

    lost = LostAdmissionAck()
    with pytest.raises(SystemExit, match="server admission"):
        dispatch_callback_outbox(store, "epoch-a", adapter=lost, execute=True, confirm_delivery=True)
    # Simulate lease expiry after restart. GET finds the deterministic admitted
    # message, so the collector acknowledges it without sending a duplicate.
    with store.transaction() as con:
        con.execute("UPDATE callback_outbox SET lease_until='2000-01-01T00:00:00Z' WHERE epoch_id='epoch-a'")
    recovered = dispatch_callback_outbox(EpochStore(db), "epoch-a", adapter=lost,
                                         execute=True, confirm_delivery=True)
    assert recovered["results"][0]["status"] == "reconciled_admission"
    assert len(lost.sent) == 1

    cancel_dir = tmp_path / "cancel-case"
    cancel_dir.mkdir()
    root2, base2 = _repo(cancel_dir)
    (root2 / "reports").mkdir()
    cancel_store = EpochStore(tmp_path / "cancel.sqlite3")
    _start(cancel_store)
    register_worker(
        cancel_store, epoch_id="epoch-a", worker_id="worker-auto", worktree=root2, base_commit=base2,
        model="fnn/qwen3.8-27b#medium", role="build",
        allowed_files=["src/kernel.py", report_path],
        permissions={"read_paths": ["src/kernel.py", report_path],
                     "write_paths": ["src/kernel.py", report_path], "shell": False, "network": False},
        stop_conditions=["one bounded task is complete"],
        callback={"mode": "opencode_session", "recipient": "ses_parent123456", "report_path": report_path},
    )
    cancel_store.add_hypothesis("epoch-a", "hypothesis-auto", "report a bounded result")
    # An explicitly recorded callback is enough to create a pending outbox row.
    record_worker_callback(cancel_store, epoch_id="epoch-a", worker_id="worker-auto",
                           idempotency_key="manual-callback", status="done", summary="Pending.")
    cancel_store.transition("epoch-a", "pause")
    paused = dispatch_callback_outbox(cancel_store, "epoch-a", adapter=fake,
                                      execute=True, confirm_delivery=True)
    assert "epoch is paused" in paused["epoch_blocked"]
    assert fake.sent == []
    cancel_store.transition("epoch-a", "resume")
    cancel_store.transition("epoch-a", "cancel", "operator stop")
    no_send = dispatch_callback_outbox(cancel_store, "epoch-a", adapter=fake,
                                       execute=True, confirm_delivery=True)
    assert no_send["delivered"] == 0
    assert fake.sent == []
