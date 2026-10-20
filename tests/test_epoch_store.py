import json

import pytest

from tools.dreamrsi.epochs import EpochError, EpochStore, main


def budgets(**overrides):
    value = {
        "max_workers": 2,
        "max_hypotheses": 3,
        "max_assignments": 4,
        "max_prompts": 8,
        "max_tokens": 1000,
        "max_cost_usd": "2.500000",
        "max_wall_seconds": 120,
    }
    value.update(overrides)
    return value


def test_epochs_are_explicit_durable_and_resume_without_budget_reset(tmp_path):
    db = tmp_path / "lane" / "epochs.sqlite3"
    store = EpochStore(db)

    started = store.create_epoch("measure-a", budgets())
    assert started["status"] == "active"
    assert started["counts"]["workers"] == 0
    with pytest.raises(EpochError, match="already exists"):
        store.create_epoch("measure-a", budgets())

    paused = store.transition("measure-a", "pause")
    assert paused["status"] == "paused"
    with pytest.raises(EpochError, match="no new work"):
        store.add_hypothesis("measure-a", "h1", "an explicit hypothesis")
    resumed = store.transition("measure-a", "resume")
    assert resumed["status"] == "active"
    assert resumed["dispatch_fence"] == paused["dispatch_fence"] + 1
    assert resumed["budgets"] == started["budgets"]
    assert resumed["reserved"] == started["reserved"]

    reopened = EpochStore(db)
    assert reopened.get_epoch("measure-a")["status"] == "active"
    cancelled = reopened.transition("measure-a", "cancel", "manual stop")
    assert cancelled["status"] == "cancelled"
    assert "does not interrupt already-launched" in cancelled["cancellation_scope"]
    with pytest.raises(EpochError, match="cannot resume"):
        reopened.transition("measure-a", "resume")


def test_events_are_sparse_idempotent_and_cli_returns_int(tmp_path, capsys):
    db = tmp_path / "events.sqlite3"
    assert main([
        "--db", str(db), "start", "--id", "event-run", "--max-workers", "1",
        "--max-hypotheses", "1", "--max-assignments", "2", "--max-prompts", "2",
        "--max-tokens", "100", "--max-cost-usd", "1", "--max-wall-seconds", "30",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["epoch_id"] == "event-run"

    store = EpochStore(db)
    one = store.append_event("event-run", "research.note", "note:1", {"count": 1})
    same = store.append_event("event-run", "research.note", "note:1", {"count": 1})
    assert same["event_id"] == one["event_id"]
    with pytest.raises(EpochError, match="reused"):
        store.append_event("event-run", "research.note", "note:1", {"count": 2})
    events = store.list_events("event-run")
    assert [event["event_type"] for event in events] == ["epoch.started", "research.note"]


def test_budget_validation_rejects_unbounded_or_ambiguous_limits(tmp_path):
    store = EpochStore(tmp_path / "epochs.sqlite3")
    invalid = budgets(max_workers=0)
    with pytest.raises(EpochError, match="max_workers"):
        store.create_epoch("bad", invalid)
    invalid = budgets(extra="unbounded")
    with pytest.raises(EpochError, match="unknown budget"):
        store.create_epoch("bad", invalid)
