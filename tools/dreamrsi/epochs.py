"""Durable, manual research epochs and sparse event packets.

Run with ``python -m tools.dreamrsi.epochs``. State is kept in a lane-owned local
SQLite database (``NINFER_BENCH_EPOCH_DB`` or the user state directory); this
module never opens the submission queue database and imports do no I/O.

Epoch IDs are supplied by the caller. Pausing/resuming preserves every bound
and reservation; cancelling is terminal and fences all later work dispatch.
Events are append-only and idempotent by caller-supplied key. Callback packets
are durable, manually drained records, not host wake-ups or LLM polling.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Iterator


SCHEMA_VERSION = 1
MAX_EVENT_BYTES = 32 * 1024
MAX_COST_MICROUSD = 1_000_000_000_000  # one million USD; protects decimal conversion and bounds policy input
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_EVENT_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,79}$")


class EpochError(ValueError):
    """Invalid or disallowed epoch operation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise EpochError(f"{label} must be 1-80 ASCII letters, digits, dots, underscores, or hyphens")
    return value


def cost_to_microusd(value: Any, label: str = "cost") -> int:
    raw = str(value)
    if len(raw) > 128:
        raise EpochError(f"{label} is too long")
    try:
        amount = Decimal(raw)
    except (InvalidOperation, ValueError):
        raise EpochError(f"{label} must be a finite decimal USD amount") from None
    if not amount.is_finite() or amount < 0 or amount.as_tuple().exponent < -6:
        raise EpochError(f"{label} must be non-negative with at most six decimal places")
    if amount > Decimal("1000000"):
        raise EpochError(f"{label} may not exceed 1000000 USD")
    micros = int(amount * 1_000_000)
    if micros > MAX_COST_MICROUSD:
        raise EpochError(f"{label} exceeds the maximum supported budget")
    return micros


def microusd_to_string(value: int) -> str:
    return f"{value // 1_000_000}.{value % 1_000_000:06d}"


def normalize_budgets(budgets: dict[str, Any]) -> dict[str, int]:
    """Validate the hard scheduler bounds for one manually-created epoch."""
    expected = {
        "max_workers": (1, 32),
        "max_hypotheses": (1, 10_000),
        "max_assignments": (1, 10_000),
        "max_prompts": (1, 100_000),
        "max_tokens": (1, 10**12),
        "max_wall_seconds": (1, 365 * 24 * 60 * 60),
    }
    result: dict[str, int] = {}
    if not isinstance(budgets, dict):
        raise EpochError("budgets must be an object")
    unknown = set(budgets) - (set(expected) | {"max_cost_usd"})
    missing = (set(expected) | {"max_cost_usd"}) - set(budgets)
    if unknown:
        raise EpochError(f"unknown budget fields: {', '.join(sorted(unknown))}")
    if missing:
        raise EpochError(f"missing budget fields: {', '.join(sorted(missing))}")
    for key, (minimum, maximum) in expected.items():
        value = budgets[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise EpochError(f"{key} must be an integer from {minimum} through {maximum}")
        result[key] = value
    cost = cost_to_microusd(budgets["max_cost_usd"], "max_cost_usd")
    if cost <= 0:
        raise EpochError("max_cost_usd must be greater than zero")
    result["max_cost_microusd"] = cost
    return result


def default_db_path() -> Path:
    configured = os.environ.get("NINFER_BENCH_EPOCH_DB")
    if configured:
        if configured == ":memory:":
            raise EpochError("an on-disk database is required for durable epoch state")
        return Path(configured).expanduser().absolute()
    state_home = Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")).expanduser()
    return (state_home / "ninfer-bench" / "epochs.sqlite3").absolute()


class EpochStore:
    """SQLite owner for epochs, hypotheses, worker records, assignments and events."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is not None and str(db_path) == ":memory:":
            raise EpochError("an on-disk database is required for durable epoch state")
        self.path = Path(db_path).expanduser().absolute() if db_path is not None else default_db_path()
        if str(self.path) == ":memory:":
            raise EpochError("an on-disk database is required for durable epoch state")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=40, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA busy_timeout = 40000")
        return con

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        con = self._connect()
        try:
            yield con
        finally:
            con.close()

    def _initialize(self) -> None:
        with self.connection() as con:
            con.execute("PRAGMA journal_mode = WAL")
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise EpochError(f"unsupported epoch database schema version {version}")
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS epochs (
                    epoch_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('active','paused','cancelled','completed')),
                    budgets_json TEXT NOT NULL,
                    reserved_tokens INTEGER NOT NULL DEFAULT 0,
                    reserved_cost_microusd INTEGER NOT NULL DEFAULT 0,
                    reserved_wall_seconds INTEGER NOT NULL DEFAULT 0,
                    reserved_prompts INTEGER NOT NULL DEFAULT 0,
                    dispatch_fence INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    cancel_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS hypotheses (
                    epoch_id TEXT NOT NULL REFERENCES epochs(epoch_id),
                    hypothesis_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(epoch_id, hypothesis_id)
                );
                CREATE TABLE IF NOT EXISTS workers (
                    epoch_id TEXT NOT NULL REFERENCES epochs(epoch_id),
                    worker_id TEXT NOT NULL,
                    worktree TEXT NOT NULL,
                    base_commit TEXT NOT NULL,
                    model TEXT NOT NULL,
                    role TEXT NOT NULL,
                    allowed_paths_json TEXT NOT NULL,
                    permissions_json TEXT NOT NULL,
                    stop_conditions_json TEXT NOT NULL,
                    callback_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(epoch_id, worker_id),
                    UNIQUE(epoch_id, worktree)
                );
                CREATE TABLE IF NOT EXISTS assignments (
                    assignment_id TEXT PRIMARY KEY,
                    epoch_id TEXT NOT NULL REFERENCES epochs(epoch_id),
                    worker_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    read_paths_json TEXT NOT NULL,
                    write_paths_json TEXT NOT NULL,
                    token_budget INTEGER NOT NULL,
                    cost_budget_microusd INTEGER NOT NULL,
                    wall_budget_seconds INTEGER NOT NULL,
                    prompt_budget INTEGER NOT NULL,
                    prompts_used INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK(status IN ('planned','launching','launched','completed','cancelled','launch_uncertain','launch_failed')),
                    launch_token TEXT,
                    launch_fence INTEGER,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(epoch_id, worker_id) REFERENCES workers(epoch_id, worker_id),
                    FOREIGN KEY(epoch_id, hypothesis_id) REFERENCES hypotheses(epoch_id, hypothesis_id),
                    UNIQUE(epoch_id, idempotency_key)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_assignment_per_worker
                    ON assignments(epoch_id, worker_id)
                    WHERE status IN ('planned','launching','launched');
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    epoch_id TEXT NOT NULL REFERENCES epochs(epoch_id),
                    event_type TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    worker_id TEXT,
                    recipient TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(epoch_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS callback_acks (
                    event_id INTEGER NOT NULL REFERENCES events(event_id),
                    recipient TEXT NOT NULL,
                    acked_at TEXT NOT NULL,
                    PRIMARY KEY(event_id, recipient)
                );
                """
            )
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize mutations; callers may use this to fence external dispatch."""
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def read_connection(self) -> sqlite3.Connection:
        """Return a short-lived read-only-by-convention connection to this DB."""
        return self._connect()

    @staticmethod
    def require_epoch(con: sqlite3.Connection, epoch_id: str) -> sqlite3.Row:
        row = con.execute("SELECT * FROM epochs WHERE epoch_id = ?", (epoch_id,)).fetchone()
        if row is None:
            raise EpochError(f"unknown epoch: {epoch_id}")
        return row

    @staticmethod
    def require_dispatchable(con: sqlite3.Connection, epoch_id: str) -> sqlite3.Row:
        row = EpochStore.require_epoch(con, epoch_id)
        if row["status"] != "active":
            raise EpochError(f"epoch {epoch_id} is {row['status']}; no new work may be dispatched")
        return row

    def create_epoch(self, epoch_id: str, budgets: dict[str, Any]) -> dict[str, Any]:
        epoch_id = _identifier(epoch_id, "epoch_id")
        normalized = normalize_budgets(budgets)
        now = utc_now()
        with self.transaction() as con:
            try:
                con.execute(
                    "INSERT INTO epochs(epoch_id,status,budgets_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                    (epoch_id, "active", canonical_json(normalized), now, now),
                )
            except sqlite3.IntegrityError:
                raise EpochError(f"epoch ID already exists: {epoch_id}") from None
            self._append_event_tx(con, epoch_id, "epoch.started", f"state:active:0", None, None,
                                  {"status": "active", "budgets": normalized})
            return self._epoch_json(con, epoch_id)

    def get_epoch(self, epoch_id: str) -> dict[str, Any]:
        with self.connection() as con:
            return self._epoch_json(con, epoch_id)

    def list_epochs(self) -> list[dict[str, Any]]:
        with self.connection() as con:
            return [self._epoch_json(con, row["epoch_id"]) for row in con.execute(
                "SELECT epoch_id FROM epochs ORDER BY created_at, epoch_id")]

    def _epoch_json(self, con: sqlite3.Connection, epoch_id: str) -> dict[str, Any]:
        row = self.require_epoch(con, epoch_id)
        budgets = json.loads(row["budgets_json"])
        budgets["max_cost_usd"] = microusd_to_string(budgets.pop("max_cost_microusd"))
        counts = {
            "workers": con.execute("SELECT COUNT(*) FROM workers WHERE epoch_id=?", (epoch_id,)).fetchone()[0],
            "hypotheses": con.execute("SELECT COUNT(*) FROM hypotheses WHERE epoch_id=?", (epoch_id,)).fetchone()[0],
            "assignments": con.execute("SELECT COUNT(*) FROM assignments WHERE epoch_id=?", (epoch_id,)).fetchone()[0],
            "active_assignments": con.execute(
                "SELECT COUNT(*) FROM assignments WHERE epoch_id=? AND status IN ('planned','launching','launched')",
                (epoch_id,),
            ).fetchone()[0],
        }
        return {
            "epoch_id": row["epoch_id"], "status": row["status"], "budgets": budgets,
            "reserved": {
                "tokens": row["reserved_tokens"], "cost_usd": microusd_to_string(row["reserved_cost_microusd"]),
                "wall_seconds": row["reserved_wall_seconds"], "prompts": row["reserved_prompts"],
            },
            "counts": counts, "dispatch_fence": row["dispatch_fence"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "cancel_reason": row["cancel_reason"],
            "cancellation_scope": "stops new dispatch; does not interrupt already-launched OpenCode sessions",
        }

    def transition(self, epoch_id: str, action: str, reason: str | None = None) -> dict[str, Any]:
        epoch_id = _identifier(epoch_id, "epoch_id")
        now = utc_now()
        with self.transaction() as con:
            row = self.require_epoch(con, epoch_id)
            state = row["status"]
            transitions = {"pause": ("active", "paused"), "resume": ("paused", "active")}
            if action == "cancel":
                if state == "cancelled":
                    return self._epoch_json(con, epoch_id)
                if state == "completed":
                    raise EpochError("completed epochs cannot be cancelled")
                new_state = "cancelled"
                reason = (reason or "cancelled by user").strip()
                if not reason or len(reason) > 1000:
                    raise EpochError("cancel reason must contain 1-1000 characters")
                con.execute(
                    "UPDATE epochs SET status=?,updated_at=?,dispatch_fence=dispatch_fence+1,cancel_reason=? WHERE epoch_id=?",
                    (new_state, now, reason, epoch_id),
                )
                con.execute(
                    "UPDATE assignments SET status='cancelled',updated_at=? WHERE epoch_id=? AND status='planned'",
                    (now, epoch_id),
                )
                # A session-creation request is inert until a prompt is sent. A
                # launch currently holding the dispatch transaction completes
                # before this cancellation can commit.
                con.execute(
                    "UPDATE assignments SET status='cancelled',updated_at=? WHERE epoch_id=? AND status='launching'",
                    (now, epoch_id),
                )
                self._append_event_tx(con, epoch_id, "epoch.cancelled", f"state:cancelled:{row['dispatch_fence'] + 1}",
                                      None, None, {"status": new_state, "reason": reason})
            else:
                if action not in transitions:
                    raise EpochError(f"unsupported epoch action: {action}")
                expected, new_state = transitions[action]
                if state != expected:
                    raise EpochError(f"cannot {action} epoch in {state} state")
                con.execute(
                    "UPDATE epochs SET status=?,updated_at=?,dispatch_fence=dispatch_fence+1 WHERE epoch_id=?",
                    (new_state, now, epoch_id),
                )
                fence = row["dispatch_fence"] + 1
                self._append_event_tx(con, epoch_id, f"epoch.{new_state}", f"state:{new_state}:{fence}",
                                      None, None, {"status": new_state, "dispatch_fence": fence})
            return self._epoch_json(con, epoch_id)

    def add_hypothesis(self, epoch_id: str, hypothesis_id: str, description: str) -> dict[str, Any]:
        epoch_id = _identifier(epoch_id, "epoch_id")
        hypothesis_id = _identifier(hypothesis_id, "hypothesis_id")
        if not isinstance(description, str) or not description.strip() or len(description) > 4000:
            raise EpochError("hypothesis description must contain 1-4000 characters")
        with self.transaction() as con:
            epoch = self.require_dispatchable(con, epoch_id)
            budgets = json.loads(epoch["budgets_json"])
            existing = con.execute("SELECT * FROM hypotheses WHERE epoch_id=? AND hypothesis_id=?",
                                   (epoch_id, hypothesis_id)).fetchone()
            if existing:
                if existing["description"] != description.strip():
                    raise EpochError("hypothesis ID already exists with different content")
                return dict(existing)
            count = con.execute("SELECT COUNT(*) FROM hypotheses WHERE epoch_id=?", (epoch_id,)).fetchone()[0]
            if count >= budgets["max_hypotheses"]:
                raise EpochError("epoch hypothesis bound is exhausted")
            now = utc_now()
            con.execute("INSERT INTO hypotheses VALUES(?,?,?,?)", (epoch_id, hypothesis_id, description.strip(), now))
            return dict(con.execute("SELECT * FROM hypotheses WHERE epoch_id=? AND hypothesis_id=?",
                                    (epoch_id, hypothesis_id)).fetchone())

    def _append_event_tx(self, con: sqlite3.Connection, epoch_id: str, event_type: str,
                         idempotency_key: str, worker_id: str | None, recipient: str | None,
                         payload: dict[str, Any]) -> dict[str, Any]:
        if not _EVENT_RE.fullmatch(event_type):
            raise EpochError("event_type must be a lowercase identifier")
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200:
            raise EpochError("idempotency_key must contain 1-200 characters")
        if not isinstance(payload, dict):
            raise EpochError("event payload must be an object")
        try:
            body = canonical_json(payload)
        except (TypeError, ValueError):
            raise EpochError("event payload must contain only finite JSON values") from None
        if len(body.encode("utf-8")) > MAX_EVENT_BYTES:
            raise EpochError(f"event payload exceeds {MAX_EVENT_BYTES} bytes")
        previous = con.execute("SELECT * FROM events WHERE epoch_id=? AND idempotency_key=?",
                               (epoch_id, idempotency_key)).fetchone()
        if previous:
            same = (previous["event_type"] == event_type and previous["worker_id"] == worker_id
                    and previous["recipient"] == recipient and previous["payload_json"] == body)
            if not same:
                raise EpochError("event idempotency key was reused with different content")
            return self._event_json(previous)
        now = utc_now()
        cur = con.execute(
            "INSERT INTO events(epoch_id,event_type,idempotency_key,worker_id,recipient,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (epoch_id, event_type, idempotency_key, worker_id, recipient, body, now),
        )
        return self._event_json(con.execute("SELECT * FROM events WHERE event_id=?", (cur.lastrowid,)).fetchone())

    @staticmethod
    def _event_json(row: sqlite3.Row) -> dict[str, Any]:
        return {"event_id": row["event_id"], "epoch_id": row["epoch_id"], "event_type": row["event_type"],
                "idempotency_key": row["idempotency_key"], "worker_id": row["worker_id"],
                "recipient": row["recipient"], "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"]}

    def append_event(self, epoch_id: str, event_type: str, idempotency_key: str, payload: dict[str, Any],
                     *, worker_id: str | None = None, recipient: str | None = None) -> dict[str, Any]:
        epoch_id = _identifier(epoch_id, "epoch_id")
        with self.transaction() as con:
            self.require_epoch(con, epoch_id)
            if worker_id is not None:
                _identifier(worker_id, "worker_id")
            return self._append_event_tx(con, epoch_id, event_type, idempotency_key, worker_id, recipient, payload)

    def list_events(self, epoch_id: str, after: int = 0) -> list[dict[str, Any]]:
        epoch_id = _identifier(epoch_id, "epoch_id")
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise EpochError("after must be a non-negative event ID")
        with self.connection() as con:
            self.require_epoch(con, epoch_id)
            return [self._event_json(row) for row in con.execute(
                "SELECT * FROM events WHERE epoch_id=? AND event_id>? ORDER BY event_id", (epoch_id, after))]

    def drain_callbacks(self, epoch_id: str, recipient: str = "coordinator", limit: int = 100) -> list[dict[str, Any]]:
        """Return stable pending packets. Does not call or wake another process."""
        epoch_id = _identifier(epoch_id, "epoch_id")
        if not isinstance(recipient, str) or not recipient or len(recipient) > 120:
            raise EpochError("recipient must contain 1-120 characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise EpochError("limit must be from 1 through 1000")
        with self.connection() as con:
            self.require_epoch(con, epoch_id)
            rows = con.execute(
                """SELECT e.* FROM events e LEFT JOIN callback_acks a
                   ON a.event_id=e.event_id AND a.recipient=e.recipient
                   WHERE e.epoch_id=? AND e.recipient=? AND e.event_type='worker.callback'
                     AND a.event_id IS NULL ORDER BY e.event_id LIMIT ?""",
                (epoch_id, recipient, limit),
            )
            packets = []
            for row in rows:
                event = self._event_json(row)
                packets.append({"packet_id": f"{epoch_id}:{event['event_id']}:{recipient}", "event": event})
            return packets

    def acknowledge_callback(self, epoch_id: str, event_id: int, recipient: str = "coordinator") -> bool:
        epoch_id = _identifier(epoch_id, "epoch_id")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id <= 0:
            raise EpochError("event_id must be positive")
        with self.transaction() as con:
            event = con.execute("SELECT * FROM events WHERE event_id=? AND epoch_id=? AND event_type='worker.callback'",
                                (event_id, epoch_id)).fetchone()
            if event is None or event["recipient"] != recipient:
                raise EpochError("unknown callback packet for this recipient")
            cur = con.execute("INSERT OR IGNORE INTO callback_acks(event_id,recipient,acked_at) VALUES(?,?,?)",
                              (event_id, recipient, utc_now()))
            return cur.rowcount == 1


def _emit_state_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and manage manual, durable research epochs.")
    parser.add_argument("--db", help="lane-owned SQLite DB (default: NINFER_BENCH_EPOCH_DB or user state directory)")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="create a new epoch; IDs are explicit and never auto-generated")
    start.add_argument("--id", required=True)
    start.add_argument("--max-workers", type=int, required=True)
    start.add_argument("--max-hypotheses", type=int, required=True)
    start.add_argument("--max-assignments", type=int, required=True)
    start.add_argument("--max-prompts", type=int, required=True)
    start.add_argument("--max-tokens", type=int, required=True)
    start.add_argument("--max-cost-usd", required=True)
    start.add_argument("--max-wall-seconds", type=int, required=True)

    for name, help_text in (("status", "show one epoch"), ("pause", "stop new dispatch without resetting bounds"),
                            ("resume", "resume a paused epoch without resetting bounds")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--id", required=True)
    cancel = sub.add_parser("cancel", help="terminally cancel an epoch and fence future dispatch")
    cancel.add_argument("--id", required=True)
    cancel.add_argument("--reason", default="cancelled by user")

    event = sub.add_parser("event", help="append one idempotent event")
    event.add_argument("--id", required=True, dest="epoch_id")
    event.add_argument("--type", required=True, dest="event_type")
    event.add_argument("--key", required=True, dest="idempotency_key")
    event.add_argument("--payload", required=True, help="JSON object")

    events = sub.add_parser("events", help="list sparse durable events")
    events.add_argument("--id", required=True, dest="epoch_id")
    events.add_argument("--after", type=int, default=0)
    drain = sub.add_parser("callback-drain", help="manually print pending callback packets; no host wake occurs")
    drain.add_argument("--id", required=True, dest="epoch_id")
    drain.add_argument("--recipient", default="coordinator")
    drain.add_argument("--limit", type=int, default=100)
    ack = sub.add_parser("callback-ack", help="acknowledge a manually consumed callback packet")
    ack.add_argument("--id", required=True, dest="epoch_id")
    ack.add_argument("--event-id", type=int, required=True)
    ack.add_argument("--recipient", default="coordinator")

    hypothesis = sub.add_parser("hypothesis", help="add one bounded hypothesis to an active epoch")
    hypothesis.add_argument("--id", required=True, dest="epoch_id")
    hypothesis.add_argument("--hypothesis-id", required=True)
    hypothesis.add_argument("--description", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = EpochStore(args.db)
        if args.command == "start":
            result = store.create_epoch(args.id, {
                "max_workers": args.max_workers, "max_hypotheses": args.max_hypotheses,
                "max_assignments": args.max_assignments, "max_prompts": args.max_prompts,
                "max_tokens": args.max_tokens, "max_cost_usd": args.max_cost_usd,
                "max_wall_seconds": args.max_wall_seconds,
            })
        elif args.command == "status":
            result = store.get_epoch(args.id)
        elif args.command in ("pause", "resume", "cancel"):
            result = store.transition(args.id, args.command, getattr(args, "reason", None))
        elif args.command == "event":
            payload = json.loads(args.payload)
            if not isinstance(payload, dict):
                raise EpochError("event payload must be a JSON object")
            result = store.append_event(args.epoch_id, args.event_type, args.idempotency_key, payload)
        elif args.command == "events":
            result = store.list_events(args.epoch_id, args.after)
        elif args.command == "callback-drain":
            result = store.drain_callbacks(args.epoch_id, args.recipient, args.limit)
        elif args.command == "callback-ack":
            result = {"acknowledged": store.acknowledge_callback(args.epoch_id, args.event_id, args.recipient)}
        elif args.command == "hypothesis":
            result = store.add_hypothesis(args.epoch_id, args.hypothesis_id, args.description)
        else:
            raise EpochError(f"unsupported command: {args.command}")
        _emit_state_json(result)
        return 0
    except (EpochError, sqlite3.Error, OSError, json.JSONDecodeError) as exc:
        print(f"epochs: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
