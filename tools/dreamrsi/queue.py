"""Crash-safe SQLite FIFO with fenced leases and explicit ingress backpressure."""

from __future__ import annotations

from datetime import datetime
import argparse
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import time
from typing import Any
from contextlib import contextmanager

from .contracts import read_json, sha256_file, validate_receipt, validate_submission


DEFAULT_MAX_PENDING = 128


class QueueError(RuntimeError):
    """Base class for queue contract failures."""


class QueueFullError(QueueError):
    """The bounded ingress queue has no capacity for another submission."""


class ActiveWorkerError(QueueError):
    """The epoch already has an active submission for this worker."""


class StaleClaimError(QueueError):
    """A lease token is stale, expired, or no longer owns this submission."""


class QueueConflictError(QueueError):
    """An idempotency key was reused for different submission content."""


def _now(value: float | int | datetime | None = None) -> float:
    if value is None:
        return time.time()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("now datetime must be timezone-aware")
        return value.timestamp()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("now must be a finite timestamp")
    return float(value)


def _lease(lease_seconds: float) -> float:
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)):
        raise ValueError("lease_seconds must be positive")
    if not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive and finite")
    return float(lease_seconds)


class Queue:
    """Local authoritative SQLite queue; do not place its database on NFS."""

    def __init__(self, db_path: str | os.PathLike[str], *, max_pending: int = DEFAULT_MAX_PENDING):
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        self.db_path = Path(db_path).expanduser().absolute()
        self.max_pending = max_pending
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            if connection.in_transaction:
                connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS submissions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT NOT NULL UNIQUE,
                    epoch_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    submission_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('queued','running','cancel_requested','cancelled','completed')),
                    attempt INTEGER NOT NULL DEFAULT 0,
                    claim_token TEXT,
                    claim_worker TEXT,
                    lease_expires_at REAL,
                    receipt_json TEXT,
                    cancel_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_epoch_worker
                    ON submissions(epoch_id, worker_id)
                    WHERE status IN ('queued','running','cancel_requested');
                CREATE UNIQUE INDEX IF NOT EXISTS unique_submission_idempotency
                    ON submissions(epoch_id, worker_id, idempotency_key);
                CREATE INDEX IF NOT EXISTS fifo_queued ON submissions(status, sequence);
                """
            )
        try:
            os.chmod(self.db_path, 0o600)
        except FileNotFoundError:
            pass

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "status": row["status"],
            "submission": json.loads(row["submission_json"]),
            "receipt": json.loads(row["receipt_json"]) if row["receipt_json"] is not None else None,
            "attempt": row["attempt"],
            # The claim identity lets a worker notice that its claim was
            # reclaimed by another worker.
            "claim_token": row["claim_token"],
            "claim_worker": row["claim_worker"],
        }

    @staticmethod
    def _decode_submission(row: sqlite3.Row) -> dict[str, Any]:
        return json.loads(row["submission_json"])

    def enqueue(self, submission: dict[str, Any]) -> str:
        normalized = validate_submission(submission)
        archive_path = Path(normalized["source"]["archive_path"])
        if not archive_path.is_file() or sha256_file(archive_path) != normalized["source"]["archive_sha256"]:
            raise QueueError("frozen source archive is missing or its hash does not match submission")
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        now = time.time()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            by_id = connection.execute(
                "SELECT * FROM submissions WHERE submission_id = ?",
                (normalized["submission_id"],),
            ).fetchone()
            if by_id is not None:
                old = self._decode_submission(by_id)
                # created_at and archive_path are transport metadata, not identity.
                old_identity = dict(old)
                new_identity = dict(normalized)
                old_identity.pop("created_at", None)
                new_identity.pop("created_at", None)
                old_identity["source"].pop("archive_path", None)
                new_identity["source"].pop("archive_path", None)
                if old_identity != new_identity:
                    raise QueueConflictError("submission_id is already bound to different content")
                return normalized["submission_id"]
            key_row = connection.execute(
                "SELECT * FROM submissions WHERE epoch_id = ? AND worker_id = ? AND idempotency_key = ?",
                (normalized["epoch_id"], normalized["worker_id"], normalized["idempotency_key"]),
            ).fetchone()
            if key_row is not None:
                old = self._decode_submission(key_row)
                old_identity = dict(old)
                new_identity = dict(normalized)
                old_identity.pop("created_at", None)
                new_identity.pop("created_at", None)
                old_identity["source"].pop("archive_path", None)
                new_identity["source"].pop("archive_path", None)
                if old_identity == new_identity:
                    return str(key_row["submission_id"])
                raise QueueConflictError("idempotency_key is already bound to different submission content")
            pending = connection.execute(
                "SELECT count(*) FROM submissions WHERE status IN ('queued','running','cancel_requested')"
            ).fetchone()[0]
            if pending >= self.max_pending:
                raise QueueFullError(f"queue is full ({self.max_pending} active submissions)")
            try:
                connection.execute(
                    """INSERT INTO submissions (
                        submission_id, epoch_id, worker_id, idempotency_key,
                        submission_json, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)""",
                    (
                        normalized["submission_id"], normalized["epoch_id"], normalized["worker_id"],
                        normalized["idempotency_key"], payload, now, now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                existing = connection.execute(
                    "SELECT submission_id FROM submissions WHERE epoch_id = ? AND worker_id = ? AND status IN ('queued','running','cancel_requested')",
                    (normalized["epoch_id"], normalized["worker_id"]),
                ).fetchone()
                if existing is not None:
                    raise ActiveWorkerError(
                        f"worker {normalized['worker_id']!r} already has an active submission in epoch {normalized['epoch_id']!r}"
                    ) from exc
                raise QueueConflictError("submission conflicts with an existing queue record") from exc
        return normalized["submission_id"]

    def get(self, submission_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM submissions WHERE submission_id = ?", (submission_id,)
            ).fetchone()
        if row is None:
            raise KeyError(submission_id)
        return self._record(row)

    def claim(
        self,
        worker_id: str,
        lease_seconds: float = 900,
    ) -> dict[str, Any] | None:
        if not isinstance(worker_id, str) or not worker_id or worker_id != worker_id.strip():
            raise ValueError("worker_id must be a nonempty trimmed string")
        duration = _lease(lease_seconds)
        # Expired leases are immediately fenced and made available to the next
        # FIFO claimant, even if no separate maintenance loop ran after a crash.
        self.recover()
        token = secrets.token_urlsafe(32)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            expires = now + duration
            if expires <= now:
                raise ValueError("lease_seconds is too small for the system clock resolution")
            row = connection.execute(
                "SELECT * FROM submissions WHERE status = 'queued' ORDER BY sequence LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            submission = json.loads(row["submission_json"])
            archive_path = Path(submission["source"]["archive_path"])
            if not archive_path.is_file() or sha256_file(archive_path) != submission["source"]["archive_sha256"]:
                raise QueueError(f"frozen source archive for {row['submission_id']} is missing or corrupt")
            changed = connection.execute(
                """UPDATE submissions
                   SET status = 'running', attempt = attempt + 1, claim_token = ?,
                       claim_worker = ?, lease_expires_at = ?, updated_at = ?
                   WHERE submission_id = ? AND status = 'queued'""",
                (token, worker_id, expires, now, row["submission_id"]),
            ).rowcount
            if changed != 1:
                raise QueueError("FIFO claim lost despite write transaction")
            return {
                "submission_id": row["submission_id"],
                "submission": submission,
                "claim_token": token,
                "lease_expires_at": expires,
            }

    def heartbeat(
        self,
        submission_id: str,
        claim_token: str,
        lease_seconds: float = 900,
    ) -> float:
        duration = _lease(lease_seconds)
        if not isinstance(claim_token, str) or not claim_token:
            raise StaleClaimError("claim token is required")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            expires = now + duration
            if expires <= now:
                raise ValueError("lease_seconds is too small for the system clock resolution")
            changed = connection.execute(
                """UPDATE submissions SET lease_expires_at = ?, updated_at = ?
                   WHERE submission_id = ? AND claim_token = ? AND status = 'running'
                     AND lease_expires_at > ?""",
                (expires, now, submission_id, claim_token, now),
            ).rowcount
            if changed != 1:
                raise StaleClaimError(f"claim for {submission_id!r} is stale or cancelled")
        return expires

    def release(
        self,
        submission_id: str,
        claim_token: str,
        reason: str = "claim released before evaluation",
    ) -> dict[str, Any]:
        """Return a fenced claim to FIFO, unless cancellation already won."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("release reason must be nonempty")
        now = time.time()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, claim_token, lease_expires_at FROM submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(submission_id)
            if not isinstance(claim_token, str) or not claim_token or row["claim_token"] != claim_token:
                raise StaleClaimError(f"claim for {submission_id!r} is stale")
            if row["status"] not in {"running", "cancel_requested"}:
                raise StaleClaimError(f"submission {submission_id!r} is no longer active")
            if row["lease_expires_at"] is None or row["lease_expires_at"] <= now:
                raise StaleClaimError(f"claim for {submission_id!r} has expired")
            next_status = "cancelled" if row["status"] == "cancel_requested" else "queued"
            connection.execute(
                """UPDATE submissions SET status = ?, claim_token = NULL,
                   claim_worker = NULL, lease_expires_at = NULL,
                   cancel_reason = CASE WHEN ? = 'cancelled' THEN COALESCE(cancel_reason, ?) ELSE cancel_reason END,
                   updated_at = ? WHERE submission_id = ? AND claim_token = ?""",
                (next_status, next_status, reason, now, submission_id, claim_token),
            )
        return self.get(submission_id)

    def finish(
        self,
        submission_id: str,
        claim_token: str,
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_receipt = validate_receipt(receipt)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            row = connection.execute(
                "SELECT * FROM submissions WHERE submission_id = ?", (submission_id,)
            ).fetchone()
            if row is None:
                raise KeyError(submission_id)
            if not isinstance(claim_token, str) or not claim_token or row["claim_token"] != claim_token:
                raise StaleClaimError(f"claim for {submission_id!r} is stale")
            if row["status"] not in {"running", "cancel_requested"}:
                raise StaleClaimError(f"submission {submission_id!r} is no longer active")
            if row["lease_expires_at"] is None or row["lease_expires_at"] <= now:
                raise StaleClaimError(f"claim for {submission_id!r} has expired")
            submission = json.loads(row["submission_json"])
            for key in ("submission_id", "policy_id", "contract_id"):
                if normalized_receipt[key] != submission[key]:
                    raise ValueError(f"receipt {key} does not match submission")
            if normalized_receipt["source_snapshot_sha256"] != submission["source"]["snapshot_sha256"]:
                raise ValueError("receipt source snapshot does not match submission")
            for receipt_key, submission_key in (("candidate_commit", "candidate_commit"), ("base_commit", "base_commit")):
                if receipt_key in normalized_receipt and normalized_receipt[receipt_key] != submission.get(submission_key):
                    raise ValueError(f"receipt {receipt_key} does not match submission")
            if row["status"] == "cancel_requested":
                connection.execute(
                    """UPDATE submissions SET status = 'cancelled', claim_token = NULL,
                       claim_worker = NULL, lease_expires_at = NULL, receipt_json = NULL,
                       updated_at = ? WHERE submission_id = ? AND claim_token = ?""",
                    (now, submission_id, claim_token),
                )
            else:
                receipt_json = json.dumps(
                    normalized_receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                changed = connection.execute(
                    """UPDATE submissions SET status = 'completed', claim_token = NULL,
                       claim_worker = NULL, lease_expires_at = NULL, receipt_json = ?, updated_at = ?
                       WHERE submission_id = ? AND claim_token = ? AND status = 'running'""",
                    (receipt_json, now, submission_id, claim_token),
                ).rowcount
                if changed != 1:
                    raise StaleClaimError(f"claim for {submission_id!r} is stale")
        return self.get(submission_id)

    def cancel(self, submission_id: str, reason: str = "cancelled by user",
               claim_token: str | None = None) -> dict[str, Any]:
        """Cancel a submission.

        When ``claim_token`` is given, a running item is only cancelled if the
        caller still holds that claim. A worker whose lease expired must not be
        able to cancel the claim a later worker now owns.
        """
        if not isinstance(reason, str) or not reason.strip() or any(ord(c) < 0x20 and c not in "\t\n\r" for c in reason):
            raise ValueError("cancellation reason must be nonempty plain text")
        if claim_token is not None and (not isinstance(claim_token, str) or not claim_token):
            raise ValueError("claim_token must be a nonempty string when given")
        now = time.time()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM submissions WHERE submission_id = ?", (submission_id,)
            ).fetchone()
            if row is None:
                raise KeyError(submission_id)
            if row["status"] == "queued":
                connection.execute(
                    """UPDATE submissions SET status = 'cancelled', cancel_reason = ?, updated_at = ?
                       WHERE submission_id = ? AND status = 'queued'""",
                    (reason, now, submission_id),
                )
            elif row["status"] == "running":
                if claim_token is not None and row["claim_token"] != claim_token:
                    # The caller no longer owns this claim. Leave it alone.
                    raise StaleClaimError(f"claim for {submission_id!r} is stale")
                connection.execute(
                    """UPDATE submissions SET status = 'cancel_requested', cancel_reason = ?, updated_at = ?
                       WHERE submission_id = ? AND status = 'running'""",
                    (reason, now, submission_id),
                )
            # Repeated cancellation and already-terminal records are idempotent.
        return self.get(submission_id)

    def recover(self, now: float | int | datetime | None = None) -> list[str]:
        instant = _now(now)
        recovered: list[str] = []
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expired = connection.execute(
                """SELECT submission_id, status FROM submissions
                   WHERE status IN ('running','cancel_requested') AND lease_expires_at <= ?
                   ORDER BY sequence""",
                (instant,),
            ).fetchall()
            for row in expired:
                if row["status"] == "running":
                    connection.execute(
                        """UPDATE submissions SET status = 'queued', claim_token = NULL,
                           claim_worker = NULL, lease_expires_at = NULL, updated_at = ?
                           WHERE submission_id = ? AND status = 'running'""",
                        (instant, row["submission_id"]),
                    )
                else:
                    connection.execute(
                        """UPDATE submissions SET status = 'cancelled', claim_token = NULL,
                           claim_worker = NULL, lease_expires_at = NULL, updated_at = ?
                           WHERE submission_id = ? AND status = 'cancel_requested'""",
                        (instant, row["submission_id"]),
                    )
                recovered.append(row["submission_id"])
        return recovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--max-pending", type=int, default=DEFAULT_MAX_PENDING)
    commands = parser.add_subparsers(dest="command", required=True)
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("submission", type=Path)
    get = commands.add_parser("get")
    get.add_argument("submission_id")
    claim = commands.add_parser("claim")
    claim.add_argument("worker_id")
    claim.add_argument("--lease-seconds", type=float, default=900)
    heartbeat = commands.add_parser("heartbeat")
    heartbeat.add_argument("submission_id")
    heartbeat.add_argument("claim_token")
    heartbeat.add_argument("--lease-seconds", type=float, default=900)
    finish = commands.add_parser("finish")
    finish.add_argument("submission_id")
    finish.add_argument("claim_token")
    finish.add_argument("receipt", type=Path)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("submission_id")
    cancel.add_argument("--reason", default="cancelled by user")
    recover = commands.add_parser("recover")
    recover.add_argument("--now", type=float)
    args = parser.parse_args(argv)
    try:
        queue = Queue(args.db, max_pending=args.max_pending)
        if args.command == "enqueue":
            result: Any = {"submission_id": queue.enqueue(read_json(args.submission))}
        elif args.command == "get":
            result = queue.get(args.submission_id)
        elif args.command == "claim":
            result = queue.claim(args.worker_id, args.lease_seconds)
        elif args.command == "heartbeat":
            result = {"lease_expires_at": queue.heartbeat(args.submission_id, args.claim_token, args.lease_seconds)}
        elif args.command == "finish":
            result = queue.finish(args.submission_id, args.claim_token, read_json(args.receipt))
        elif args.command == "cancel":
            result = queue.cancel(args.submission_id, args.reason)
        else:
            result = {"recovered": queue.recover(args.now)}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, QueueError, KeyError) as exc:
        print(f"queue.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
