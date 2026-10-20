"""Thin, one-job coordinator over the local SQLite queue and trusted evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

from . import contracts
from .epochs import EpochError, EpochStore
from .queue import Queue, QueueError, StaleClaimError


LEASE_SECONDS = 900
HEARTBEAT_INTERVAL_SECONDS = 300
MAX_DRAIN_ITEMS = 128


class CoordinatorError(RuntimeError):
    """A coordinator-level setup or execution error."""


Executor = Callable[..., dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_evaluator() -> Executor:
    try:
        from .evaluator import evaluate_submission
    except ImportError as exc:
        raise CoordinatorError(
            "trusted local evaluator is unavailable; install/integrate tools.dreamrsi.evaluator"
        ) from exc
    return evaluate_submission


def _receipt(
    submission: dict[str, Any],
    *,
    verdict: str,
    stage: str,
    failure_class: str | None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": contracts.SCHEMA_VERSION,
        "submission_id": submission["submission_id"],
        "policy_id": submission["policy_id"],
        "contract_id": submission["contract_id"],
        "source_snapshot_sha256": submission["source"]["snapshot_sha256"],
        "verdict": verdict,
        "stage": stage,
        "failure_class": failure_class,
        "evidence": [],
        "created_at": _utc_now(),
    }
    if submission.get("candidate_commit") is not None:
        value["candidate_commit"] = submission["candidate_commit"]
    value["base_commit"] = submission["base_commit"]
    return contracts.validate_receipt(value)


def _verify_receipt(receipt: Any, submission: dict[str, Any]) -> dict[str, Any]:
    normalized = contracts.validate_receipt(receipt)
    for field in ("submission_id", "policy_id", "contract_id"):
        if normalized[field] != submission[field]:
            raise ValueError(f"evaluator receipt {field} does not match claimed submission")
    if normalized["source_snapshot_sha256"] != submission["source"]["snapshot_sha256"]:
        raise ValueError("evaluator receipt source snapshot does not match claimed submission")
    if normalized.get("candidate_commit") != submission.get("candidate_commit"):
        raise ValueError("evaluator receipt candidate_commit does not match claimed submission")
    if normalized.get("base_commit") != submission.get("base_commit"):
        raise ValueError("evaluator receipt base_commit does not match claimed submission")
    return normalized


def _defer_claim(queue: Queue, claim: dict[str, Any], reason: str, *, cancel: bool = False) -> dict[str, Any]:
    submission_id = claim["submission_id"]
    token = claim["claim_token"]
    if cancel:
        queue.cancel(submission_id, reason)
    record = queue.release(submission_id, token, reason)
    return {
        "submission_id": submission_id,
        "status": record["status"],
        "attempt": record["attempt"],
        "deferred": reason,
    }


def _append_result_event(
    store: EpochStore,
    submission: dict[str, Any],
    evaluator_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    receipt = record["receipt"]
    payload = {
        "submission_id": submission["submission_id"],
        "queue_status": record["status"],
        "attempt": record["attempt"],
        "verdict": None if receipt is None else receipt["verdict"],
        "stage": None if receipt is None else receipt["stage"],
        "failure_class": None if receipt is None else receipt["failure_class"],
    }
    event_type = "coordinator.evaluation_cancelled" if record["status"] == "cancelled" else "coordinator.evaluation_finished"
    key = f"coordinator.finish:{submission['submission_id']}"
    return store.append_event(
        submission["epoch_id"],
        event_type,
        key,
        payload,
        worker_id=evaluator_id,
    )


def run_once(
    queue_db: Path,
    policy_path: Path,
    run_root: Path,
    *,
    evaluator_id: str,
    epoch_db: Path | None = None,
    executor: Executor | None = None,
) -> dict[str, Any] | None:
    """Claim/evaluate/finish one FIFO item; imports and calls perform no work by themselves."""
    if not isinstance(evaluator_id, str) or not evaluator_id or evaluator_id != evaluator_id.strip():
        raise ValueError("evaluator_id must be a nonempty trimmed identifier")
    evaluator = executor if executor is not None else _local_evaluator()
    if not callable(evaluator):
        raise ValueError("executor must be callable")

    policy = contracts.validate_policy(contracts.read_json(policy_path))
    queue = Queue(queue_db)
    epochs = EpochStore(epoch_db) if epoch_db is not None else None
    if epochs is not None and queue.db_path.resolve() == epochs.path.resolve():
        raise CoordinatorError("queue and epoch state must use separate SQLite databases")

    claim = queue.claim(evaluator_id, lease_seconds=LEASE_SECONDS)
    if claim is None:
        return None
    submission = contracts.validate_submission(claim["submission"])
    if epochs is not None:
        try:
            epoch = epochs.get_epoch(submission["epoch_id"])
        except EpochError:
            return _defer_claim(queue, claim, "submission references an unknown epoch")
        if epoch["status"] != "active":
            return _defer_claim(
                queue,
                claim,
                f"epoch is {epoch['status']}; evaluation was not started",
                cancel=epoch["status"] in {"cancelled", "completed"},
            )

    if submission["policy_id"] != policy["policy_id"] or submission["contract_id"] != policy["contract_id"]:
        rejected = _receipt(
            submission, verdict="rejected", stage="admission", failure_class="candidate"
        )
        record = queue.finish(claim["submission_id"], claim["claim_token"], rejected)
        event = None
        if epochs is not None:
            event = _append_result_event(epochs, submission, evaluator_id, record)
        return {**record, "event": event}

    next_heartbeat = time.monotonic() + HEARTBEAT_INTERVAL_SECONDS

    def cancelled() -> bool:
        nonlocal next_heartbeat
        try:
            state = queue.get(claim["submission_id"])
        except (KeyError, QueueError):
            return True
        if state["status"] in {"cancel_requested", "cancelled", "completed"}:
            return True
        if state.get("claim_token") != claim["claim_token"]:
            # The claim was reclaimed by another worker. This one has no
            # authority over the item and must not cancel or finish it.
            return True
        if epochs is not None:
            try:
                epoch_state = epochs.get_epoch(submission["epoch_id"])
            except EpochError:
                return True
            if epoch_state["status"] == "cancelled":
                try:
                    queue.cancel(claim["submission_id"], "epoch cancelled during evaluation",
                                 claim_token=claim["claim_token"])
                except (KeyError, QueueError, StaleClaimError):
                    return True
                return True
        now = time.monotonic()
        if now >= next_heartbeat:
            try:
                queue.heartbeat(
                    claim["submission_id"], claim["claim_token"], lease_seconds=LEASE_SECONDS
                )
            except (KeyError, QueueError):
                return True
            next_heartbeat = now + HEARTBEAT_INTERVAL_SECONDS
        return False

    try:
        if cancelled():
            try:
                queue.cancel(claim["submission_id"], "cancelled before evaluation",
                             claim_token=claim["claim_token"])
            except (KeyError, QueueError, StaleClaimError):
                pass
            receipt = _receipt(
                submission, verdict="blocked", stage="admission", failure_class=None
            )
        else:
            result = evaluator(submission, policy, Path(run_root), cancelled=cancelled)
            receipt = _verify_receipt(result, submission)
    except Exception:
        # Evaluator exceptions are infrastructure failures, not candidate evidence.
        receipt = _receipt(
            submission, verdict="blocked", stage="admission", failure_class="infrastructure"
        )
    try:
        record = queue.finish(claim["submission_id"], claim["claim_token"], receipt)
    except StaleClaimError:
        # Another worker owns this item now. Its result, not this one, decides.
        return {
            "submission_id": claim["submission_id"],
            "status": "stale_claim",
            "attempt": claim.get("attempt"),
            "deferred": "the claim was reclaimed by another worker",
        }
    event = None
    event_error = None
    if epochs is not None:
        try:
            event = _append_result_event(epochs, submission, evaluator_id, record)
        except (EpochError, OSError, ValueError) as exc:
            # Queue and epoch records live in separate owner databases. Surface
            # an event persistence failure; never roll back a fenced queue finish.
            event_error = f"{type(exc).__name__}: {exc}"
    result_record = {**record, "event": event}
    if event_error is not None:
        result_record["event_error"] = event_error
    return result_record


def drain(
    queue_db: Path,
    policy_path: Path,
    run_root: Path,
    *,
    evaluator_id: str,
    max_items: int = 1,
    epoch_db: Path | None = None,
    executor: Executor | None = None,
) -> list[dict[str, Any]]:
    if isinstance(max_items, bool) or not isinstance(max_items, int) or not 1 <= max_items <= MAX_DRAIN_ITEMS:
        raise ValueError(f"max_items must be between 1 and {MAX_DRAIN_ITEMS}")
    results: list[dict[str, Any]] = []
    for _ in range(max_items):
        item = run_once(
            queue_db,
            policy_path,
            run_root,
            evaluator_id=evaluator_id,
            epoch_db=epoch_db,
            executor=executor,
        )
        if item is None:
            break
        results.append(item)
        if item.get("status") == "queued":
            break
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-db", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument("--epoch-db", type=Path)
    parser.add_argument("--max-items", type=int, default=1)
    parser.add_argument("--transport", choices=("local", "ssh"), default="local")
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--trusted-harness", type=Path)
    parser.add_argument("--ssh-alias", default="fnn")
    args = parser.parse_args(argv)
    try:
        executor = None
        if args.transport == "ssh":
            if args.remote_root is None or args.trusted_harness is None:
                raise CoordinatorError("SSH transport requires --remote-root and --trusted-harness")
            from .transport import SSHExecutor

            executor = SSHExecutor(
                remote_root=args.remote_root,
                trusted_harness=args.trusted_harness,
                alias=args.ssh_alias,
            )
        print(json.dumps(drain(
            args.queue_db,
            args.policy,
            args.run_root,
            evaluator_id=args.evaluator_id,
            max_items=args.max_items,
            epoch_db=args.epoch_db,
            executor=executor,
        ), ensure_ascii=False, sort_keys=True))
        return 0
    except (CoordinatorError, QueueError, EpochError, OSError, ValueError) as exc:
        print(f"coordinator.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
