"""Bounded worker assignments and an explicit OpenCode v2 adapter.

Run with ``python -m tools.dreamrsi.workers``. A worker is bound to one exact
existing Git worktree, base commit, model, role and path allowlist. Read/edit
permissions are enforced in the OpenCode v2 session ruleset; shell, network,
subagents and all other actions are denied. Assignments are separately scoped
to a subset of those paths.

The V2 API documents ``session.create`` with a ``location`` and says that a
``parentID`` instead creates the child at its parent's location. This adapter
therefore never sets ``parentID`` and always supplies the preassigned worktree
location. API documentation/CLI capability detection is read-only. The v2 API
does not document a host wake callback. Its ``event.subscribe`` route is a
volatile SSE stream: the OpenAPI contract says slow consumers can overflow and
events during disconnection are missed. Completion events are therefore durable
callback packets which a coordinator must manually drain/ack; no background
polling or synthetic callback success is claimed. The documented API, CLI and
permission references are ``https://opencode.ai/v2/docs/api``,
``https://opencode.ai/v2/docs/cli`` and
``https://opencode.ai/v2/docs/permissions``.

The scheduler hard-bounds registered workers, hypotheses, assignments, prompt
dispatches, and cumulative reserved token/cost/wall-time envelopes. OpenCode's
documented ``session.prompt`` request has no provider-side token/cost/time
quota. Those three envelopes are accounting reservations, not claims that the
server enforces generation usage. A launched prompt may continue its internal
agent loop until OpenCode completes it; cancellation fences later dispatches
but does not interrupt already-launched sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid
from typing import Any, Protocol

from tools.dreamrsi.epochs import (
    EpochError,
    EpochStore,
    _identifier,
    canonical_json,
    cost_to_microusd,
    microusd_to_string,
    utc_now,
)


_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_MODEL_RE = re.compile(r"^([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)(?:#([A-Za-z0-9._-]+))?$")
_MAX_TEXT = 8000


class WorkerError(EpochError):
    """Invalid, unbounded or unsafe worker operation."""


def _run_git(path: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True,
            text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkerError(f"cannot validate Git worktree: {exc}") from None
    return completed.stdout.strip()


def _verify_worktree(path_value: str | Path, base_commit: str, *, require_clean: bool,
                     require_head: bool = True) -> tuple[Path, str]:
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        raise WorkerError("worktree must be a non-empty absolute path")
    requested = Path(path_value).expanduser()
    if not requested.is_absolute():
        raise WorkerError("worktree must be an absolute path")
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise WorkerError(f"worktree does not exist: {exc}") from None
    if not root.is_dir() or str(root) != str(requested):
        raise WorkerError("worktree must be the canonical absolute path to an existing directory")
    if not _COMMIT_RE.fullmatch(base_commit or ""):
        raise WorkerError("base_commit must be a full 40- or 64-character commit hash")
    top = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != root:
        raise WorkerError("worktree must name the Git worktree root, not a subdirectory")
    entries = _run_git(root, "worktree", "list", "--porcelain").splitlines()
    registered_paths = [Path(line[len("worktree "):]).resolve(strict=True)
                        for line in entries if line.startswith("worktree ")]
    if root not in registered_paths:
        raise WorkerError("path is not present in this repository's Git worktree list")
    resolved_base = _run_git(root, "rev-parse", "--verify", f"{base_commit}^{{commit}}")
    if resolved_base.lower() != base_commit.lower():
        raise WorkerError("base_commit must be the full canonical commit hash")
    head = _run_git(root, "rev-parse", "HEAD")
    if require_head and head.lower() != resolved_base.lower():
        raise WorkerError("worktree HEAD must equal the registered base commit")
    if require_clean and _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise WorkerError("worktree must be clean when registering/starting a worker")
    return root, resolved_base.lower()


def _normalize_scope(root: Path, raw: str, label: str) -> str:
    if not isinstance(raw, str) or not raw or len(raw) > 512 or raw.startswith("/") or "\\" in raw:
        raise WorkerError(f"{label} must be a non-empty worktree-relative path")
    directory_scope = raw.endswith("/**")
    relative_text = raw[:-3] if directory_scope else raw
    if not relative_text or any(part in ("", ".", "..") for part in relative_text.split("/")):
        raise WorkerError(f"{label} may not contain empty, dot, or parent path components")
    pure = PurePosixPath(relative_text)
    if not pure.parts:
        raise WorkerError(f"{label} may not contain empty, dot, or parent path components")
    if any("*" in part or "?" in part or "[" in part or "]" in part for part in pure.parts):
        raise WorkerError(f"{label} supports only an optional trailing '/**' directory scope")
    if any(part.casefold() == ".git" for part in pure.parts):
        raise WorkerError(f"{label} may not include Git metadata")
    relative = pure.as_posix()
    target = root.joinpath(*pure.parts)
    try:
        resolved = target.resolve(strict=target.exists())
    except OSError as exc:
        raise WorkerError(f"{label} cannot be resolved: {exc}") from None
    if not resolved.is_relative_to(root):
        raise WorkerError(f"{label} resolves outside the registered worktree")
    # Do not authorize a path that already traverses a symlink. This prevents a
    # narrow lexical scope from becoming a broad permission through an alias.
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise WorkerError(f"{label} may not traverse symlinks")
    if directory_scope:
        if not target.exists() or not target.is_dir():
            raise WorkerError(f"{label} directory must already exist")
        return relative + "/**"
    if target.exists() and not target.is_file():
        raise WorkerError(f"{label} must name a file; use '/**' for a directory tree")
    if not target.exists() and not target.parent.is_dir():
        raise WorkerError(f"parent directory for {label} must already exist")
    return relative


def _scope_covers(parent: str, child: str) -> bool:
    if parent == child:
        return True
    if parent.endswith("/**"):
        prefix = parent[:-3].rstrip("/") + "/"
        return child.startswith(prefix)
    return False


def _normalize_scopes(root: Path, values: list[str] | tuple[str, ...], label: str,
                      *, allow_empty: bool = False) -> list[str]:
    if allow_empty and values == []:
        return []
    if not isinstance(values, (list, tuple)) or not values:
        raise WorkerError(f"{label} must be a non-empty list of explicit paths")
    if len(values) > 64:
        raise WorkerError(f"{label} may contain at most 64 explicit paths")
    result = [_normalize_scope(root, value, label) for value in values]
    if len(set(result)) != len(result):
        raise WorkerError(f"{label} contains duplicate paths")
    return sorted(result)


def _is_subset(requested: list[str], allowed: list[str]) -> bool:
    return all(any(_scope_covers(parent, child) for parent in allowed) for child in requested)


def _worker_from_row(row: Any) -> dict[str, Any]:
    return {
        "epoch_id": row["epoch_id"], "worker_id": row["worker_id"], "worktree": row["worktree"],
        "base_commit": row["base_commit"], "model": row["model"], "role": row["role"],
        "allowed_paths": json.loads(row["allowed_paths_json"]),
        "permissions": json.loads(row["permissions_json"]),
        "stop_conditions": json.loads(row["stop_conditions_json"]),
        "callback": json.loads(row["callback_json"]), "created_at": row["created_at"],
    }


def _assignment_from_row(row: Any) -> dict[str, Any]:
    return {
        "assignment_id": row["assignment_id"], "epoch_id": row["epoch_id"],
        "worker_id": row["worker_id"], "hypothesis_id": row["hypothesis_id"],
        "idempotency_key": row["idempotency_key"], "instructions": row["instructions"],
        "read_paths": json.loads(row["read_paths_json"]), "write_paths": json.loads(row["write_paths_json"]),
        "budget": {"tokens": row["token_budget"], "cost_usd": microusd_to_string(row["cost_budget_microusd"]),
                   "wall_seconds": row["wall_budget_seconds"], "prompts": row["prompt_budget"]},
        "prompts_used": row["prompts_used"], "status": row["status"], "session_id": row["session_id"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def register_worker(
    store: EpochStore,
    *,
    epoch_id: str,
    worker_id: str,
    worktree: str | Path,
    base_commit: str,
    model: str,
    role: str,
    allowed_files: list[str],
    stop_conditions: list[str],
    callback: dict[str, Any],
    permissions: dict[str, Any],
) -> dict[str, Any]:
    """Persist one immutable, explicitly-scoped worker registration."""
    epoch_id = _identifier(epoch_id, "epoch_id")
    worker_id = _identifier(worker_id, "worker_id")
    root, resolved_base = _verify_worktree(worktree, base_commit, require_clean=True)
    allowed = _normalize_scopes(root, allowed_files, "allowed_files")
    if not isinstance(model, str) or len(model) > 256 or not _MODEL_RE.fullmatch(model):
        raise WorkerError("model must be an explicit provider/model or provider/model#variant reference")
    if not isinstance(role, str) or not role.strip() or len(role) > 120:
        raise WorkerError("role must contain 1-120 characters")
    if not isinstance(stop_conditions, list) or not 1 <= len(stop_conditions) <= 32:
        raise WorkerError("stop_conditions must contain 1-32 explicit conditions")
    if any(not isinstance(value, str) or not value.strip() or len(value) > 1000 for value in stop_conditions):
        raise WorkerError("each stop condition must contain 1-1000 characters")
    if not isinstance(callback, dict) or set(callback) != {"mode", "recipient"}:
        raise WorkerError("callback must explicitly be {'mode':'manual','recipient':...}; no host wake hook is available")
    if callback["mode"] != "manual" or not isinstance(callback["recipient"], str) \
            or not callback["recipient"].strip() or len(callback["recipient"]) > 120:
        raise WorkerError("callback must use manual mode with a 1-120 character recipient")
    if not isinstance(permissions, dict) or set(permissions) != {"read_paths", "write_paths", "shell", "network"}:
        raise WorkerError("permissions must explicitly specify read_paths, write_paths, shell and network")
    if permissions["shell"] is not False or permissions["network"] is not False:
        raise WorkerError("shell and network permissions must be explicitly denied")
    read_paths = _normalize_scopes(root, permissions["read_paths"], "permissions.read_paths")
    write_paths = _normalize_scopes(root, permissions["write_paths"], "permissions.write_paths", allow_empty=True)
    if not _is_subset(read_paths, allowed) or not _is_subset(write_paths, allowed):
        raise WorkerError("read/write permissions must be subsets of allowed_files")
    if not _is_subset(write_paths, read_paths):
        raise WorkerError("write_paths must also be readable")
    normalized_permissions = {"read_paths": read_paths, "write_paths": write_paths,
                              "shell": False, "network": False}
    normalized_callback = {"mode": "manual", "recipient": callback["recipient"].strip()}
    normalized_stop = [item.strip() for item in stop_conditions]
    now = utc_now()
    with store.transaction() as con:
        epoch = store.require_dispatchable(con, epoch_id)
        budgets = json.loads(epoch["budgets_json"])
        existing = con.execute("SELECT * FROM workers WHERE epoch_id=? AND worker_id=?",
                               (epoch_id, worker_id)).fetchone()
        payload = (str(root), resolved_base, model, role.strip(), canonical_json(allowed),
                   canonical_json(normalized_permissions), canonical_json(normalized_stop), canonical_json(normalized_callback))
        if existing:
            old = (existing["worktree"], existing["base_commit"], existing["model"], existing["role"],
                   existing["allowed_paths_json"], existing["permissions_json"], existing["stop_conditions_json"],
                   existing["callback_json"])
            if old != payload:
                raise WorkerError("worker ID is already registered with different immutable fields")
            return _worker_from_row(existing)
        count = con.execute("SELECT COUNT(*) FROM workers WHERE epoch_id=?", (epoch_id,)).fetchone()[0]
        if count >= budgets["max_workers"]:
            raise WorkerError("epoch worker bound is exhausted")
        try:
            con.execute(
                """INSERT INTO workers(epoch_id,worker_id,worktree,base_commit,model,role,allowed_paths_json,
                   permissions_json,stop_conditions_json,callback_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (epoch_id, worker_id, str(root), resolved_base, model, role.strip(), *payload[4:], now),
            )
        except sqlite3.IntegrityError as exc:
            raise WorkerError(f"worker registration conflicts with an existing worktree or record: {exc}") from None
        return _worker_from_row(con.execute("SELECT * FROM workers WHERE epoch_id=? AND worker_id=?",
                                            (epoch_id, worker_id)).fetchone())


def get_worker(store: EpochStore, epoch_id: str, worker_id: str) -> dict[str, Any]:
    with store.connection() as con:
        row = con.execute("SELECT * FROM workers WHERE epoch_id=? AND worker_id=?", (epoch_id, worker_id)).fetchone()
        if row is None:
            raise WorkerError(f"unknown worker {worker_id} in epoch {epoch_id}")
        return _worker_from_row(row)


def add_hypothesis(store: EpochStore, epoch_id: str, hypothesis_id: str, description: str) -> dict[str, Any]:
    """Explicit worker-lane alias for the epoch store's bounded hypothesis API."""
    return store.add_hypothesis(epoch_id, hypothesis_id, description)


def create_assignment(
    store: EpochStore,
    *,
    epoch_id: str,
    assignment_id: str,
    worker_id: str,
    hypothesis_id: str,
    idempotency_key: str,
    instructions: str,
    read_paths: list[str],
    write_paths: list[str],
    token_budget: int,
    cost_budget_usd: str | int | float,
    wall_budget_seconds: int,
    prompt_budget: int,
) -> dict[str, Any]:
    """Reserve a bounded, path-specific assignment; no worker is launched."""
    epoch_id = _identifier(epoch_id, "epoch_id")
    assignment_id = _identifier(assignment_id, "assignment_id")
    worker_id = _identifier(worker_id, "worker_id")
    hypothesis_id = _identifier(hypothesis_id, "hypothesis_id")
    if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200:
        raise WorkerError("idempotency_key must contain 1-200 characters")
    if not isinstance(instructions, str) or not instructions.strip() or len(instructions) > _MAX_TEXT:
        raise WorkerError(f"instructions must contain 1-{_MAX_TEXT} characters")
    for label, value, max_value in (("token_budget", token_budget, 10**12),
                                    ("wall_budget_seconds", wall_budget_seconds, 365 * 24 * 60 * 60),
                                    ("prompt_budget", prompt_budget, 100_000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= max_value:
            raise WorkerError(f"{label} must be an integer from 1 through {max_value}")
    cost = cost_to_microusd(cost_budget_usd, "cost_budget_usd")
    if cost <= 0:
        raise WorkerError("cost_budget_usd must be greater than zero")
    with store.transaction() as con:
        epoch = store.require_dispatchable(con, epoch_id)
        worker_row = con.execute("SELECT * FROM workers WHERE epoch_id=? AND worker_id=?",
                                 (epoch_id, worker_id)).fetchone()
        if worker_row is None:
            raise WorkerError("worker is not registered in this epoch")
        hypothesis = con.execute("SELECT * FROM hypotheses WHERE epoch_id=? AND hypothesis_id=?",
                                 (epoch_id, hypothesis_id)).fetchone()
        if hypothesis is None:
            raise WorkerError("hypothesis must be explicitly registered before assignment")
        worker = _worker_from_row(worker_row)
        root = Path(worker["worktree"])
        read_scopes = _normalize_scopes(root, read_paths, "assignment.read_paths")
        write_scopes = _normalize_scopes(root, write_paths, "assignment.write_paths", allow_empty=True)
        allowed_read = worker["permissions"]["read_paths"]
        allowed_write = worker["permissions"]["write_paths"]
        if not _is_subset(read_scopes, allowed_read) or not _is_subset(write_scopes, allowed_write):
            raise WorkerError("assignment paths must be subsets of the worker's explicit read/write permissions")
        if not _is_subset(write_scopes, read_scopes):
            raise WorkerError("assignment write_paths must also be readable")
        budgets = json.loads(epoch["budgets_json"])
        old = con.execute("SELECT * FROM assignments WHERE epoch_id=? AND idempotency_key=?",
                          (epoch_id, idempotency_key)).fetchone()
        values = (assignment_id, worker_id, hypothesis_id, instructions.strip(), canonical_json(read_scopes),
                  canonical_json(write_scopes), token_budget, cost, wall_budget_seconds, prompt_budget)
        if old:
            previous = (old["assignment_id"], old["worker_id"], old["hypothesis_id"], old["instructions"],
                        old["read_paths_json"], old["write_paths_json"], old["token_budget"],
                        old["cost_budget_microusd"], old["wall_budget_seconds"], old["prompt_budget"])
            if previous != values:
                raise WorkerError("assignment idempotency key was reused with different content")
            return _assignment_from_row(old)
        count = con.execute("SELECT COUNT(*) FROM assignments WHERE epoch_id=?", (epoch_id,)).fetchone()[0]
        if count >= budgets["max_assignments"]:
            raise WorkerError("epoch assignment bound is exhausted")
        prompts_reserved = epoch["reserved_prompts"] + prompt_budget
        tokens_reserved = epoch["reserved_tokens"] + token_budget
        cost_reserved = epoch["reserved_cost_microusd"] + cost
        seconds_reserved = epoch["reserved_wall_seconds"] + wall_budget_seconds
        limits = (("prompt", prompts_reserved, budgets["max_prompts"]),
                  ("token", tokens_reserved, budgets["max_tokens"]),
                  ("cost", cost_reserved, budgets["max_cost_microusd"]),
                  ("wall-time", seconds_reserved, budgets["max_wall_seconds"]))
        for label, reserved, ceiling in limits:
            if reserved > ceiling:
                raise WorkerError(f"epoch {label} reservation budget would be exceeded")
        active = con.execute(
            "SELECT COUNT(*) FROM assignments WHERE epoch_id=? AND status IN ('planned','launching','launched')",
            (epoch_id,),
        ).fetchone()[0]
        if active >= budgets["max_workers"]:
            raise WorkerError("epoch concurrent worker bound is exhausted")
        now = utc_now()
        try:
            con.execute(
                """INSERT INTO assignments(assignment_id,epoch_id,worker_id,hypothesis_id,idempotency_key,
                   instructions,read_paths_json,write_paths_json,token_budget,cost_budget_microusd,
                   wall_budget_seconds,prompt_budget,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (assignment_id, epoch_id, worker_id, hypothesis_id, idempotency_key, instructions.strip(),
                 canonical_json(read_scopes), canonical_json(write_scopes), token_budget, cost,
                 wall_budget_seconds, prompt_budget, "planned", now, now),
            )
        except sqlite3.IntegrityError as exc:
            raise WorkerError(f"assignment conflicts with an existing active assignment: {exc}") from None
        con.execute(
            "UPDATE epochs SET reserved_prompts=?,reserved_tokens=?,reserved_cost_microusd=?,reserved_wall_seconds=?,updated_at=? WHERE epoch_id=?",
            (prompts_reserved, tokens_reserved, cost_reserved, seconds_reserved, now, epoch_id),
        )
        store._append_event_tx(con, epoch_id, "worker.assignment_planned", f"assignment:{assignment_id}:planned",
                               worker_id, None, {"assignment_id": assignment_id, "hypothesis_id": hypothesis_id,
                                                "read_paths": read_scopes, "write_paths": write_scopes,
                                                "prompt_budget": prompt_budget})
        return _assignment_from_row(con.execute("SELECT * FROM assignments WHERE assignment_id=?",
                                                (assignment_id,)).fetchone())


def list_workers(store: EpochStore, epoch_id: str) -> dict[str, list[dict[str, Any]]]:
    with store.connection() as con:
        store.require_epoch(con, epoch_id)
        workers = [_worker_from_row(row) for row in con.execute(
            "SELECT * FROM workers WHERE epoch_id=? ORDER BY worker_id", (epoch_id,))]
        assignments = [_assignment_from_row(row) for row in con.execute(
            "SELECT * FROM assignments WHERE epoch_id=? ORDER BY created_at,assignment_id", (epoch_id,))]
        return {"workers": workers, "assignments": assignments}


def recover_launches(store: EpochStore, epoch_id: str) -> list[dict[str, Any]]:
    """Fence interrupted launch attempts without retrying possibly-sent prompts."""
    epoch_id = _identifier(epoch_id, "epoch_id")
    recovered = []
    with store.transaction() as con:
        store.require_epoch(con, epoch_id)
        rows = con.execute(
            "SELECT * FROM assignments WHERE epoch_id=? AND status='launching' ORDER BY assignment_id",
            (epoch_id,),
        ).fetchall()
        for row in rows:
            con.execute("UPDATE assignments SET status='launch_uncertain',updated_at=? WHERE assignment_id=?",
                        (utc_now(), row["assignment_id"]))
            store._append_event_tx(
                con, epoch_id, "worker.launch_uncertain",
                f"recover:{row['assignment_id']}:{row['launch_token']}", row["worker_id"], None,
                {"assignment_id": row["assignment_id"], "session_id": row["session_id"],
                 "note": "process ended during launch; inspect session before manual retry"},
            )
            recovered.append({"assignment_id": row["assignment_id"], "status": "launch_uncertain",
                              "session_id": row["session_id"]})
    return recovered


def _get_assignment(store: EpochStore, assignment_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    with store.connection() as con:
        row = con.execute("SELECT * FROM assignments WHERE assignment_id=?", (assignment_id,)).fetchone()
        if row is None:
            raise WorkerError(f"unknown assignment: {assignment_id}")
        assignment = _assignment_from_row(row)
        worker_row = con.execute("SELECT * FROM workers WHERE epoch_id=? AND worker_id=?",
                                 (assignment["epoch_id"], assignment["worker_id"])).fetchone()
        hypothesis_row = con.execute("SELECT * FROM hypotheses WHERE epoch_id=? AND hypothesis_id=?",
                                     (assignment["epoch_id"], assignment["hypothesis_id"])).fetchone()
        epoch = store._epoch_json(con, assignment["epoch_id"])
        if worker_row is None or hypothesis_row is None:
            raise WorkerError("assignment references missing worker or hypothesis")
        return assignment, _worker_from_row(worker_row), dict(hypothesis_row), epoch


def opencode_permission_rules(read_paths: list[str], write_paths: list[str]) -> list[dict[str, str]]:
    """Fail-closed V2 rules: deny every action, then allow selected read/edit paths."""
    rules = [
        {"action": "*", "resource": "*", "effect": "deny"},
        {"action": "external_directory", "resource": "*", "effect": "deny"},
    ]
    for path in read_paths:
        rules.append({"action": "read", "resource": path, "effect": "allow"})
    for path in write_paths:
        rules.append({"action": "edit", "resource": path, "effect": "allow"})
    return rules


def _model_ref(model: str) -> dict[str, str]:
    match = _MODEL_RE.fullmatch(model)
    if not match:
        raise WorkerError("invalid registered model reference")
    provider, model_id, variant = match.groups()
    result = {"providerID": provider, "id": model_id}
    if variant:
        result["variant"] = variant
    return result


def _prompt_text(worker: dict[str, Any], hypothesis: dict[str, Any], assignment: dict[str, Any]) -> str:
    lines = [
        "Research-bench assignment. This is one bounded worker in an explicitly registered worktree.",
        f"Worker: {worker['worker_id']} (role: {worker['role']})",
        f"Epoch: {assignment['epoch_id']}; assignment: {assignment['assignment_id']}",
        f"Base commit: {worker['base_commit']}",
        f"Hypothesis {hypothesis['hypothesis_id']}: {hypothesis['description']}",
        "Permitted read paths: " + ", ".join(assignment["read_paths"]),
        "Permitted write paths: " + ", ".join(assignment["write_paths"]),
        "Do not use shell, network, subagents, or paths outside the explicit allowlist. The OpenCode session",
        "permissions independently enforce this restriction; do not request broader permissions.",
        "Stop conditions:",
    ]
    lines.extend(f"- {condition}" for condition in worker["stop_conditions"])
    lines.extend([
        "Task:", assignment["instructions"],
        "When stopping, report a concise result and any blocker. The coordinator callback is a durable",
        "manual packet; it is not a live wake hook. Do not create a new branch or worktree.",
    ])
    return "\n".join(lines)


def build_launch_plan(store: EpochStore, assignment_id: str, *, opencode: str = "opencode",
                      server: str | None = None) -> dict[str, Any]:
    """Build an inspectable API plan without invoking OpenCode or changing state."""
    assignment, worker, hypothesis, epoch = _get_assignment(store, assignment_id)
    root, _ = _verify_worktree(worker["worktree"], worker["base_commit"], require_clean=True)
    for scope in assignment["read_paths"] + assignment["write_paths"]:
        normalized = _normalize_scope(root, scope, "assignment scope")
        if normalized != scope:
            raise WorkerError("registered assignment scope no longer resolves canonically")
    if assignment["status"] != "planned":
        raise WorkerError(f"only planned assignments can be initially launched; status is {assignment['status']}")
    if epoch["status"] != "active":
        raise WorkerError(f"epoch {epoch['status']}; assignment cannot be launched")
    title = f"bench:{assignment['epoch_id']}:{assignment['worker_id']}:{assignment_id}"
    create_body = {
        "title": title,
        "agent": worker["role"],
        "model": _model_ref(worker["model"]),
        "location": {"directory": worker["worktree"]},
        "permissions": opencode_permission_rules(assignment["read_paths"], assignment["write_paths"]),
    }
    prompt = _prompt_text(worker, hypothesis, assignment)
    prompt_id = "msg_" + hashlib.sha256(f"{assignment_id}:1".encode()).hexdigest()[:32]
    prompt_body = {"id": prompt_id, "text": prompt, "resume": True}
    create_command = [opencode, "api"]
    if server:
        create_command.extend(["--server", server])
    create_command.extend(["--data", canonical_json(create_body), "session.create"])
    return {
        "adapter": "opencode-v2-cli",
        "dry_run": True,
        "server_api_verified": False,
        "live_launch_verified": False,
        "host_wake_supported": False,
        "callback_delivery": "durable manual drain/ack only",
        "session_create": {"operation": "session.create", "body": create_body,
                           "command": create_command},
        "session_prompt_template": {"operation": "session.prompt", "param": "sessionID=<created-session-id>",
                                    "body": prompt_body},
        "parentID_included": False,
        "budget_enforcement": {
            "hard": ["worker registrations", "hypotheses", "assignments", "prompt dispatch count", "path permissions"],
            "accounting_only": ["provider tokens", "provider cost", "wall time"],
            "reason": "documented OpenCode v2 session.prompt exposes no provider token/cost/time quota",
        },
        "base_commit": worker["base_commit"],
    }


class Adapter(Protocol):
    def create_session(self, body: dict[str, Any]) -> str: ...
    def send_prompt(self, session_id: str, body: dict[str, Any]) -> Any: ...


class OpenCodeAdapter:
    """V2 CLI client using documented operation IDs; never shells command text."""

    def __init__(self, executable: str = "opencode", server: str | None = None, timeout: int = 30):
        self.executable = executable
        self.server = server
        self.timeout = timeout

    def _prefix(self) -> list[str]:
        prefix = [self.executable, "api"]
        if self.server:
            prefix.extend(["--server", self.server])
        return prefix

    def _api(self, operation: str, *, body: dict[str, Any], session_id: str | None = None) -> Any:
        command = self._prefix()
        command.extend(["--data", canonical_json(body)])
        if session_id is not None:
            command.extend(["--param", f"sessionID={session_id}"])
        command.append(operation)
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise WorkerError(f"OpenCode {operation} timed out; outcome may be uncertain") from None
        except OSError as exc:
            raise WorkerError(f"cannot execute OpenCode CLI: {exc}") from None
        if result.returncode != 0:
            # Do not copy arbitrary server output into durable state; it may
            # include private project/provider details.
            raise WorkerError(f"OpenCode {operation} failed with exit code {result.returncode}")
        if not result.stdout.strip():
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            raise WorkerError(f"OpenCode {operation} returned non-JSON output") from None

    def create_session(self, body: dict[str, Any]) -> str:
        response = self._api("session.create", body=body)
        try:
            session_id = response["data"]["id"]
        except (TypeError, KeyError):
            raise WorkerError("OpenCode session.create response had no session ID") from None
        if not isinstance(session_id, str) or not session_id:
            raise WorkerError("OpenCode session.create returned an invalid session ID")
        return session_id

    def send_prompt(self, session_id: str, body: dict[str, Any]) -> Any:
        return self._api("session.prompt", body=body, session_id=session_id)


def detect_opencode_capabilities(executable: str = "opencode") -> dict[str, Any]:
    """Read only the installed CLI version/help; no API/auth/model query occurs."""
    located = shutil.which(executable)
    if located is None and Path(executable).is_file():
        located = str(Path(executable).resolve())
    if located is None:
        return {"available": False, "executable": executable, "launch_supported": False,
                "server_api_verified": False, "live_launch_verified": False,
                "host_wake_supported": False, "callback_delivery": "durable manual drain/ack only"}
    try:
        version = subprocess.run([located, "--version"], capture_output=True, text=True, timeout=5, check=False)
        api_help = subprocess.run([located, "api", "--help"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "executable": located, "launch_supported": False,
                "server_api_verified": False, "live_launch_verified": False,
                "host_wake_supported": False, "error": str(exc),
                "callback_delivery": "durable manual drain/ack only"}
    text = version.stdout.strip() or version.stderr.strip()
    match = re.search(r"\bv(\d+)\.", text)
    major = int(match.group(1)) if match else None
    api_supported = api_help.returncode == 0 and "operation" in (api_help.stdout + api_help.stderr).lower()
    return {
        "available": version.returncode == 0,
        "executable": located,
        "version": text,
        "major_version": major,
        "api_cli": api_supported,
        "documented_v2_operations": ["session.create", "session.prompt"],
        "session_location_field": True,
        "parent_id_forbidden_for_isolation": True,
        "launch_supported": version.returncode == 0 and major == 2 and api_supported,
        "server_api_verified": False,
        "live_launch_verified": False,
        "host_wake_supported": False,
        "callback_delivery": "durable manual drain/ack only",
        "side_effects": "version and --help only; no API request, credentials, model, or session touched",
    }


def _launch_token() -> str:
    return uuid.uuid4().hex


def _begin_launch(store: EpochStore, assignment_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, int]:
    with store.transaction() as con:
        row = con.execute("SELECT * FROM assignments WHERE assignment_id=?", (assignment_id,)).fetchone()
        if row is None:
            raise WorkerError(f"unknown assignment: {assignment_id}")
        epoch = store.require_dispatchable(con, row["epoch_id"])
        if row["status"] != "planned":
            raise WorkerError(f"assignment cannot launch from status {row['status']}")
        if row["prompts_used"] >= row["prompt_budget"]:
            raise WorkerError("assignment prompt budget is exhausted")
        worker_row = con.execute("SELECT * FROM workers WHERE epoch_id=? AND worker_id=?",
                                 (row["epoch_id"], row["worker_id"])).fetchone()
        hypothesis_row = con.execute("SELECT * FROM hypotheses WHERE epoch_id=? AND hypothesis_id=?",
                                     (row["epoch_id"], row["hypothesis_id"])).fetchone()
        if worker_row is None or hypothesis_row is None:
            raise WorkerError("assignment references missing worker or hypothesis")
        token = _launch_token()
        now = utc_now()
        con.execute("UPDATE assignments SET status='launching',launch_token=?,launch_fence=?,updated_at=? WHERE assignment_id=?",
                    (token, epoch["dispatch_fence"], now, assignment_id))
        store._append_event_tx(con, row["epoch_id"], "worker.launch_prepared", f"launch:{assignment_id}:prepared",
                               row["worker_id"], None, {"assignment_id": assignment_id,
                                                        "dispatch_fence": epoch["dispatch_fence"]})
        assignment = _assignment_from_row(con.execute("SELECT * FROM assignments WHERE assignment_id=?",
                                                      (assignment_id,)).fetchone())
        epoch_json = store._epoch_json(con, row["epoch_id"])
        return assignment, _worker_from_row(worker_row), dict(hypothesis_row), token, epoch["dispatch_fence"]


def _bind_session(store: EpochStore, assignment_id: str, token: str, session_id: str) -> bool:
    with store.transaction() as con:
        row = con.execute("SELECT * FROM assignments WHERE assignment_id=?", (assignment_id,)).fetchone()
        if row is None or row["launch_token"] != token or row["status"] not in ("launching", "cancelled"):
            raise WorkerError("launch was fenced before session binding")
        con.execute("UPDATE assignments SET session_id=?,updated_at=? WHERE assignment_id=?",
                    (session_id, utc_now(), assignment_id))
        epoch = store.require_epoch(con, row["epoch_id"])
        return epoch["status"] == "active" and row["launch_fence"] == epoch["dispatch_fence"] \
            and row["status"] == "launching"


def _dispatch_prompt(store: EpochStore, assignment_id: str, token: str, prompt_id: str,
                     body: dict[str, Any], adapter: Adapter) -> dict[str, Any]:
    """Hold the SQLite writer fence across prompt admission and durable status."""
    with store.transaction() as con:
        row = con.execute("SELECT * FROM assignments WHERE assignment_id=?", (assignment_id,)).fetchone()
        if row is None or row["status"] != "launching" or row["launch_token"] != token:
            raise WorkerError("assignment launch was cancelled or superseded before prompt dispatch")
        epoch = store.require_dispatchable(con, row["epoch_id"])
        if row["launch_fence"] != epoch["dispatch_fence"]:
            raise WorkerError("epoch dispatch fence changed before prompt dispatch")
        if row["prompts_used"] >= row["prompt_budget"]:
            raise WorkerError("assignment prompt budget is exhausted")
        session_id = row["session_id"]
        if not session_id:
            raise WorkerError("assignment has no bound OpenCode session")
        # Cancellation uses BEGIN IMMEDIATE too, so it cannot commit between
        # this final fence check and the external prompt admission.
        response = adapter.send_prompt(session_id, body)
        now = utc_now()
        used = row["prompts_used"] + 1
        con.execute("UPDATE assignments SET status='launched',prompts_used=?,updated_at=? WHERE assignment_id=?",
                    (used, now, assignment_id))
        store._append_event_tx(con, row["epoch_id"], "worker.prompt_dispatched",
                               f"launch:{assignment_id}:prompt:{used}", row["worker_id"], None,
                               {"assignment_id": assignment_id, "session_id": session_id,
                                "prompt_id": prompt_id, "dispatch_fence": epoch["dispatch_fence"]})
        return {"assignment_id": assignment_id, "session_id": session_id, "status": "launched",
                "prompts_used": used, "response": response}


def _mark_launch_uncertain(store: EpochStore, assignment_id: str, token: str, error: str) -> None:
    with store.transaction() as con:
        row = con.execute("SELECT * FROM assignments WHERE assignment_id=?", (assignment_id,)).fetchone()
        if row is None or row["launch_token"] != token or row["status"] not in ("launching", "cancelled"):
            return
        if row["status"] == "cancelled":
            return
        con.execute("UPDATE assignments SET status='launch_uncertain',updated_at=? WHERE assignment_id=?",
                    (utc_now(), assignment_id))
        store._append_event_tx(con, row["epoch_id"], "worker.launch_uncertain", f"launch:{assignment_id}:uncertain",
                               row["worker_id"], None, {"assignment_id": assignment_id, "error": error[:500]})


def launch_assignment(store: EpochStore, assignment_id: str, *, adapter: Adapter | None = None,
                      opencode: str = "opencode", server: str | None = None, execute: bool = False,
                      confirm_launch: bool = False) -> dict[str, Any]:
    """Return a dry-run plan by default; launch only on two explicit flags."""
    plan = build_launch_plan(store, assignment_id, opencode=opencode, server=server)
    if not execute:
        return plan
    if not confirm_launch:
        raise WorkerError("live launch requires both execute=True and confirm_launch=True")
    adapter = adapter or OpenCodeAdapter(opencode, server)
    assignment, worker, hypothesis, token, fence = _begin_launch(store, assignment_id)
    try:
        session_id = adapter.create_session(plan["session_create"]["body"])
    except Exception as exc:
        _mark_launch_uncertain(store, assignment_id, token, str(exc))
        raise WorkerError(f"session creation did not complete cleanly; inspect OpenCode before retrying ({exc})") from None
    can_send = _bind_session(store, assignment_id, token, session_id)
    if not can_send:
        return {"assignment_id": assignment_id, "session_id": session_id, "status": "cancelled_before_prompt",
                "note": "session is dormant; epoch fence prevented model prompt dispatch"}
    prompt_body = plan["session_prompt_template"]["body"]
    try:
        return _dispatch_prompt(store, assignment_id, token, prompt_body["id"], prompt_body, adapter)
    except Exception as exc:
        _mark_launch_uncertain(store, assignment_id, token, str(exc))
        with store.connection() as con:
            current = con.execute("SELECT status,session_id FROM assignments WHERE assignment_id=?",
                                  (assignment_id,)).fetchone()
        if current and current["status"] == "cancelled":
            return {"assignment_id": assignment_id, "session_id": current["session_id"],
                    "status": "cancelled_before_prompt",
                    "note": "epoch cancellation fenced prompt dispatch; session remains dormant"}
        raise WorkerError(f"prompt dispatch may be uncertain; no automatic retry was attempted ({exc})") from None


def continue_assignment(store: EpochStore, assignment_id: str, message: str, *, adapter: Adapter | None = None,
                        opencode: str = "opencode", server: str | None = None, execute: bool = False,
                        confirm_launch: bool = False) -> dict[str, Any]:
    """Continue a launched session in the same worktree/session, never a new branch."""
    if not isinstance(message, str) or not message.strip() or len(message) > _MAX_TEXT:
        raise WorkerError(f"continuation message must contain 1-{_MAX_TEXT} characters")
    assignment, worker, _hypothesis, epoch = _get_assignment(store, assignment_id)
    if assignment["status"] != "launched" or not assignment["session_id"]:
        raise WorkerError("only a launched assignment with a known session can continue")
    _verify_worktree(worker["worktree"], worker["base_commit"], require_clean=False, require_head=False)
    if epoch["status"] != "active":
        raise WorkerError(f"epoch {epoch['status']}; no continuation may be dispatched")
    if not execute:
        return {"dry_run": True, "operation": "session.prompt", "session_id": assignment["session_id"],
                "location": worker["worktree"], "new_session": False, "new_branch": False,
                "next_prompt": assignment["prompts_used"] + 1,
                "prompt_budget": assignment["budget"]["prompts"], "server": server,
                "host_wake_supported": False}
    if not confirm_launch:
        raise WorkerError("live continuation requires both execute=True and confirm_launch=True")
    adapter = adapter or OpenCodeAdapter(opencode, server)
    token = _launch_token()
    with store.transaction() as con:
        row = con.execute("SELECT * FROM assignments WHERE assignment_id=?", (assignment_id,)).fetchone()
        if row is None or row["status"] != "launched" or row["session_id"] != assignment["session_id"]:
            raise WorkerError("assignment changed before continuation reservation")
        epoch_row = store.require_dispatchable(con, row["epoch_id"])
        if row["prompts_used"] >= row["prompt_budget"]:
            raise WorkerError("assignment prompt budget is exhausted")
        con.execute("UPDATE assignments SET status='launching',launch_token=?,launch_fence=?,updated_at=? WHERE assignment_id=?",
                    (token, epoch_row["dispatch_fence"], utc_now(), assignment_id))
    prompt_number = assignment["prompts_used"] + 1
    prompt_id = "msg_" + hashlib.sha256(f"{assignment_id}:{prompt_number}".encode()).hexdigest()[:32]
    body = {"id": prompt_id, "text": message.strip(), "resume": True}
    try:
        return _dispatch_prompt(store, assignment_id, token, prompt_id, body, adapter)
    except Exception as exc:
        _mark_launch_uncertain(store, assignment_id, token, str(exc))
        with store.connection() as con:
            current = con.execute("SELECT status,session_id FROM assignments WHERE assignment_id=?",
                                  (assignment_id,)).fetchone()
        if current and current["status"] == "cancelled":
            return {"assignment_id": assignment_id, "session_id": current["session_id"],
                    "status": "cancelled_before_prompt"}
        raise WorkerError(f"continuation dispatch may be uncertain; no automatic retry was attempted ({exc})") from None


def record_worker_callback(store: EpochStore, *, epoch_id: str, worker_id: str,
                           idempotency_key: str, status: str, summary: str,
                           details: dict[str, Any] | None = None,
                           assignment_id: str | None = None) -> dict[str, Any]:
    """Append an idempotent callback event for manual coordinator delivery."""
    worker = get_worker(store, epoch_id, worker_id)
    if status not in {"done", "blocked", "stopped", "needs_review"}:
        raise WorkerError("callback status must be done, blocked, stopped, or needs_review")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 4000:
        raise WorkerError("callback summary must contain 1-4000 characters")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise WorkerError("callback details must be an object")
    if assignment_id is not None:
        assignment_id = _identifier(assignment_id, "assignment_id")
    with store.transaction() as con:
        store.require_epoch(con, epoch_id)
        current_worker = con.execute("SELECT 1 FROM workers WHERE epoch_id=? AND worker_id=?",
                                     (epoch_id, worker_id)).fetchone()
        if current_worker is None:
            raise WorkerError("worker is not registered in this epoch")
        if assignment_id is not None:
            row = con.execute("SELECT * FROM assignments WHERE assignment_id=? AND epoch_id=? AND worker_id=?",
                              (assignment_id, epoch_id, worker_id)).fetchone()
            if row is None:
                raise WorkerError("assignment does not belong to this worker and epoch")
            if status == "done" and row["status"] in ("launched", "launch_uncertain"):
                con.execute("UPDATE assignments SET status='completed',updated_at=? WHERE assignment_id=?",
                            (utc_now(), assignment_id))
        payload_details = dict(details)
        if assignment_id is not None:
            payload_details["assignment_id"] = assignment_id
        return store._append_event_tx(con, epoch_id, "worker.callback", idempotency_key,
                                      worker_id, worker["callback"]["recipient"],
                                      {"status": status, "summary": summary.strip(), "details": payload_details})


def drain_callbacks(store: EpochStore, epoch_id: str, recipient: str = "coordinator", limit: int = 100) -> list[dict[str, Any]]:
    """Fetch unacked callback packets; repeated drains return the same packet ID."""
    return store.drain_callbacks(epoch_id, recipient, limit)


def acknowledge_callback(store: EpochStore, epoch_id: str, event_id: int,
                         recipient: str = "coordinator") -> bool:
    return store.acknowledge_callback(epoch_id, event_id, recipient)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register bounded workers and create path-scoped assignments.")
    parser.add_argument("--db", help="lane-owned SQLite DB (default: NINFER_BENCH_EPOCH_DB or user state directory)")
    parser.add_argument("--opencode", default="opencode", help="OpenCode v2 CLI executable")
    parser.add_argument("--server", help="explicit OpenCode server URL; never changes global configuration")
    sub = parser.add_subparsers(dest="command", required=True)

    capabilities = sub.add_parser("capabilities", help="read CLI version/help only; no API or credential query")
    capabilities.set_defaults(_no_db=True)

    register = sub.add_parser("register", help="register one exact worktree/model/role/permission record")
    register.add_argument("--epoch-id", required=True)
    register.add_argument("--worker-id", required=True)
    register.add_argument("--worktree", required=True)
    register.add_argument("--base-commit", required=True)
    register.add_argument("--model", required=True, help="provider/model[#variant]")
    register.add_argument("--role", required=True)
    register.add_argument("--allow", action="append", required=True, dest="allowed_files", help="explicit file or dir/**")
    register.add_argument("--read", action="append", required=True, dest="read_paths")
    register.add_argument("--write", action="append", default=[], dest="write_paths")
    register.add_argument("--stop", action="append", required=True, dest="stop_conditions")
    register.add_argument("--callback-recipient", required=True, help="manual pending packet recipient")

    hypothesis = sub.add_parser("hypothesis", help="add a bounded hypothesis")
    hypothesis.add_argument("--epoch-id", required=True)
    hypothesis.add_argument("--hypothesis-id", required=True)
    hypothesis.add_argument("--description", required=True)

    assign = sub.add_parser("assign", help="reserve one bounded assignment; does not launch it")
    assign.add_argument("--epoch-id", required=True)
    assign.add_argument("--assignment-id", required=True)
    assign.add_argument("--worker-id", required=True)
    assign.add_argument("--hypothesis-id", required=True)
    assign.add_argument("--key", required=True, dest="idempotency_key")
    assign.add_argument("--instructions", required=True)
    assign.add_argument("--read", action="append", required=True, dest="read_paths")
    assign.add_argument("--write", action="append", default=[], dest="write_paths")
    assign.add_argument("--token-budget", type=int, required=True)
    assign.add_argument("--cost-budget-usd", required=True)
    assign.add_argument("--wall-budget-seconds", type=int, required=True)
    assign.add_argument("--prompt-budget", type=int, required=True)

    launch = sub.add_parser("launch", help="dry-run by default; use both flags for a live OpenCode session")
    launch.add_argument("--assignment-id", required=True)
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--confirm-launch", action="store_true")

    cont = sub.add_parser("continue", help="continue the same bounded session/worktree")
    cont.add_argument("--assignment-id", required=True)
    cont.add_argument("--message", required=True)
    cont.add_argument("--execute", action="store_true")
    cont.add_argument("--confirm-launch", action="store_true")

    listing = sub.add_parser("list", help="list workers and assignments for an epoch")
    listing.add_argument("--epoch-id", required=True)

    callback = sub.add_parser("callback", help="append one worker callback event for manual delivery")
    callback.add_argument("--epoch-id", required=True)
    callback.add_argument("--worker-id", required=True)
    callback.add_argument("--key", required=True, dest="idempotency_key")
    callback.add_argument("--status", required=True, choices=("done", "blocked", "stopped", "needs_review"))
    callback.add_argument("--summary", required=True)
    callback.add_argument("--assignment-id")

    drain = sub.add_parser("drain", help="manually fetch pending packets; this does not wake their recipient")
    drain.add_argument("--epoch-id", required=True)
    drain.add_argument("--recipient", default="coordinator")
    drain.add_argument("--limit", type=int, default=100)

    ack = sub.add_parser("ack", help="acknowledge a packet after manual consumption")
    ack.add_argument("--epoch-id", required=True)
    ack.add_argument("--event-id", type=int, required=True)
    ack.add_argument("--recipient", default="coordinator")

    recover = sub.add_parser("recover", help="mark interrupted launch attempts uncertain; never auto-retries")
    recover.add_argument("--epoch-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capabilities":
            _print_json(detect_opencode_capabilities(args.opencode))
            return 0
        store = EpochStore(args.db)
        if args.command == "register":
            result = register_worker(
                store, epoch_id=args.epoch_id, worker_id=args.worker_id, worktree=args.worktree,
                base_commit=args.base_commit, model=args.model, role=args.role,
                allowed_files=args.allowed_files, stop_conditions=args.stop_conditions,
                callback={"mode": "manual", "recipient": args.callback_recipient},
                permissions={"read_paths": args.read_paths, "write_paths": args.write_paths,
                             "shell": False, "network": False},
            )
        elif args.command == "hypothesis":
            result = add_hypothesis(store, args.epoch_id, args.hypothesis_id, args.description)
        elif args.command == "assign":
            result = create_assignment(
                store, epoch_id=args.epoch_id, assignment_id=args.assignment_id,
                worker_id=args.worker_id, hypothesis_id=args.hypothesis_id,
                idempotency_key=args.idempotency_key, instructions=args.instructions,
                read_paths=args.read_paths, write_paths=args.write_paths,
                token_budget=args.token_budget, cost_budget_usd=args.cost_budget_usd,
                wall_budget_seconds=args.wall_budget_seconds, prompt_budget=args.prompt_budget,
            )
        elif args.command == "launch":
            adapter = OpenCodeAdapter(args.opencode, args.server) if args.execute else None
            result = launch_assignment(store, args.assignment_id, adapter=adapter, opencode=args.opencode,
                                       server=args.server, execute=args.execute,
                                       confirm_launch=args.confirm_launch)
        elif args.command == "continue":
            adapter = OpenCodeAdapter(args.opencode, args.server) if args.execute else None
            result = continue_assignment(store, args.assignment_id, args.message, adapter=adapter,
                                         opencode=args.opencode, server=args.server, execute=args.execute,
                                         confirm_launch=args.confirm_launch)
        elif args.command == "list":
            result = list_workers(store, args.epoch_id)
        elif args.command == "callback":
            result = record_worker_callback(store, epoch_id=args.epoch_id, worker_id=args.worker_id,
                                            idempotency_key=args.idempotency_key, status=args.status,
                                            summary=args.summary, assignment_id=args.assignment_id)
        elif args.command == "drain":
            result = drain_callbacks(store, args.epoch_id, args.recipient, args.limit)
        elif args.command == "ack":
            result = {"acknowledged": acknowledge_callback(store, args.epoch_id, args.event_id, args.recipient)}
        elif args.command == "recover":
            result = recover_launches(store, args.epoch_id)
        else:
            raise WorkerError(f"unsupported command: {args.command}")
        _print_json(result)
        return 0
    except (WorkerError, EpochError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(f"workers: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
