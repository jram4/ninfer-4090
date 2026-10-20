#!/usr/bin/env bash
# Start, stop, or inspect labelled capture windows on the FNN build host.
#
# One window is one independent session for corpus purposes: its own session
# label, its own private capsule directory, its own descriptor file, and its
# own stratum label. Captured bodies never leave the private research root.
#
# Usage:
#   capture_window.sh start <label> [stratum] [max-seconds] [max-requests]
#   capture_window.sh stop
#   capture_window.sh status
#   capture_window.sh windows
#   capture_window.sh merge <manifest.jsonl>
set -euo pipefail

ALIAS="${NINFER_CAPTURE_ALIAS:-fnn}"
PORT="${NINFER_CAPTURE_PORT:-18082}"
REMOTE_ROOT="/home/aramirezfamily/.local/share/ninfer-bench"
REMOTE_CAPTURE="${REMOTE_ROOT}/capture-setup-c4deb9c/capture.py"
REMOTE_PYTHON="/home/aramirezfamily/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
UPSTREAM="${NINFER_CAPTURE_UPSTREAM:-http://100.119.241.20:8080}"
STATE_FILE="/tmp/opencode/ninfer-capture-window.state"
OPENCODE_CONFIG="${HOME}/.config/opencode/opencode.json"
CLAUDE_CONFIG="${HOME}/.claude/settings.json"

remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${ALIAS}" "$@"; }

safe_label() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }

# --- client redirection -------------------------------------------------------
# A window only captures traffic that is actually pointed at it. The clients
# keep their own base URL, so a window temporarily redirects the two known
# client configurations at the capture port and restores them on stop.

config_hash() {
  if [ -f "$1" ]; then
    sha256sum "$1" | cut -d' ' -f1
  else
    echo "absent"
  fi
}

restore_configs() {
  local state="$1"
  local restored=0
  for entry in $(grep -oE '(opencode|claude)=[^ ]+' "${state}" 2>/dev/null || true); do
    local kind="${entry%%=*}"
    local backup="${entry#*=}"
    case "${kind}" in
      opencode) target="${OPENCODE_CONFIG}" ;;
      claude) target="${CLAUDE_CONFIG}" ;;
      *) continue ;;
    esac
    if [ -f "${backup}" ]; then
      mkdir -p "$(dirname "${target}")"
      cp -p "${backup}" "${target}"
      rm -f "${backup}"
      restored=$((restored + 1))
    fi
  done
  echo "${restored}"
}

redirect_configs() {
  local state="$1"
  : > "${state}"
  if [ -f "${OPENCODE_CONFIG}" ]; then
    local backup="${OPENCODE_CONFIG}.capture-backup"
    cp -p "${OPENCODE_CONFIG}" "${backup}"
    OPENCODE_CONFIG="${OPENCODE_CONFIG}" python3.11 - "${OPENCODE_CONFIG}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
provider = data.get("providers", {}).get("fnn")
if isinstance(provider, dict):
    settings = provider.setdefault("settings", {})
    settings["baseURL"] = "http://100.119.241.20:18082/v1"
temporary = path.with_suffix(path.suffix + ".capture-tmp")
temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
    echo "opencode=${backup}" >> "${state}"
  fi
  if [ -f "${CLAUDE_CONFIG}" ]; then
    local backup="${CLAUDE_CONFIG}.capture-backup"
    cp -p "${CLAUDE_CONFIG}" "${backup}"
    CLAUDE_CONFIG="${CLAUDE_CONFIG}" python3.11 - "${CLAUDE_CONFIG}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
env = data.get("env")
if isinstance(env, dict) and "ANTHROPIC_BASE_URL" in env:
    env["ANTHROPIC_BASE_URL"] = "http://100.119.241.20:18082"
temporary = path.with_suffix(path.suffix + ".capture-tmp")
temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
    echo "claude=${backup}" >> "${state}"
  fi
}

# Return the PID of the proxy that currently owns the capture port, if any.
# The socket owner is used rather than a process-name match, because a
# pattern match also matches the shell that runs the pattern.
listening_pid() {
  remote "ss -ltnp 2>/dev/null | awk '/:${PORT}[[:space:]]/ {print}' \
    | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2" 2>/dev/null || true
}

require_free_port() {
  local pid
  pid="$(listening_pid)"
  if [ -n "${pid}" ]; then
    printf 'ERROR: capture port %s is already in use by proxy pid %s.\n' "${PORT}" "${pid}" >&2
    printf 'Stop the running window first: bash %s stop\n' "$0" >&2
    exit 1
  fi
}

start() {
  local label="${1:?session label required}"
  local stratum="${2:-live-coding}"
  local max_seconds="${3:-7200}"
  local max_requests="${4:-2000}"
  local safe
  safe="$(safe_label "${label}")"
  # A previous window that was never stopped may still hold the port or the
  # client redirection. Recover it before starting a new one.
  if [ -f "${STATE_FILE}" ]; then
    printf 'recovering an unfinished previous window\n' >&2
    restore_configs "${STATE_FILE}" >/dev/null 2>&1 || true
    local stale
    stale="$(listening_pid)"
    if [ -n "${stale}" ]; then
      remote "kill ${stale} 2>/dev/null || true"
      sleep 1
    fi
    rm -f "${STATE_FILE}"
  fi
  require_free_port
  local remote_dir="${REMOTE_ROOT}/live-capture/${safe}"
  remote "umask 077; mkdir -p '${remote_dir}'; chmod 700 '${remote_dir}'; \
    rm -f '${remote_dir}/descriptors.jsonl'"
  remote "setsid nohup '${REMOTE_PYTHON}' '${REMOTE_CAPTURE}' \
      --upstream '${UPSTREAM}' \
      --capture-dir '${remote_dir}/capsules' \
      --descriptor-jsonl '${remote_dir}/descriptors.jsonl' \
      --listen-host 100.119.241.20 --listen-port ${PORT} \
      --session-id '${safe}' --task-label '${safe}' --run-label 'live-capture' \
      --traffic-class real_user --stratum '${safe}' --cache-state unknown \
      --max-requests ${max_requests} --max-seconds ${max_seconds} \
      --max-active-connections 4 --timeout 900 \
      > '${remote_dir}/capture.log' 2>&1 < /dev/null & echo \$! > '${remote_dir}/capture.pid'"

  # The window only counts as started once the proxy owns the port.
  local waited=0 pid=""
  while [ "${waited}" -lt 20 ]; do
    pid="$(listening_pid)"
    if [ -n "${pid}" ]; then
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  if [ -z "${pid}" ]; then
    printf 'ERROR: the capture proxy did not start. Remote log:\n' >&2
    remote "tail -20 '${remote_dir}/capture.log'" >&2 || true
    exit 1
  fi
  printf '%s\n' "${safe} ${pid} ${stratum} ${remote_dir}" > "${STATE_FILE}"
  redirect_configs "${STATE_FILE}"
  printf 'capture window STARTED\n'
  printf '  label    : %s\n' "${safe}"
  printf '  stratum  : %s\n' "${safe}"
  printf '  pid      : %s\n' "${pid}"
  printf '  endpoint : http://100.119.241.20:%s\n' "${PORT}"
  printf '  clients  : OpenCode and Claude Code now point at the capture port\n'
  printf '  NOTE     : start the client AFTER this, so it reads the new endpoint\n'
  printf '  when done: bash %s stop\n' "$0"
}

stop() {
  local pid
  pid="$(listening_pid)"
  if [ -n "${pid}" ]; then
    remote "kill ${pid} 2>/dev/null || true"
    local waited=0
    while [ "${waited}" -lt 15 ] && [ -n "$(listening_pid)" ]; do
      sleep 1
      waited=$((waited + 1))
    done
    if [ -n "$(listening_pid)" ]; then
      remote "kill -9 ${pid} 2>/dev/null || true"
      sleep 1
    fi
    printf 'capture window STOPPED (pid %s)\n' "${pid}"
  else
    printf 'no capture window was running\n'
  fi
  if [ -f "${STATE_FILE}" ]; then
    local restored
    restored="$(restore_configs "${STATE_FILE}")"
    printf 'client configuration restored (%s file(s))\n' "${restored}"
    rm -f "${STATE_FILE}"
  fi
}

restore() {
  if [ ! -f "${STATE_FILE}" ]; then
    printf 'no capture window state to restore\n'
    return 0
  fi
  local restored
  restored="$(restore_configs "${STATE_FILE}")"
  rm -f "${STATE_FILE}"
  printf 'client configuration restored (%s file(s))\n' "${restored}"
}

status() {
  local pid
  pid="$(listening_pid)"
  if [ -z "${pid}" ]; then
    printf 'no capture window is running\n'
  else
    printf 'capture window RUNNING (pid %s) on port %s\n' "${pid}" "${PORT}"
    if [ -f "${STATE_FILE}" ]; then
      cat "${STATE_FILE}"
    fi
  fi
  remote "for d in '${REMOTE_ROOT}'/live-capture/*/; do \
    n=\$(wc -l < \"\$d/descriptors.jsonl\" 2>/dev/null || echo 0); \
    printf '  %s %s requests\n' \"\$(basename \"\$d\")\" \"\$n\"; done" 2>/dev/null || true
}

windows() {
  remote "for d in '${REMOTE_ROOT}'/live-capture/*/; do \
    n=\$(wc -l < \"\$d/descriptors.jsonl\" 2>/dev/null || echo 0); \
    printf '%s %s\n' \"\$(basename \"\$d\")\" \"\$n\"; done" 2>/dev/null || true
}

merge() {
  local manifest="${1:?output manifest path required}"
  remote "cat '${REMOTE_ROOT}'/live-capture/*/descriptors.jsonl" > "${manifest}"
  local count
  count="$(wc -l < "${manifest}")"
  printf 'merged %s descriptors into %s\n' "${count}" "${manifest}"
  printf 'sessions:\n'
  python3.11 - "${manifest}" <<'PY'
import json
import sys
from collections import Counter

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
for session, count in sorted(Counter(row.get("session_id", "?") for row in rows).items()):
    print(f"  {session}: {count}")
PY
}

case "${1:-status}" in
  start) shift; start "$@" ;;
  stop) stop ;;
  restore) restore ;;
  status) status ;;
  windows) windows ;;
  merge) shift; merge "$@" ;;
  *) printf 'usage: %s {start <label> [stratum] [max-seconds] [max-requests]|stop|status|windows|merge <manifest>}\n' "$0" >&2; exit 2 ;;
esac
