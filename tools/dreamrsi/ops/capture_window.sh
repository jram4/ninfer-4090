#!/usr/bin/env bash
# Start, stop, or inspect one labeled capture window on the FNN build host.
#
# Each window is one independent coding session for corpus purposes: it gets its
# own session label, its own private capsule directory, and its own descriptor
# file. Captured bodies never leave the private research root on fnn.
#
# Usage:
#   capture_window.sh start <session-label> [max-seconds] [max-requests]
#   capture_window.sh stop
#   capture_window.sh status
set -euo pipefail

ALIAS="${NINFER_CAPTURE_ALIAS:-fnn}"
PORT="${NINFER_CAPTURE_PORT:-18082}"
REMOTE_ROOT="/home/aramirezfamily/.local/share/ninfer-bench"
REMOTE_CAPTURE="${REMOTE_ROOT}/capture-setup-c4deb9c/capture.py"
REMOTE_PYTHON="/home/aramirezfamily/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
UPSTREAM="${NINFER_CAPTURE_UPSTREAM:-http://100.119.241.20:8080}"
PID_FILE="/tmp/opencode/ninfer-capture-window.pid"
LOG_FILE="/tmp/opencode/ninfer-capture-window.log"

state() { printf '%s\n' "$1" > "${PID_FILE}.state"; }
read_state() { cat "${PID_FILE}.state" 2>/dev/null || true; }

start() {
  local label="${1:?session label required}"
  local max_seconds="${2:-7200}"
  local max_requests="${3:-4000}"
  local safe
  safe="$(printf '%s' "${label}" | tr -c 'A-Za-z0-9._-' '_')"
  local remote_dir="${REMOTE_ROOT}/live-capture/${safe}"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${ALIAS}" \
    "umask 077; mkdir -p '${remote_dir}'; chmod 700 '${remote_dir}'; \
     setsid nohup '${REMOTE_PYTHON}' '${REMOTE_CAPTURE}' \
       --upstream '${UPSTREAM}' \
       --capture-dir '${remote_dir}/capsules' \
       --descriptor-jsonl '${remote_dir}/descriptors.jsonl' \
       --listen-host 100.119.241.20 --listen-port ${PORT} \
       --session-id '${safe}' --task-label 'live-coding' --run-label 'live-capture' \
       --traffic-class real_user --stratum 'live-coding' --cache-state unknown \
       --max-requests ${max_requests} --max-seconds ${max_seconds} \
       --max-active-connections 4 --timeout 900 \
       > '${remote_dir}/capture.log' 2>&1 < /dev/null & echo \$! > '${remote_dir}/capture.pid'"
  local remote_pid
  remote_pid="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${ALIAS}" "cat '${remote_dir}/capture.pid' 2>/dev/null || true")"
  state "${remote_dir} ${remote_pid} ${label}"
  printf 'capture window started: label=%s remote_dir=%s pid=%s port=%s\n' \
    "${label}" "${remote_dir}" "${remote_pid}" "${PORT}"
  printf 'point the coding client at: http://100.119.241.20:%s\n' "${PORT}"
}

stop() {
  local current
  current="$(read_state)"
  local remote_dir remote_pid
  remote_dir="$(printf '%s' "${current}" | cut -d' ' -f1)"
  remote_pid="$(printf '%s' "${current}" | cut -d' ' -f2)"
  if [ -n "${remote_dir}" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=15 "${ALIAS}" \
      "if [ -f '${remote_dir}/capture.pid' ]; then kill '\$(cat '${remote_dir}/capture.pid')' 2>/dev/null || true; fi" || true
  fi
  printf 'capture window stopped: %s (pid %s)\n' "${remote_dir}" "${remote_pid:-unknown}"
  rm -f "${PID_FILE}.state"
}

status() {
  local current
  current="$(read_state)"
  if [ -z "${current}" ]; then
    printf 'no local capture window state\n'
    return 0
  fi
  printf '%s\n' "${current}"
  local remote_dir
  remote_dir="$(printf '%s' "${current}" | cut -d' ' -f1)"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${ALIAS}" \
    "if [ -f '${remote_dir}/descriptors.jsonl' ]; then wc -l < '${remote_dir}/descriptors.jsonl'; else echo 0; fi" || true
}

case "${1:-status}" in
  start) shift; start "$@" ;;
  stop) stop ;;
  status) status ;;
  *) printf 'usage: %s {start <label> [max-seconds] [max-requests]|stop|status}\n' "$0" >&2; exit 2 ;;
esac
