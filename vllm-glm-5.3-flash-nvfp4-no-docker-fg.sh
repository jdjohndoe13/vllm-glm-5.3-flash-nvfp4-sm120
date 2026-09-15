#!/usr/bin/env bash
# ============================================================================
# vllm-glm-5.3-flash-nvfp4-no-docker-fg.sh — FOREGROUND WRAPPER for the
# no-docker launcher (vllm-glm-5.3-flash-nvfp4-no-docker.sh), so `llmglmfnd`
# can run the 800 GiB hugetlbfs CPU-tier engine in the foreground:
#
#   * `stop` / `status` subcommands pass straight through to the launcher
#     (exec'd in the foreground, no wrapping).
#   * Start (no args, or VAR=value args): first re-checks the launcher's own
#     pidfile (vllm-no-docker.pid, plain pid == pgid thanks to setsid) with
#     kill -0 + a vllm cmdline check. If the engine is already running it
#     prints a hint and exits 0 — never a duplicate engine.
#   * Otherwise it starts the launcher as a BACKGROUND CHILD whose stdout
#     stays on the terminal, so the multi-minute hugepage pre-flight progress
#     streams live. While no NEW logs/engine-*.log has appeared it polls every
#     2 s (one-line hint every ~60 s); when the new engine-<ts>.log appears it
#     tails it in the foreground (`tail -n 30 -F`); when the launcher child
#     exits rc=0 (detached success) it prints a READY banner and keeps
#     tailing. If the child exits non-zero before any new log appears, the
#     wrapper exits with that rc.
#   * Ctrl+C (SIGINT) / SIGTERM: stops the engine CLEANLY by running the
#     launcher's `stop` subcommand (SIGTERM the engine process group, escalate
#     to SIGKILL, then do_shm_cleanup — the same kill + shm cleanup the
#     launcher always does), kills the background child / tail, and exits 0.
#   * A normal wrapper exit does NOT touch the engine: the engine is detached
#     (setsid, own process group, pidfile-tracked) exactly as when started by
#     the bare launcher, so `llmglmfnd status` / `stop` keep working.
#
# Overridable env (for sandbox tests):
#   LLMGLMFND_KIT       kit directory
#                       (default /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4)
#   LLMGLMFND_LAUNCHER  launcher script path
#                       (default $KIT_DIR/vllm-glm-5.3-flash-nvfp4-no-docker.sh)
#   LLMGLMFND_LOGDIR    engine log directory (default $KIT_DIR/logs)
# ============================================================================
set -u

TAG="llmglmfnd-fg"
KIT_DIR="${LLMGLMFND_KIT:-/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4}"
LAUNCHER="${LLMGLMFND_LAUNCHER:-$KIT_DIR/vllm-glm-5.3-flash-nvfp4-no-docker.sh}"
LOGDIR="${LLMGLMFND_LOGDIR:-$KIT_DIR/logs}"
PIDFILE="$KIT_DIR/vllm-no-docker.pid"

# --- stop/status: pass straight through to the launcher, in the foreground ---
case "${1-}" in
  stop|status) exec bash "$LAUNCHER" "$@" ;;
esac

newest_engine_log() {
  ls -t "$LOGDIR"/engine-*.log 2>/dev/null | head -n 1 || true
}

# --- already-running guard (the launcher's own semantics: pidfile pid alive
#     AND a vllm cmdline). Stale pidfile is removed exactly like the launcher's
#     start path does. ---
if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null \
     && grep -aq vllm "/proc/${OLD_PID}/cmdline" 2>/dev/null; then
    echo "$TAG: engine already running (pid $OLD_PID) — nothing to start."
    echo "        follow the log:  llmglmfnd status"
    echo "        stop the engine: llmglmfnd stop"
    exit 0
  fi
  echo "$TAG: stale pidfile $PIDFILE (pid ${OLD_PID:-?} not alive) — removing."
  rm -f "$PIDFILE"
fi

CHILD_PID=""
TAIL_PID=""
READY_DONE=0

# --- INT/TERM handler: clean engine stop via the launcher's own `stop` ---
stopping() {
  trap - INT TERM          # disable the trap first (avoid recursion)
  echo "$TAG: Ctrl+C — stopping the engine..."
  if [ -n "$TAIL_PID" ]; then
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
  fi
  bash "$LAUNCHER" stop    # kills the pidfile pgid + runs do_shm_cleanup
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  echo "$TAG: engine stopped (launcher stop = process-group kill + shm cleanup)."
  exit 0
}
trap stopping INT TERM

# --- start path: launcher as background child, stdout still on the terminal
#     so the hugepage pre-flight progress streams live ---
BEFORE="$(newest_engine_log)"
echo "$TAG: starting the no-docker engine (800 GiB hugetlbfs CPU tier) via $LAUNCHER"
CPU_TIER_HUGETLB=1 CPU_TIER_GB=800 bash "$LAUNCHER" "$@" &
CHILD_PID=$!

# Phase 1 — until a NEW engine-*.log (name != $BEFORE) appears: poll every 2 s.
_iter=0
while :; do
  NEW_LOG="$(newest_engine_log)"
  if [ -n "$NEW_LOG" ] && [ "$NEW_LOG" != "$BEFORE" ]; then
    break
  fi
  if ! kill -0 "$CHILD_PID" 2>/dev/null; then
    wait "$CHILD_PID" 2>/dev/null
    CHILD_RC=$?
    NEW_LOG="$(newest_engine_log)"   # log may have appeared right before death
    if [ -n "$NEW_LOG" ] && [ "$NEW_LOG" != "$BEFORE" ]; then
      break
    fi
    if [ "$CHILD_RC" -ne 0 ]; then
      echo "$TAG: launcher exited rc=$CHILD_RC before creating a new engine log — aborting."
      exit "$CHILD_RC"
    fi
    if [ "$READY_DONE" -eq 0 ]; then
      READY_DONE=1
      echo "$TAG: engine READY and detached — following the log; Ctrl+C stops the engine."
    fi
  fi
  _iter=$((_iter + 1))
  if [ $((_iter % 30)) -eq 0 ]; then
    echo "$TAG: pre-flight running (hugepage reservation can take minutes)..."
  fi
  sleep 2
done

# Phase 2 — follow the new log in the foreground until Ctrl+C.
echo "$TAG: following $NEW_LOG (Ctrl+C stops the engine)"
tail -n 30 -F "$NEW_LOG" &
TAIL_PID=$!

while :; do
  if [ "$READY_DONE" -eq 0 ] && ! kill -0 "$CHILD_PID" 2>/dev/null; then
    wait "$CHILD_PID" 2>/dev/null
    CHILD_RC=$?
    READY_DONE=1
    if [ "$CHILD_RC" -eq 0 ]; then
      echo "$TAG: engine READY and detached — following the log; Ctrl+C stops the engine."
    else
      echo "$TAG: launcher exited rc=$CHILD_RC — engine may have failed to start; Ctrl+C stops the engine."
    fi
  fi
  if ! kill -0 "$TAIL_PID" 2>/dev/null; then
    wait "$TAIL_PID" 2>/dev/null
    echo "$TAG: log follow ended — exiting (engine untouched; check with: llmglmfnd status)."
    exit 0
  fi
  sleep 1
done
