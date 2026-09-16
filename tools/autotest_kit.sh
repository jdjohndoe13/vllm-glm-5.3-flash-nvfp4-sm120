#!/bin/bash
# ============================================================================
# autotest_kit.sh — BLOCKING one-shot research cycle (run on testcomp2):
#
#   ssh testcomp2 'bash /tmp/autotest_kit.sh'
#
# Cycle:
#   P0  stop dockerized vLLM (container, NO rm — reused for restore),
#       non-dockerized kit engine, and sglang; wait for GPUs + port free.
#   P1  start the non-dockerized GLM kit engine (llmglmfnd semantics:
#       bare launcher detached) and wait for health + fingerprint gate.
#   P2  spawn an independent crash watchdog (setsid): two consecutive
#       failed health polls (~20 s, or boot never healthy within 16 min)
#       -> capture engine-log + swap_diag evidence, run the launcher's own
#       clean `stop`, then `docker start` the orig container and wait for
#       its health (~20 min cap). Watchdog exits when the parent writes
#       /tmp/autotest-verdict/FINAL.
#   P3  run the preempt-flush storm (test_preempt_flush.py, 15 min cap).
#   P4  verdict:
#       storm green            -> leave kit running, FINAL=KIT_SURVIVED, exit 0
#       crash (watchdog acted) -> FINAL=TEST_CRASHED_ORIG_RESTORED, exit 1
#       boot never healthy     -> watchdog path too, exit 2
#       storm FAILED/timeout but engine ALIVE -> leave kit running,
#                                                exit 3 / 4 (ambiguous)
#
# All long-lived things (engine, watchdog) are setsid'd and survive an ssh
# drop; the parent script is the only blocking piece.
# ============================================================================
set -uo pipefail

PORT=1025
KIT_DIR=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4
LAUNCHER="$KIT_DIR/vllm-glm-5.3-flash-nvfp4-no-docker.sh"
ORIG_CONTAINER=vllm-glm-5.3-flash-nvfp4
LOGDIR="$KIT_DIR/logs"
STORM="$KIT_DIR/test_preempt_flush.py"
DIAG_DIR="$KIT_DIR/swap_diag"
VDIR=/tmp/autotest-verdict
MONITOR=/tmp/autotest_monitor.sh
HEALTH="http://127.0.0.1:${PORT}/health"
BASE=http://127.0.0.1:${PORT}
BOOT_TIMEOUT=900
STORM_TIMEOUT=900
ORIG_RESTORE_TIMEOUT=1200
GPU_FREE_TIMEOUT=360
GPU_FREE_MIB=4096

mkdir -p "$VDIR"   # every later redirect targets $VDIR/* — must exist FIRST

say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
health(){ curl -sf --max-time 4 "$HEALTH" >/dev/null 2>&1; }
newest_log(){ ls -t "$LOGDIR"/engine-*.log 2>/dev/null | head -n 1; }
metrics_kv(){ curl -s --max-time 6 "$BASE/metrics" 2>/dev/null | grep -ac 'kv_offload' || echo 0; }
gpu_busy(){
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -v t="$GPU_FREE_MIB" '$1+0 > t {f=1} END{exit !f}'
}
wait_gpu_free(){
  local w=0
  while gpu_busy; do
    (( w >= GPU_FREE_TIMEOUT )) && { say "WARN: GPUs still busy after ${GPU_FREE_TIMEOUT}s"; return 1; }
    sleep 5; w=$((w+5))
  done
  say "GPUs free"
  return 0
}

# --------------------------------------------------------------------------
# P0 — stop every server variant, free GPUs and the port
# --------------------------------------------------------------------------
say "P0: stopping all servers (dockerized vLLM, non-docker kit, sglang)"
docker stop "$ORIG_CONTAINER" >/dev/null 2>&1 && say "dockerized vLLM container stopped" \
  || say "dockerized vLLM container: already stopped/absent"
# NOTE: never `docker rm` — the same container is started back for restore.

if [ -f "$KIT_DIR/vllm-no-docker.pid" ] || pgrep -f 'vllm-glm-5.3-flash-nvfp4-no-docke[r]' >/dev/null 2>&1; then
  bash "$LAUNCHER" stop >> "$VDIR/p0-kit-stop.log" 2>&1 \
    && say "kit launcher stop done" || say "kit launcher stop returned non-zero (see p0-kit-stop.log)"
fi

SGLANG_ROOT=/mnt/data/shared/models/sglang-flashnext-sm120
pkill -f 'serve_best_kv_offload[.]sh'      2>/dev/null
pkill -f 'sglang serv[e]'                  2>/dev/null
pkill -f "$SGLANG_ROOT/sglang-official/.venv/bin/sglang" 2>/dev/null
pkill -f 'scripts/serve[.]sh'              2>/dev/null
sleep 3
pkill -9 -f 'sglang serv[e]'               2>/dev/null || true

wait_gpu_free || true

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  say "FATAL: port ${PORT} still bound after P0 — refusing to boot."
  bash "$LAUNCHER" stop >/dev/null 2>&1 || true
  docker start "$ORIG_CONTAINER" >/dev/null 2>&1 || true
  exit 9
fi
say "P0 done: GPUs free, port ${PORT} free"

# --------------------------------------------------------------------------
# P1 — boot the non-dockerized kit engine (llmglmfnd semantics, detached)
# --------------------------------------------------------------------------
say "P1: booting kit engine via $LAUNCHER (detached)"
setsid nohup env CPU_TIER_HUGETLB=1 CPU_TIER_GB=800 \
  bash "$LAUNCHER" > "$VDIR/kit-boot.log" 2>&1 < /dev/null &
LAUNCHER_JOB=$!

w=0; up=0
while (( w < BOOT_TIMEOUT )); do
  if health; then up=1; break; fi
  kill -0 "$LAUNCHER_JOB" 2>/dev/null || break   # launcher script exited
  sleep 5; w=$((w+5))
done
if (( ! up )); then
  sleep 5
  health && up=1
fi
if (( ! up )); then
  say "P1 FAILED: engine not healthy (waited ${w}s of ${BOOT_TIMEOUT}s cap) -> recovery path"
  [ -f "$VDIR/kit-boot.log" ] \
    && tail -n 8 "$VDIR/kit-boot.log" > "$VDIR/p1-boot-fail-human.log"
  bash "$LAUNCHER" stop >> "$VDIR/p1-boot-fail-stop.log" 2>&1
  sleep 5
  wait_gpu_free || true
  docker start "$ORIG_CONTAINER" >> "$VDIR/p1-restore.log" 2>&1 || true
  ow=0; oup=0
  while (( ow < ORIG_RESTORE_TIMEOUT )); do
    health && { oup=1; break; }
    sleep 10; ow=$((ow+10))
  done
  echo "FINAL=BOOT_FAILED_ORIG_${oup:+RESTORED}${oup:-DEAD}" > "$VDIR/FINAL"
  say "### RESULT: BOOT_FAILED orig restored=$oup"
  exit 2
fi
FP=$(metrics_kv)
say "P1: kit engine healthy, kv_offload metrics=$FP"
if (( FP < 50 )); then
  say "P1 GATE FAILED: metrics=$FP is not the kit fingerprint — wrong engine?"
  bash "$LAUNCHER" stop >> "$VDIR/p1-gate-stop.log" 2>&1
  docker start "$ORIG_CONTAINER" >> "$VDIR/p1-restore.log" 2>&1 || true
  echo "FINAL=FINGERPRINT_GATE_FAILED" > "$VDIR/FINAL"
  exit 2
fi
NEWLOG=$(newest_log)
grep -E 'row-aligned|pinned in|cudaHostRegister rank' "$NEWLOG" 2>/dev/null | tail -n 3 \
  > "$VDIR/pin-lines.txt" || true
say "P1: new engine log: $NEWLOG (pin lines captured)"

# --------------------------------------------------------------------------
# P2 — independent watchdog (crash -> evidence -> kit stop -> orig restore)
# --------------------------------------------------------------------------
cat > "$MONITOR" <<'MONEOF'
#!/bin/bash
# crash watchdog: parent-independent; exits when /tmp/autotest-verdict/FINAL
# appears; on crash: evidence -> launcher stop -> docker start orig.
set -uo pipefail
KIT_DIR=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4
LAUNCHER="$KIT_DIR/vllm-glm-5.3-flash-nvfp4-no-docker.sh"
ORIG_CONTAINER=vllm-glm-5.3-flash-nvfp4
LOGDIR="$KIT_DIR/logs"
DIAG_DIR="$KIT_DIR/swap_diag"
VDIR=/tmp/autotest-verdict
mkdir -p "$VDIR"
HEALTH="http://127.0.0.1:1025/health"
ORIG_RESTORE_TIMEOUT=1200
echo $$ > "$VDIR/monitor.pid"
say(){ echo "[mon $(date -u +%H:%M:%S)] $*" >> "$VDIR/monitor.log"; }
health(){ curl -sf --max-time 4 "$HEALTH" >/dev/null 2>&1; }
newest_log(){ ls -t "$LOGDIR"/engine-*.log 2>/dev/null | head -n 1; }
take_over(){
  [ -f "$VDIR/FINAL" ] && exit 0     # parent finished first
  touch "$VDIR/MONITOR_TOOK_OVER"
  say "CRASH detected"
  local nl; nl="$(newest_log)"
  [ -n "$nl" ] && tail -n 400 "$nl" > "$VDIR/crash-engine-tail.log" 2>/dev/null
  grep -aE 'swap_diag|Traceback|CUDA error|EngineCore|kv_offload' \
    "$VDIR/crash-engine-tail.log" 2>/dev/null | tail -n 40 \
    > "$VDIR/crash-grep.log" 2>/dev/null || true
  ls -t "$DIAG_DIR"/log_swap_T*.jsonl 2>/dev/null | head -n 8 \
    | xargs -r -I{} cp {} "$VDIR/" 2>/dev/null || true
  say "stopping kit engine (launcher stop)"
  bash "$LAUNCHER" stop > "$VDIR/kit-stop.log" 2>&1
  sleep 5
  pgrep -f 'vllm-glm-5.3-flash-nvfp4-no-docke[r]' >/dev/null 2>&1 \
    && pkill -9 -f 'vllm-glm-5.3-flash-nvfp4-no-docke[r]' 2>/dev/null
  say "restoring orig container $ORIG_CONTAINER"
  docker start "$ORIG_CONTAINER" >> "$VDIR/orig-restore.log" 2>&1
  ow=0; oup=0
  while (( ow < ORIG_RESTORE_TIMEOUT )); do
    health && { oup=1; break; }
    sleep 10; ow=$((ow+10))
  done
  say "orig restored=$oup"
  echo "FINAL=TEST_CRASHED_ORIG_${oup:+RESTORED}${oup:-DEAD}" > "$VDIR/FINAL"
  exit 0
}
healthy_seen=0; fails=0; boot_wait=0
while :; do
  [ -f "$VDIR/FINAL" ] && exit 0
  if health; then
    healthy_seen=1; fails=0
  else
    if (( healthy_seen )); then
      fails=$((fails+1))
      (( fails >= 2 )) && take_over
    else
      boot_wait=$((boot_wait+10))
      (( boot_wait >= 960 )) && take_over
    fi
  fi
  sleep 10
done
MONEOF
chmod +x "$MONITOR"
setsid nohup bash "$MONITOR" > "$VDIR/monitor.stdout" 2>&1 < /dev/null &
say "P2: watchdog spawned (pid $(cat "$VDIR/monitor.pid" 2>/dev/null || echo '?'))"

# --------------------------------------------------------------------------
# P3 — preempt-flush storm against the fresh engine
# --------------------------------------------------------------------------
say "P3: firing storm (cap ${STORM_TIMEOUT}s)"
STORM_RC=0
timeout "$STORM_TIMEOUT" bash -c "cd '$KIT_DIR' && python3 '$STORM'" \
    > "$VDIR/storm.log" 2>&1 || STORM_RC=$?
say "P3: storm rc=$STORM_RC"

# --------------------------------------------------------------------------
# P4 — verdict
# --------------------------------------------------------------------------
TOOK_OVER=0
[ -f "$VDIR/MONITOR_TOOK_OVER" ] && TOOK_OVER=1

if (( STORM_RC == 0 )) && ! grep -q 'FAIL' "$VDIR/storm.log" 2>/dev/null; then
  say "P4: storm GREEN — leaving kit engine running"
  echo "FINAL=KIT_SURVIVED" > "$VDIR/FINAL"
  sleep 11   # let the watchdog observe FINAL and exit
  FP=$(metrics_kv)
  say "final health: $(health && echo 200 || echo DOWN), kv_offload metrics=$FP"
  say "### RESULT: KIT_SURVIVED"
  exit 0
fi

if (( TOOK_OVER )); then
  say "P4: storm ended while watchdog restored orig (crash class)"
  say "### RESULT: TEST_CRASHED_ORIG_RESTORED (see $VDIR)"
  exit 1
fi

# storm failed but the engine is still healthy — ambiguous, keep kit alive
if health; then
  if (( STORM_RC == 124 )); then
    say "P4: storm TIMED OUT but engine alive — leaving kit running"
    echo "FINAL=STORM_TIMEOUT_ENGINE_ALIVE" > "$VDIR/FINAL"
    say "### RESULT: STORM_TIMEOUT_ENGINE_ALIVE"
    exit 4
  fi
  say "P4: storm FAILED but engine alive — leaving kit running"
  echo "FINAL=STORM_FAILED_ENGINE_ALIVE" > "$VDIR/FINAL"
  say "### RESULT: STORM_FAILED_ENGINE_ALIVE"
  exit 3
fi

# engine unhealthy and watchdog missing action — force recovery now
say "P4: engine unhealthy without watchdog takeover — forcing recovery"
bash "$LAUNCHER" stop >> "$VDIR/p4-force-stop.log" 2>&1 || true
docker start "$ORIG_CONTAINER" >> "$VDIR/p4-restore.log" 2>&1 || true
ow=0; oup=0
while (( ow < ORIG_RESTORE_TIMEOUT )); do
  health && { oup=1; break; }
  sleep 10; ow=$((ow+10))
done
echo "FINAL=ENGINE_DOWN_ORIG_${oup:+RESTORED}${oup:-DEAD}" > "$VDIR/FINAL"
say "### RESULT: ENGINE_DOWN_ORIG_${oup:+RESTORED}${oup:-DEAD}"
exit 1
