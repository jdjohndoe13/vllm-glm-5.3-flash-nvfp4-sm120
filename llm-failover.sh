#!/usr/bin/env bash
# llm-failover.sh — ensure exactly one LLM server is up on :1025.
#
# Flow:
#   1. stop BOTH servers (vLLM container + sglang, whichever is running), free GPUs
#   2. boot GLM (vLLM) via the launcher; watch for READY vs DEAD
#   3. if vLLM dies -> stop its container (clears the unless-stopped crash loop),
#      free the GPUs, and start sglang detached (survives this script exiting)
#
# Never runs both at once (both bind :1025). Ends with a server running when
# possible; prints a machine-parseable "### RESULT:" line and the log paths.
#
# Env overrides: VLLM_BOOT_TIMEOUT / SGLANG_BOOT_TIMEOUT / GPU_FREE_TIMEOUT /
# GPU_FREE_MIB / SERVED_MODEL_NAMES.
#
# Run on the server:  bash /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/llm-failover.sh
# Detached (recommended):  setsid nohup bash .../llm-failover.sh >/tmp/llm-failover.out 2>&1 </dev/null &
set -uo pipefail

PORT="${PORT:-1025}"
GLM_DIR="/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4"
GLM_LAUNCHER="$GLM_DIR/vllm-glm-5.3-flash-nvfp4.sh"
GLM_CONTAINER="vllm-glm-5.3-flash-nvfp4"
SGLANG_LAUNCHER="/mnt/data/shared/models/sglang-flashnext-sm120/scripts/serve_best_kv_offload.sh"
SGLANG_ROOT="/mnt/data/shared/models/sglang-flashnext-sm120"
SGLANG_LOG="$SGLANG_ROOT/logs/serve.log"
SGLANG_PIDFILE="/tmp/llm-failover-sglang.pid"
VLLM_SERVED_NAMES="${SERVED_MODEL_NAMES:-glm-5.3-flash qwen-3.8-flash-next}"
VLLM_BOOT_TIMEOUT="${VLLM_BOOT_TIMEOUT:-900}"
SGLANG_BOOT_TIMEOUT="${SGLANG_BOOT_TIMEOUT:-1200}"
GPU_FREE_TIMEOUT="${GPU_FREE_TIMEOUT:-180}"
GPU_FREE_MIB="${GPU_FREE_MIB:-4096}"
FORCE="${FORCE:-1}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
LOG_DIR="/tmp/llm-failover"
mkdir -p "$LOG_DIR"

FATAL_RE='Engine core initialization failed|EngineCore failed to start|RuntimeError: Engine core|ValueError: To serve at least one request|No available memory for the cache|out of memory|OutOfMemoryError'

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
result(){ echo "### RESULT: $*"; }
health_ok(){ curl -sf --max-time 4 "$HEALTH_URL" >/dev/null 2>&1; }
container_state(){ docker inspect -f '{{.State.Status}}' "$GLM_CONTAINER" 2>/dev/null || echo none; }
container_id(){ docker inspect -f '{{.Id}}' "$GLM_CONTAINER" 2>/dev/null || echo none; }
gpu_busy(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -v t="$GPU_FREE_MIB" '$1+0 > t {f=1} END{exit !f}'; }

sglang_pids(){
  { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
      | awk -F', ' 'tolower($2) ~ /sglang/ {print $1}'
    pgrep -f "$SGLANG_ROOT/sglang-official/.venv/bin/sglang" 2>/dev/null
    pgrep -f 'sglang serve' 2>/dev/null
    pgrep -f 'serve_best_kv_offload.sh' 2>/dev/null
  } 2>/dev/null | sort -u
}

stop_sglang(){
  log "stopping sglang..."
  pkill -f 'serve_best_kv_offload.sh' 2>/dev/null
  pkill -f 'sglang serve' 2>/dev/null
  pkill -f "$SGLANG_ROOT/sglang-official/.venv/bin/sglang" 2>/dev/null
  pkill -f 'scripts/serve.sh' 2>/dev/null
  local pids; pids="$(sglang_pids)"
  [[ -n "$pids" ]] && echo "$pids" | xargs -r kill 2>/dev/null
  sleep 3
  pids="$(sglang_pids)"
  [[ -n "$pids" ]] && { log "SIGKILL sglang leftovers: $(echo "$pids"|tr '\n' ' ')"; echo "$pids" | xargs -r kill -9 2>/dev/null; }
}

stop_vllm(){
  log "stopping vLLM container ($GLM_CONTAINER)..."
  docker stop "$GLM_CONTAINER" >/dev/null 2>&1
  docker rm -f "$GLM_CONTAINER" >/dev/null 2>&1
}

wait_gpu_free(){
  log "waiting for GPUs to free (<=${GPU_FREE_MIB}MiB/card)..."
  local w=0
  while gpu_busy; do
    (( w >= GPU_FREE_TIMEOUT )) && { log "WARN: GPUs still busy after ${GPU_FREE_TIMEOUT}s"; return 1; }
    sleep 5; w=$((w+5))
  done
  log "GPUs free."
}

start_vllm(){
  log "booting GLM (vLLM): $GLM_LAUNCHER"
  SERVED_MODEL_NAMES="$VLLM_SERVED_NAMES" setsid nohup bash "$GLM_LAUNCHER" \
    > "$LOG_DIR/vllm-launch.log" 2>&1 </dev/null &
}

start_sglang(){
  log "starting sglang (detached): $SGLANG_LAUNCHER"
  ( cd "$SGLANG_ROOT" && setsid nohup bash "$SGLANG_LAUNCHER" \
      > "$LOG_DIR/sglang-launch.log" 2>&1 </dev/null & echo $! > "$SGLANG_PIDFILE" )
}

# ---- Phase 0: stop BOTH servers (whichever is running), free GPUs ----
stop_vllm
stop_sglang
wait_gpu_free

# ---- Phase 1: try vLLM ----
pre_id="$(container_id)"
start_vllm
# wait for the NEW container to appear (different id than before, or from none)
w=0
while [[ "$(container_id)" == "$pre_id" ]]; do
  (( w >= 60 )) && { log "WARN: new container not seen within 60s"; break; }
  sleep 3; w=$((w+3))
done
log "container id $(container_id) state=$(container_state); waiting for vLLM health (timeout ${VLLM_BOOT_TIMEOUT}s)..."

w=0; vllm_up=0
while (( w < VLLM_BOOT_TIMEOUT )); do
  if health_ok; then vllm_up=1; break; fi
  st="$(container_state)"
  [[ "$st" == "exited" || "$st" == "dead" || "$st" == "restarting" || "$st" == "none" ]] && { log "vLLM state=$st -> dead"; break; }
  docker logs "$GLM_CONTAINER" 2>&1 | grep -qE "$FATAL_RE" && { log "fatal error in vLLM logs -> dead"; break; }
  sleep 5; w=$((w+5))
done

if (( vllm_up )); then
  log "vLLM is healthy on :$PORT"
  result VLLM_UP
  log "GLM logs: docker logs $GLM_CONTAINER ; KV-LOOKUP: docker logs $GLM_CONTAINER 2>&1 | grep KV-LOOKUP"
  exit 0
fi

# ---- Phase 2: vLLM died -> capture logs, clean up, start sglang ----
log "vLLM failed — capturing failure log"
docker logs --tail 80 "$GLM_CONTAINER" > "$LOG_DIR/vllm-failure.log" 2>&1 || true
stop_vllm
wait_gpu_free
start_sglang

log "waiting for sglang health (timeout ${SGLANG_BOOT_TIMEOUT}s)..."
w=0; sg_up=0
while (( w < SGLANG_BOOT_TIMEOUT )); do
  if health_ok; then sg_up=1; break; fi
  if ! pgrep -f 'sglang serve' >/dev/null 2>&1; then
    log "no sglang serve process -> sglang not running"
    break
  fi
  sleep 5; w=$((w+5))
done

if (( sg_up )); then
  result SGLANG_UP
  log "sglang is healthy on :$PORT"
else
  result SGLANG_STARTING_OR_DEAD
  log "sglang did not confirm health within ${SGLANG_BOOT_TIMEOUT}s — inspect the log"
fi
log "sglang log: $SGLANG_LOG"
log "vLLM failure log: $LOG_DIR/vllm-failure.log"
exit 0
