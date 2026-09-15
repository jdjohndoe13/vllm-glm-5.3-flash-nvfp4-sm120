#!/usr/bin/env bash
# ============================================================================
# KV-offloading validation test for GLM-5.3-Flash-NVFP4 (patched tree)
#
# Boots the OFFLOAD launcher (vllm-glm-5.3-flash-nvfp4.sh, port 1025), runs a
# needle battery, prints a verdict, then stops the container and frees the
# GPUs. NOTE: this takes over the production port while it runs — restart the
# launcher afterwards (it will not restart itself).
#
# RUN THIS AFTER STOPPING WHATEVER LLM CURRENTLY OCCUPIES THE GPUs.
#
# Battery:
#   T1  cold ~110k-token request   -> baseline prefill time
#   T2  identical repeat           -> GPU prefix-cache hit (fast)
#   T3  fresh ~124k-token request  -> fills GPU pool further
#   T4  fresh ~134k-token request  -> forces eviction of T1 blocks
#   T5  repeat of T1 AFTER EVICTION-> KEY TEST: restore from CPU tier.
#       fast + correct = offload works; slow + correct = tier not serving
#       (re-prefill path); wrong = corruption
#   T6  repeat of T3 (control, still in GPU pool)
#
# Logs: kvoff-server.log (full engine log), kvoff-report.txt (summary)
# ============================================================================
set -uo pipefail
F="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRVLOG="$F/kvoff-server.log"
REPORT="$F/kvoff-report.txt"
# docker container name under test (same default the launchers use)
CONTAINER_NAME=vllm-glm-5.3-flash-nvfp4
PORT=1025
# CPU tier budget (GiB) passed to the launcher (launcher default 512).
# 64 = fast boot + the "safe zone" for testing the upstream >64GB claim;
# raise via TIER_GB=128 bash test.sh for the >64GB leg of the comparison.
: "${TIER_GB:=64}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------- preflight ----------------
if curl -s --max-time 3 -o /dev/null http://localhost:$PORT/health 2>/dev/null; then
  log "WARNING: something is already answering on port $PORT"
fi
# Wait for GPUs to actually free: VRAM release lags container teardown, and a
# boot started too early dies in engine init (observed: 20:06 boot death).
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
  if [ -z "$USED" ] || [ "${USED:-99999}" -le 4096 ]; then break; fi
  if [ "$i" = 1 ]; then log "waiting for GPUs to free (max card ${USED}MiB)..."; fi
  sleep 5
done
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  log "ERROR: port $PORT already in use"; exit 1
fi

# ---------------- boot ----------------
# Launcher pre-flight silently stalls freeing orphaned tier mmaps (512-GiB
# tmpfs page reclaim takes tens of seconds, no output until done) — remove
# them here first so the launcher's own cleanup is instant.
sudo -n sh -c 'rm -f /dev/shm/vllm_offload_*.mmap' 2>/dev/null || rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
log "starting server (log: $SRVLOG)"
rm -f "$REPORT"
nohup env CPU_TIER_GB="$TIER_GB" SERVED_MODEL_NAMES="glm-5.3-flash qwen-3.8-flash-next" bash "$F/vllm-glm-5.3-flash-nvfp4.sh" > "$SRVLOG" 2>&1 &
log "waiting for 'Application startup complete' (timeout 1500 s)"
READY=0
for i in $(seq 1 150); do
  sleep 10
  if curl -s --max-time 3 -o /dev/null "http://localhost:$PORT/health" 2>/dev/null; then READY=1; break; fi
  CSTATE=$(docker ps -a --format '{{.Names}} {{.Status}}' 2>/dev/null | grep "^${CONTAINER_NAME} " | head -1)
  if [ -n "$CSTATE" ]; then
    SEEN=1
  elif [ "${SEEN:-0}" = 1 ]; then
    log "container vanished (docker ps -a) - aborting"; break
  elif [ "$i" -ge 30 ]; then
    log "container never appeared after 300s - aborting"; break
  fi
  case "$CSTATE" in
    Exited*|Dead*) log "container dead: $CSTATE - aborting"; break ;;
  esac
  if grep -aq "AssertionError" "$SRVLOG" 2>/dev/null; then
    log "AssertionError in log - aborting"; break
  fi
done

if [ "$READY" -ne 1 ]; then
  log "=== server did NOT become ready; relevant log lines: ==="
  grep -aE "AssertionError|Traceback|ERROR|offload|Offloading|kv_transfer" "$SRVLOG" | tail -30
  tail -20 "$SRVLOG"
  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    log "=== dead container logs (last 60): ==="
    docker logs --tail 60 "$CONTAINER_NAME" 2>&1 | tail -60
  fi
  docker stop -t 60 "$CONTAINER_NAME" >/dev/null 2>&1; docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
  echo "RESULT: TEST FAILED AT BOOT. GPUs freed." | tee "$REPORT"
  exit 1
fi
log "server ready"

# ---------------- payloads ----------------
python3 - <<'PY'
import json
def payload(size_chars, code):
    para = ("The industrial panorama of the northern valleys shifted gradually through the season. "
            "Records kept by the survey teams described weather, crop yields, and the movement of "
            "goods along the river in meticulous detail. Every settlement kept its own ledger, and "
            "the regional office collected copies each month for review and archival storage. ")
    txt = para * (size_chars // len(para) + 1)
    txt = txt[:size_chars]
    pos = int(size_chars * 0.4)
    txt = txt[:pos] + f" Special note for auditors: the access code is {code}. " + txt[pos:]
    return txt
for name, size, code in [("P0", 160_000, "KIWI-55-JACKFRUIT"),
                         ("P1", 640_000, "BANANA-42-XYLOPHONE"),
                         ("P2", 720_000, "MANGO-77-TANGERINE"),
                         ("P3", 780_000, "CHERRY-13-PAPAYA")]:
    body = {"model": "qwen-3.8-flash-next",
            "messages": [{"role": "user", "content": payload(size, code) +
              "\n\nQuestion: What is the access code mentioned in the text? Answer with only the code."}],
            "temperature": 0, "max_tokens": 200, "seed": 42}
    with open(f"/tmp/kvoff-{name}.json", "w") as f:
        json.dump(body, f)
print("payloads written")
PY

# ---------------- run tests ----------------
fire() { # $1=test name  $2=payload file
  curl -sS --max-time 300 -o "/tmp/kvoff-resp-$1.json" \
       -w "%{http_code} %{time_total}" \
       -H "Content-Type: application/json" \
       --data @"$2" \
       "http://localhost:$PORT/v1/chat/completions" > "/tmp/kvoff-http-$1.txt" 2>/dev/null
}

log "tier budget: ${TIER_GB} GiB (launcher CPU_TIER_GB override)"
log "T1: cold P1 ...";        fire T1 /tmp/kvoff-P1.json
log "T2: repeat P1 ...";      fire T2 /tmp/kvoff-P1.json
log "T0: short P0 (fits tier) ..."; fire T0 /tmp/kvoff-P0.json
log "T3: fresh P2 ...";       fire T3 /tmp/kvoff-P2.json
log "T4: fresh P3 (forces eviction) ..."; fire T4 /tmp/kvoff-P3.json
log "T5: repeat P1 after eviction (CPU-tier restore) ..."; fire T5 /tmp/kvoff-P1.json
log "T6: repeat P2 (control, in GPU pool) ..."; fire T6 /tmp/kvoff-P2.json
log "T7: repeat P0 after eviction (tier restore, fits tier) ..."; fire T7 /tmp/kvoff-P0.json

# ---------------- verdict ----------------
python3 - <<'PY' | tee "$REPORT"
import json

WANT = {"T1": "BANANA-42", "T2": "BANANA-42", "T0": "KIWI-55", "T3": "MANGO-77",
        "T4": "CHERRY-13", "T5": "BANANA-42", "T6": "MANGO-77", "T7": "KIWI-55"}
INFO = {"T1": "cold baseline", "T2": "GPU-pool hit", "T0": "short fresh (fits tier)",
        "T3": "fresh fill",
        "T4": "fresh fill / eviction trigger", "T5": "CPU-tier restore (KEY)",
        "T6": "GPU-pool hit (control)", "T7": "tier restore after evict (fits tier)"}

times, ok, ptoks, snippet, code = {}, {}, {}, {}, {}
for t, want in WANT.items():
    try:
        code[t], ttime = open(f"/tmp/kvoff-http-{t}.txt").read().split()
        times[t] = float(ttime)
        r = json.load(open(f"/tmp/kvoff-resp-{t}.json"))
        m = r["choices"][0]["message"]
        c = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
        ok[t] = want in c
        ptoks[t] = r.get("usage", {}).get("prompt_tokens")
        snippet[t] = c.replace("\n", " ").strip()[:60]
    except Exception as e:
        times[t], ok[t], ptoks[t], snippet[t], code[t] = None, False, None, f"ERR {e}", "?"

print("=== KV-offload test report ===")
print(f"{'test':4} {'purpose':34} {'http':4} {'time_s':>8} {'ptok':>7} {'correct':8} snippet")
for t in WANT:
    tt = f"{times[t]:.1f}" if times[t] is not None else "-"
    print(f"{t:4} {INFO[t]:34} {code[t]:4} {tt:>8} {str(ptoks[t]):>7} {str(ok[t]):8} {snippet[t]}")

print()
if times["T1"] and times["T5"]:
    speedup = times["T1"] / times["T5"] if times["T5"] else float("inf")
    print(f"T5 vs T1 speedup: {speedup:.1f}x")
    if ok["T5"] and speedup > 1.7:
        print("VERDICT: KV OFFLOADING WORKS (fast restore from CPU tier, correct answer)")
    elif ok["T5"]:
        print("VERDICT: boot OK, prefix caching OK, but T5 slow => CPU tier not serving restores (re-prefill path)")
    elif ok["T1"]:
        print("VERDICT: CORRUPTION RISK: T5 answered wrong => offloaded KV restore corrupts state")
    else:
        print("VERDICT: unexpected: T1 itself failed")
else:
    print("VERDICT: incomplete results")

if times.get("T0") and times.get("T7"):
    s7 = times["T0"] / times["T7"] if times["T7"] else float("inf")
    print(f"T7 vs T0 speedup: {s7:.1f}x (tier restore, payload fits tier)")
t2_fast = times["T2"] is not None and times["T1"] and times["T2"] < 0.6 * times["T1"]
t6_fast = times["T6"] is not None and times["T3"] and times["T6"] < 0.6 * times["T3"]
print(f"GPU prefix-cache hits (T2 fast={t2_fast}, T6 fast={t6_fast})")
PY

# ---------------- offload evidence from server log ----------------
echo "=== offload-related server log lines (tail) ===" | tee -a "$REPORT"
grep -aiE "offload|OffloadingConnector|kv_transfer|CPUOffload|SharedOffload|kv connector" "$SRVLOG" | tail -25 | tee -a "$REPORT"

# ---------------- teardown ----------------
log "stopping container (frees GPUs)"
docker stop -t 120 "$CONTAINER_NAME" >/dev/null 2>&1
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
log "container removed. GPUs are now free."
log "To resume normal serving: bash $F/vllm-glm-5.3-flash-nvfp4.sh"
echo "Full engine log: $SRVLOG ; report: $REPORT"
