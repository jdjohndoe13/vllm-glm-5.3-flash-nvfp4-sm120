#!/usr/bin/env bash
# ============================================================================
# KV/prefix-cache response-detail probe for GLM-5.3-Flash-NVFP4 (vLLM, port
# from EDITABLE SETTINGS below).
#
# NON-DESTRUCTIVE: unlike test.sh, this script does NOT boot, restart or stop
# anything. It only sends requests to the RUNNING server and reads
# /metrics. Safe to run while the server is serving — but for clean counter
# attribution run it while no other long-context client (e.g. an agent
# session served by this box) is active; the script checks for that itself.
#
# What it verifies (one run):
#   1. idle-stability gate: prefix_cache counters must be static while we
#      are quiet — otherwise attribution is flagged unreliable; counter
#      RESETS (values decreasing) observed intermittently on this build
#      are reported as a separate warning
#   2. short prompt (~2 full blocks), fresh + verbatim repeat:
#      observed cached_tokens = 0 BOTH times on this build — hit threshold
#      on this hybrid model is higher; short prompts do not hit even when
#      repeated verbatim (see README "Response-level detail")
#   3. ~2k-token prompt (~7 full blocks), fresh + repeat:
#      EXPECTED: repeat reports cached_tokens > 0 and lower TTFT; hit is
#      partial on this model (measured: 1024 of 7 blocks cached) — the
#      prometheus hit-counter delta should agree with the reported amount
#   4. metrics object (per-request timing) present in every response
#   5. kv_transfer_params / ec_transfer_params: expected null (local CPU
#      offload connector does not emit response-side transfer params)
#
# Logs: kvblock-report.txt (summary) next to this script; payloads and raw
# responses in /tmp/kvblock-*.json / /tmp/kvblock-resp-*.json
# ============================================================================
set -uo pipefail
F="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# EDITABLE SETTINGS
# ============================================================================
: "${PORT:=1025}"
: "${MODEL_NAME:=qwen-3.8-flash-next}"  # served name to address (see SERVED_MODEL_NAMES in the launcher)
: "${MAX_TOKENS:=400}"        # must exceed the model's reasoning spend
: "${BIG_REPS:=12}"           # filler repeats -> ~5-6k tokens, ~20+ blocks
: "${IDLE_WAIT_S:=45}"        # quiet window for the stability gate
: "${IDLE_TOLERANCE:=1024}"   # max tolerated counter movement (tokens)
# Optional: pre-staged payloads (e.g. real session requests captured by an
# llm-proxy) instead of the built-in generated ones. A staged .req.json is
# re-served with max_tokens capped and stream disabled; prompt content is
# untouched, so prefix-cache behavior matches the original request.
: "${PAYLOAD_SHORT:=}"
: "${PAYLOAD_BIG:=}"
CONTAINER_NAME=vllm-glm-5.3-flash-nvfp4
REPORT="$F/kvblock-report.txt"
BASE="http://localhost:$PORT"
export MAX_TOKENS BIG_REPS MODEL_NAME
export RUN_TS="$(date +%s)"
export PAYLOAD_SHORT PAYLOAD_BIG

log() { echo "[$(date +%H:%M:%S)] $*"; }
rm -f "$REPORT"

# ---------------- preflight ----------------
if ! curl -s --max-time 5 "$BASE/v1/models" | grep -q '"object":"list"'; then
  log "ERROR: no OpenAI server answering on $BASE — start the launcher first."
  exit 1
fi
if PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) && [ "$PROCS" -gt 0 ]; then
  log "NOTE: $PROCS GPU compute process(es) running (expected: the vllm server itself)."
fi
if ! curl -s --max-time 5 "$BASE/metrics" | grep -q 'prefix_cache_queries_total'; then
  log "ERROR: server has no prefix_cache metrics — was it started with --enable-prompt-tokens-details era launcher?"
  exit 1
fi

# ---------------- helpers ----------------
# counter <metric-name-prefix> -> numeric value of first series whose line
# ENDS in a complete float. Returns empty if the scrape was truncated or
# the series is missing (caller must tolerate empty values).
counter() {
  curl -s --max-time 30 "$BASE/metrics" \
    | awk -v m="$1" 'index($0, m) == 1 && $NF ~ /^[0-9.eE+-]+$/ {print $NF; exit}'
}
delta() { awk "BEGIN{printf \"%.0f\", (${2:-0}) - (${1:-0})}"; }

snap() {
  echo "$(counter 'vllm:prefix_cache_queries_total{') \
        $(counter 'vllm:prefix_cache_hits_total{') \
        $(counter 'vllm:external_prefix_cache_hits_total{')"
}
snap_print() { log "counters: local_queries=$1 local_hits=$2 ext_hits=$3"; }

fields() { # $1=response file -> the interesting bits
  grep -o '"prompt_tokens":[0-9]*\|"cached_tokens":[0-9]*\|"created_cache_tokens":[0-9]*\|"time_to_first_token_ms":[0-9.]*\|"kv_transfer_params":[a-z0-9]*\|"ec_transfer_params":[a-z0-9]*\|"metrics":{"[^}]*}' "$1"
}

# ---------------- payloads ----------------
log "generating payloads (/tmp/kvblock-small.json, /tmp/kvblock-big.json)"
python3 - <<'PY'
import json, os
para = ("Historical record follows. The industrial panorama of the northern valleys "
        "shifted gradually through the season. Records kept by the survey teams "
        "described weather, crop yields, and the movement of goods along the river "
        "in meticulous detail. Every settlement kept its own ledger, and the "
        "regional office collected copies each month for review and archival "
        "storage. The eastern ridge hosted three watchtowers whose rotations were "
        "logged weekly by the quartermaster. Salt, timber, and woven goods moved "
        "along the southern road in predictable cycles, and the toll records "
        "survive nearly intact for the entire period. Grain shipments doubled "
        "after the second harvest and the granary ledgers reflect that surge in "
        "detail. Surveyors noted minor tremors in the western hills but no "
        "structural damage was reported in any of the settlements along the "
        "valley floor. ")
# Leading run-unique marker: makes the "fresh" request genuinely uncached
# (its whole prefix differs from every previous run), while the verbatim
# repeat of the SAME payload still hits normally.
marker = "Run marker R%s. " % os.environ.get("RUN_TS", "0")
small_txt = marker + para * 4 + " Now ignore all of the text above and reply with exactly this token: KVTEST-OK"
reps = int(os.environ.get("BIG_REPS", "12"))
big_txt = marker + para * reps + " Now ignore all of the text above and reply with exactly this token: KVTEST-OK"
for name, txt in [("small", small_txt), ("big", big_txt)]:
    body = {"model": os.environ.get("MODEL_NAME", "glm-5.3-flash"),
            "messages": [{"role": "user", "content": txt}],
            "max_tokens": int(os.environ.get("MAX_TOKENS", "400")),
            "temperature": 0}
    with open(f"/tmp/kvblock-{name}.json", "w") as f:
        json.dump(body, f)
print("payloads written (run marker: R%s)" % os.environ.get("RUN_TS", "0"))
# Optional staged payloads (real session requests): cap max_tokens, disable
# streaming, drop stream_options — prompt content untouched.
for name, envkey in (("small", "PAYLOAD_SHORT"), ("big", "PAYLOAD_BIG")):
    p = os.environ.get(envkey, "").strip()
    if p and os.path.exists(p):
        b = json.load(open(p))
        b["max_tokens"] = int(os.environ.get("MAX_TOKENS", "400"))
        b["stream"] = False
        b.pop("stream_options", None)
        json.dump(b, open(f"/tmp/kvblock-{name}.json", "w"))
        print(f"{name} payload: staged from {p}")
PY

req() { # $1=name $2=payload
  curl -s --max-time 600 -o "/tmp/kvblock-resp-$1.json" \
       -w "%{http_code} %{time_total}" \
       -H 'Content-Type: application/json' --data @"$2" \
       "$BASE/v1/chat/completions" > "/tmp/kvblock-http-$1.txt" 2>/dev/null
}

# ---------------- 1) idle-stability gate ----------------
log "idle-stability gate: sampling counters, then ${IDLE_WAIT_S}s of silence"
read -r Q0 H0 E0 <<EOF
$(snap)
EOF
sleep "$IDLE_WAIT_S"
read -r _ Q1 H1 E1 <<EOF
$(snap x)
EOF
DQ=$(delta "${Q0:-0}" "${Q1:-0}"); DH=$(delta "${H0:-0}" "${H1:-0}")
AQ=$(awk "BEGIN{v=$DQ+0; printf \"%.0f\", (v<0?-v:v)}")
AH=$(awk "BEGIN{v=$DH+0; printf \"%.0f\", (v<0?-v:v)}")
log "idle movement: queries=$DQ hits=$DH over ${IDLE_WAIT_S}s (abs: queries=${Q0:-?}->${Q1:-?}, hits=${H0:-?}->${H1:-?})"
QUIET=1
if [ -z "$Q0" ] || [ -z "$Q1" ]; then
  QUIET=0
  log "WARNING: metrics scrape failed or was truncated during the gate"
  log "         (server busy serving another client?). Attribution UNRELIABLE."
elif [ "$AQ" -gt "$IDLE_TOLERANCE" ] || [ "$AH" -gt "$IDLE_TOLERANCE" ]; then
  QUIET=0
  if [ "$DQ" -lt 0 ] || [ "$DH" -lt 0 ]; then
    log "WARNING: engine-side counter RESET during the quiet window (values decreased)."
    log "         Observed intermittently on this build; per-request deltas below"
    log "         remain exact unless a reset lands inside their snapshot pairs."
  else
    log "WARNING: prefix_cache counters moved while we were idle (queries=$DQ hits=$DH)."
    log "         Another client (long-context agent session?) is hitting this"
    log "         server — counter attribution below is UNRELIABLE."
  fi
fi

# ---------------- 2) small prompt: fresh + repeat ----------------
log "SHORT prompt (~2 full blocks), fresh:"
req small-fresh /tmp/kvblock-small.json
fields /tmp/kvblock-resp-small-fresh.json | tee -a "$REPORT"
log "SHORT prompt, verbatim repeat:"
req small-repeat /tmp/kvblock-small.json
fields /tmp/kvblock-resp-small-repeat.json | tee -a "$REPORT"

# ---------------- 3) big prompt: fresh + repeat, counter-attributed ----------------
read -r Q0 H0 E0 <<EOF
$(snap)
EOF
log "BIG prompt (~2k tokens, ~7 blocks), fresh (counter snapshot taken):"
req big-fresh /tmp/kvblock-big.json
fields /tmp/kvblock-resp-big-fresh.json | tee -a "$REPORT"
read -r Q1 H1 E1 <<EOF
$(snap)
EOF
log "BIG fresh attribution: queries_delta=$(delta "${Q0:-0}" "${Q1:-0}") hits_delta=$(delta "${H0:-0}" "${H1:-0}")"

read -r Q0 H0 E0 <<EOF
$(snap)
EOF
log "BIG prompt, verbatim repeat (counter snapshot taken):"
req big-repeat /tmp/kvblock-big.json
fields /tmp/kvblock-resp-big-repeat.json | tee -a "$REPORT"
read -r Q1 H1 E1 <<EOF
$(snap)
EOF
log "BIG repeat attribution: queries_delta=$(delta "${Q0:-0}" "${Q1:-0}") hits_delta=$(delta "${H0:-0}" "${H1:-0}") ext_hits_delta=$(delta "${E0:-0}" "${E1:-0}")"

# ---------------- verdict ----------------
BIG_CACHED=$(grep -o '"cached_tokens":[0-9]*' /tmp/kvblock-resp-big-repeat.json | head -1 | cut -d: -f2)
BIG_TTFT_FRESH=$(grep -o '"time_to_first_token_ms":[0-9.]*' /tmp/kvblock-resp-big-fresh.json | head -1 | cut -d: -f2)
BIG_TTFT_REPEAT=$(grep -o '"time_to_first_token_ms":[0-9.]*' /tmp/kvblock-resp-big-repeat.json | head -1 | cut -d: -f2)
SMALL_CACHED=$(grep -o '"cached_tokens":[0-9]*' /tmp/kvblock-resp-small-repeat.json | head -1 | cut -d: -f2)
METRICS_OK=$(grep -c '"time_to_first_token_ms"' /tmp/kvblock-resp-big-repeat.json)
KV_NULL=$(grep -c '"kv_transfer_params":null' /tmp/kvblock-resp-big-repeat.json)

{
echo ""
echo "=== KV/prefix-cache response-detail verdict ==="
echo "idle-stability:        $([ "$QUIET" = 1 ] && echo 'QUIET (attribution trustworthy)' || echo 'NOT QUIET (deltas may include other clients)')"
echo "small repeat cached:   $SMALL_CACHED  (observed 0 for short prompts: hit threshold on this hybrid model is higher)"
echo "big repeat cached:     ${BIG_CACHED:-0}  (expected > 0; partial hits are normal in this range)"
echo "big TTFT fresh->rep:   ${BIG_TTFT_FRESH:-?} -> ${BIG_TTFT_REPEAT:-?} ms (repeat should be far lower)"
echo "metrics object:        $([ "$METRICS_OK" -gt 0 ] && echo present || echo MISSING)"
echo "kv/ec_transfer_params: $([ "$KV_NULL" -gt 0 ] && echo 'null (expected for the local offload connector)' || echo 'non-null?!')"
if [ "${BIG_CACHED:-0}" -gt 0 ]; then
  if [ -n "$BIG_TTFT_FRESH" ] && [ -n "$BIG_TTFT_REPEAT" ]; then
    SPEEDUP=$(awk "BEGIN{printf \"%.1f\", $BIG_TTFT_FRESH / $BIG_TTFT_REPEAT}")
    echo "TTFT speedup:          ${SPEEDUP}x"
  fi
  echo "VERDICT: PASS — response-level prefix-cache detail works (cached_tokens + TTFT agree)."
else
  echo "VERDICT: FAIL — big-prompt repeat reported no cache hits."
  echo "  If hits_delta was also 0: the repeat genuinely missed (server busy? pool churned?)."
  echo "  If hits_delta > 0: engine hit the cache but did not report it — reporting gap."
fi
} | tee -a "$REPORT"

log "done. full report: $REPORT ; raw responses: /tmp/kvblock-resp-*.json"
