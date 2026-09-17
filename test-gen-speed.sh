#!/usr/bin/env bash
# ============================================================================
# test-gen-speed.sh — single-stream RAW decode-speed measurement (2026-09-16)
#
# Purpose: measure the engine's REAL token generation rate with prompt
# processing subtracted, identically for the MTP and the NON-MTP launcher
# (same prompt, same request, same math — only the engine under :1025
# changes), so the two variants can be compared 1:1.
#
# Method (the "after prefill" part of the user's formula made exact):
#   1. probe request   — same prompt, max_tokens=1     -> wall_probe
#      (one HTTPS round-trip + prefill + 1 decode step)
#   2. long request    — same prompt, max_tokens=TOKENS -> wall_long
#      (stream=false: full decode runs server-side, wall = decode time)
#   decode_time ≈ wall_long - wall_probe   (the 1-token probe subtracts
#   prefill + overhead; the ±1 decode step inside the probe is noise at
#   tens-of-thousands of tokens)
#
#            generation speed = generated_tokens / decode_time
#   plus the projection: time for 100000 tokens = 100000 / speed.
#
# Self-labeling: probes /metrics — `spec_decode_*` metrics present = MTP,
# absent = NON-MTP. One TSV row is appended to
#   logs/gen-speed-results.tsv
# and a full detail file (raw responses, tick lines, spec metrics) is
# written to logs/gen-speed-<LABEL>-<stamp>.txt
#
# Usage:
#   bash test-gen-speed.sh                 # TOKENS=30000 default
#   TOKENS=100000 bash test-gen-speed.sh   # the user's 100k formulation
#
# Caveats:
#   * engine must be READY (health 200) — this script does not wait;
#   * run while NO other traffic is in flight: concurrent requests batch
#     together and change the single-stream decode rate (Running: >1);
#   * early EOS shortens the generation but the rate stays valid; the
#     script reports finish_reason so a too-short run is visible.
# ============================================================================
set -u
F="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-1025}"
TOKENS="${TOKENS:-30000}"
LOGDIR="$F/logs"
mkdir -p "$LOGDIR"
BASE="http://127.0.0.1:${PORT}"

# --- engine must be ready --------------------------------------------------
_CODE="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/models" 2>/dev/null || echo 000)"
if [ "${_CODE:-000}" != "200" ]; then
  echo "ERROR: $BASE/v1/models -> ${_CODE:-000}; engine not READY. Nothing measured." >&2
  exit 1
fi

MODEL="$(curl -s "$BASE/v1/models" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')"
[ -n "${MODEL:-}" ] || MODEL="qwen-3.8-flash-next"

# --- label: MTP vs NON-MTP (spec-decode metrics present or not) -------------
METRICS="$(curl -s "$BASE/metrics" 2>/dev/null || true)"
if printf '%s' "$METRICS" | grep -q 'spec_decode_num_accepted'; then
  LABEL="MTP"
else
  LABEL="NON-MTP"
fi
SPEC_LINES="$(printf '%s' "$METRICS" \
  | grep -E '^vllm:spec_decode_num_accepted' | head -n 4 || true)"

# --- engine config stamp for the TSV (KV bytes, chunk budget, spec tokens) --
# Sniffed from the newest boot log (bare-metal); docker boots fall back to
# container logs. Values land in the appended TSV row so results from
# different engine configs stay distinguishable in one file.
ENGL="$(ls -t "$LOGDIR"/engine-*.log 2>/dev/null | head -n 1)"
sniffof() { # sniffof <pattern> -> trailing integer of first match (or blank)
  local out=""
  if [ -n "$ENGL" ]; then
    out="$(grep -m1 -oE "$1" "$ENGL" 2>/dev/null || true)"
  elif command -v docker >/dev/null 2>&1 \
       && docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm-glm-5\.3-flash-nvfp4$'; then
    out="$(docker logs --tail 60 vllm-glm-5.3-flash-nvfp4 2>&1 | grep -m1 -oE "$1" || true)"
  fi
  printf '%s' "$out" | grep -oE '[0-9]+$' || true
}
KVB="$(sniffof "kv_cache_memory_bytes': [0-9]+")"
MBN="$(sniffof "max_num_batched_tokens': [0-9]+")"
SPEC="$(sniffof "num_speculative_tokens': [0-9]+")"
if [ "$LABEL" = "NON-MTP" ]; then SPEC="0"; fi

TS="$(date +%Y%m%d-%H%M%S)"
DETAIL="$LOGDIR/gen-speed-${LABEL}-${TS}.txt"

# --- prompt: engineered for unbroken long generation ------------------------
PROMPT_FILE="$LOGDIR/gen-speed-prompt.txt"
cat > "$PROMPT_FILE" <<'EOF'
Write an extremely detailed, continuous technical essay on the history, design
and engineering of single-board computers — CPU, memory, I/O, expansion buses,
storage, networking, displays, power — moving decade by decade with deep
specifics, part numbers, architectures and trade-offs. Do not conclude, do not
summarize, do not stop writing; keep writing continuously until you reach the
token limit.
EOF

payload() { # payload <max_tokens> <outfile>
  # seed pinned: vLLM picks per-request seeds for unseeded requests, so
  # temp-0.6 sampling drifts acceptance length (2.3-3.0 => 142-183 tok/s
  # spread seen 2026-09-17). A fixed seed makes gen-speed rows comparable.
  python3 - "$MODEL" "$1" "$PROMPT_FILE" > "$2" <<'PY'
import sys, json
model, max_tokens, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompt = open(path, encoding="utf-8").read()
json.dump({"model": model, "temperature": 0.6, "max_tokens": max_tokens,
           "seed": 42,
           "messages": [{"role": "user", "content": prompt}]}, sys.stdout)
PY
}

run() { # run <payloadfile> <outfile> ; echo wall time in ns
  local t0="$(date +%s%N)"
  curl -s --max-time 3600 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @"$1" -o "$2"
  echo $(( $(date +%s%N) - t0 ))
}

usage_of() { # usage_of <responsefile> ; sets COMPL, FINISH
  eval "$(python3 - "$1" <<'PY'
import sys, json
try:
    d = json.load(open(sys.argv[1]))
    u = d.get("usage", {}) or {}
    ch = (d.get("choices") or [{}])[0]
    print("COMPL=%d" % (u.get("completion_tokens") or 0))
    print("FINISH=%s" % (ch.get("finish_reason") or "N/A"))
except Exception as e:
    print("COMPL=0")
    print("FINISH=parse_error:%s" % type(e).__name__)
PY
)"
}

# --- 1+2: probe, then the long run -----------------------------------------
P1="$LOGDIR/.gen-speed-p1.json"; O1="$LOGDIR/.gen-speed-o1.json"
P2="$LOGDIR/.gen-speed-p2.json"; O2="$LOGDIR/.gen-speed-o2.json"

echo "[$LABEL] probing prefill (max_tokens=1) ..."
payload 1 "$P1"
WALL_PROBE_NS="$(run "$P1" "$O1")"
echo "[$LABEL] generating up to TOKENS=$TOKENS ..."
payload "$TOKENS" "$P2"
WALL_LONG_NS="$(run "$P2" "$O2")"
usage_of "$O2"

DECODE_NS=$(( WALL_LONG_NS - WALL_PROBE_NS ))
if [ "$DECODE_NS" -le 0 ]; then DECODE_NS="$WALL_LONG_NS"; fi
RATE="$(awk -v c="$COMPL" -v n="$DECODE_NS" 'BEGIN{printf "%.2f", (n>0? c*1e9/n : 0)}')"
DECODE_S="$(awk -v n="$DECODE_NS" 'BEGIN{printf "%.1f", n/1e9}')"
WALL_S="$(awk -v n="$WALL_LONG_NS" 'BEGIN{printf "%.1f", n/1e9}')"
PROJ100K="$(awk -v c="$COMPL" -v n="$DECODE_NS" 'BEGIN{printf "%.0f", 100000*n/(c*1e9)}')"

# --- engine's own steady-state ticks (best effort, engine-agnostic) ---------
WIN_S=$(( WALL_LONG_NS / 1000000000 + 180 ))
TICKS=""
if command -v docker >/dev/null 2>&1 \
   && docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm-glm-5\.3-flash-nvfp4$'; then
  TICKS="$(docker logs --since "${WIN_S}s" vllm-glm-5.3-flash-nvfp4 2>&1 \
            | grep -E 'Running: 1 reqs' | tail -n 3 || true)"
elif ls "$LOGDIR"/engine-*.log >/dev/null 2>&1; then
  NEWLOG="$(ls -t "$LOGDIR"/engine-*.log | head -n 1)"
  TICKS="$(grep -E 'Running: 1 reqs' "$NEWLOG" | tail -n 3 || true)"
fi

# --- persist + report -------------------------------------------------------
TSV="$LOGDIR/gen-speed-results.tsv"
if [ ! -f "$TSV" ]; then
  printf 'ts\tlabel\trequested\tgenerated\tfinish\twall_total_s\tdecode_only_s\tdecode_tok_per_s\t100k_tokens_proj_s\tmodel\tkv_bytes\tmbn\tspec\n' > "$TSV"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date +%Y-%m-%dT%H:%M:%S)" "$LABEL" "$TOKENS" "$COMPL" "$FINISH" \
  "$WALL_S" "$DECODE_S" "$RATE" "$PROJ100K" "$MODEL" \
  "${KVB:-}" "${MBN:-}" "${SPEC:-}" >> "$TSV"

{
  echo "===== test-gen-speed $TS ====="
  echo "port: $PORT | model: $MODEL | label: $LABEL"
  echo "requested max_tokens: $TOKENS | generated: $COMPL | finish: $FINISH"
  echo "wall_probe (prefill):  $(awk -v n=$WALL_PROBE_NS 'BEGIN{printf "%.2f", n/1e9}')s"
  echo "wall_total:            ${WALL_S}s"
  echo "decode only:           ${DECODE_S}s"
  echo "===> generation speed:  ${RATE} tokens/s"
  echo "===> projected 100000 tokens: ${PROJ100K}s"
  if [ -n "$SPEC_LINES" ]; then echo "--- spec-decode counters (per accepted/drafted):"; printf '%s\n' "$SPEC_LINES"; fi
  if [ -n "$TICKS" ]; then echo "--- engine steady-state ticks (Running: 1):"; printf '%s\n' "$TICKS"; fi
  echo "--- raw probe response (first 200 B):"; head -c 200 "$O1"; echo
  echo "--- raw long response (first 200 B):"; head -c 200 "$O2"; echo
} | tee "$DETAIL"

echo
echo "row appended -> $TSV"
