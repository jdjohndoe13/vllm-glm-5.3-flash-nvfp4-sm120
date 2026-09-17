#!/usr/bin/env bash
# ============================================================================
# test-prefill-sweep.sh — max_num_batched_tokens (MBT) impact measurement
# (2026-09-17)
#
# Measures what the chunked-prefill scheduler budget does to this engine:
# one run per MBT configuration (2048 vs 1024), engine rebooted by the user
# in between per the standard runbook. Rows are appended to
#   logs/prefill-sweep.tsv
# with the engine config self-stamped from the newest boot log, so both runs
# land in one file and stay distinguishable.
#
# Legs (all with temperature 0.6 prompts; unique salted fillers per run so
# prefix caching cannot serve a repeat from cache):
#   warmup         8k prompt, max_tokens=4    (warms kernels after a reboot;
#                                             row recorded as warmup)
#   seq-<PT>       <PT>-token prompt, max_tokens=32 for every
#                  PT in SIZES (default 12000 32000 64000 128000)
#                  -> prefill-only cost of big prompts at this MBT
#   mixed          (~64k prompt + 30k decode) AND (~96k prompt + 32 decode)
#                  fired the way the 2026-09-17 07:26 OOM happened: long
#                  decode running, a second big prefill chunks in
#                  -> reproduces the _fp8_fp4_mqa_logits_impl OOM pattern
#                  and shows the decode-rate penalty during a big prefill
#
# Usage:
#   bash test-prefill-sweep.sh
#   SIZES="12000 32000" bash test-prefill-sweep.sh      # shorter sweep
#   DECODE_TOKENS=100000 bash test-prefill-sweep.sh     # longer mixed-A decode
#
# Caveats:
#   * engine must be READY (health 200); the script does not wait for it;
#   * run while NO other traffic is in flight (OpenCode/open-webui generate
#     bursts every ~10-30 s and pollute the prefill timing);
#   * the OOM snapshot is printed from the boot log when the engine dies;
#   * one run = one TSV append batch (7 rows: 1 warmup + |SIZES| + 1 mixed).
# ============================================================================
set -u
F="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-1025}"
SIZES="${SIZES:-12000 32000 64000 128000}"
DECODE_TOKENS="${DECODE_TOKENS:-30000}"
MIXED_B_PROMPT="${MIXED_B_PROMPT:-96000}"
MIXED_B_GEN="${MIXED_B_GEN:-32}"
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
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
[ -n "${MODEL:-}" ] || MODEL="qwen-3.8-flash-next"

# --- engine config stamp (mirrors test-gen-speed.sh) ------------------------
ENGL="$(ls -t "$LOGDIR"/engine-*.log 2>/dev/null | head -n 1)"
sniffof() {
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

TS="$(date +%Y%m%d-%H%M%S)"
DETAIL="$LOGDIR/prefill-sweep-${TS}.txt"
TSV="$LOGDIR/prefill-sweep.tsv"
if [ ! -f "$TSV" ]; then
  printf 'ts\tmbn\tkv_bytes\tspec\tkind\tgen_max_tokens\tprompt_tokens\twall_s\ttok_per_s\talive\n' > "$TSV"
fi
row() { # row <kind> <gen_max_tokens> <prompt_tokens> <wall_s> <tok_per_s> <alive>
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" "${MBN:-}" "${KVB:-}" "${SPEC:-}" \
    "$1" "$2" "$3" "$4" "$5" "$6" >> "$TSV"
}
SALT="sweep-$$-$(date +%s)"

payload_ns() { # payload_ns <prompt_tokens> <salt> <max_tokens> <outfile>
  python3 - "$MODEL" "$1" "$2" "$3" "$4" <<'PY'
import sys, json
model, ptokens, salt, mt, path = (sys.argv[1], int(sys.argv[2]),
                                  sys.argv[3], int(sys.argv[4]), sys.argv[5])
# unique filler: salted word-salad so no two runs share a prefix
block = (f"unit {salt} reference manual appendix figure table example section "
         f"commentary annotation draft revision variant configuration "
         f"parameter describing {salt} datapoint measurement calibration "
         f"declaration observation annotation repetition")
n = ptokens // 32
filler = " ".join(block for _ in range(n + 1))[: ptokens * 5]
prompt = (filler + "\n\nRepeat the word OK " + str(max(8, mt)) +
          " times, then stop.")
json.dump({"model": model, "temperature": 0.6, "max_tokens": mt,
           "messages": [{"role": "user", "content": prompt}]}, open(path, "w"))
PY
}

run() { # run <payloadfile> <outfile> ; echoes wall ns
  local t0
  t0="$(date +%s%N)"
  curl -s --max-time 3600 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @"$1" -o "$2"
  echo $(( $(date +%s%N) - t0 ))
}

token_count() { # token_count <response-payload-request-json> -> prompt token count
  curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/models"
}
alive() { [ "$(token_count)" = "200" ]; }

usage_of() { # usage_of <responsefile> ; sets COMPL, FINISH, PT
  eval "$(python3 - "$1" <<'PY'
import sys, json
try:
    d = json.load(open(sys.argv[1]))
    u = d.get("usage", {}) or {}
    ch = (d.get("choices") or [{}])[0]
    print("COMPL=%d" % (u.get("completion_tokens") or 0))
    print("FINISH=%s" % (ch.get("finish_reason") or "N/A"))
    print("PTOK=%d" % (u.get("prompt_tokens") or 0))
except Exception as e:
    print("COMPL=0")
    print("FINISH=parse_error:%s" % type(e).__name__)
    print("PTOK=0")
PY
)"
}

P_REQ="$LOGDIR/.sweep-req.$$.json"; P_RESP="$LOGDIR/.sweep-resp.$$.json"
# mixed-leg dedicated paths (no reuse of the seq-loop files)
P_MA="$LOGDIR/.pf-mA.$$.json";  P_MA_R="$LOGDIR/.pf-mA-resp.$$.json"
P_MB="$LOGDIR/.pf-mB.$$.json";  P_MB_R="$LOGDIR/.pf-mB-resp.$$.json"

{
echo "===== test-prefill-sweep $TS ====="
echo "engine stamps: mbn=${MBN:-?} kv_bytes=${KVB:-?} spec=${SPEC:-?} model=$MODEL"
echo "sizes: $SIZES | decode_tokens(A): $DECODE_TOKENS | B: ${MIXED_B_PROMPT}p+${MIXED_B_GEN}g"
} > "$DETAIL"
cat "$DETAIL"

# --- warmup (fixture: post-reboot kernel warm; row kept so rate is comparable)
echo "[warmup] 8192-token prefill, max_tokens=4 ..."
payload_ns 8192 "$SALT-w" 4 "$P_REQ"
WW="$(awk -v n="$(run "$P_REQ" "$P_RESP")" 'BEGIN{printf "%.2f", n/1e9}')"
row "warmup" "4" "8192" "$WW" "n/a" "$(alive && echo yes || echo no)"
echo "[warmup] wall ${WW}s  alive=$(alive && echo yes || echo no)"

# --- sequential prefill sweep ------------------------------------------------
for PTX in $SIZES; do
  echo "[seq-$PTX] prefilling ${PTX}-token prompt, max_tokens=32 ..."
  payload_ns "$PTX" "$SALT-$PTX" 32 "$P_REQ"
  WALL_NS="$(run "$P_REQ" "$P_RESP")"
  usage_of "$P_RESP"
  # rate on the MEASURED prompt tokens (filler lands within ~2% of the ask)
  RATE="$(awk -v p="${PTOK:-0}" -v n="$WALL_NS" 'BEGIN{printf "%.1f", (n>0? p*1e9/n : 0)}')"
  WALL_S="$(awk -v n="$WALL_NS" 'BEGIN{printf "%.2f", n/1e9}')"
  _A="no"; alive && _A="yes"
  row "seq-$PTX" "32" "$PTX" "$WALL_S" "$RATE" "$_A"
  echo "[seq-$PTX] wall ${WALL_S}s  rate ${RATE} tok/s  alive=$_A  finish=${FINISH:-?}  (measured prompt tokens: ${PTOK:-0})"
  if [ "$_A" != "yes" ]; then
    echo "ENGINE DEAD mid-sweep after ${PTX}-token leg; aborting." | tee -a "$DETAIL"
    break
  fi
done

# --- mixed leg: 64k+30k-decode AND 96k+32 fired mid-decode (07:26 OOM replay)
MIXED_STATUS="skipped"
if alive; then
  echo "[mixed] firing A: ~60000-token filler + max_tokens=$DECODE_TOKENS (bg) ..."
  payload_ns 60000 "$SALT-mA" "$DECODE_TOKENS" "$P_MA"
  t0A="$(date +%s%N)"
  ( curl -s --max-time 3600 -X POST "$BASE/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      --data-binary @"$P_MA" -o "$P_MA_R" ) &
  APID=$!
  echo "[mixed] waiting for A to enter decode (Running:1 + generation>0) ..."
  # Gate = decode IN PROGRESS: a fresh tick line (only from content appended
  # after A fired — byte offset excludes stale traffic) with Running:1 and a
  # nonzero generation throughput. Engine ticks print every ~10 s, so 2-s
  # sampling with a 120-s window is generous.
  ENTERED=no; GATE_TICK=""
  LOGOFF="$(wc -c < "$ENGL" 2>/dev/null || echo 0)"
  for _ in $(seq 1 60); do
    sleep 2
    GATE_TICK="$(tail -c "+$((LOGOFF + 1))" "$ENGL" 2>/dev/null \
                 | grep 'Running: 1 reqs' | tail -n 1 \
                 | grep -E 'Avg generation throughput: [1-9]' || true)"
    [ -n "$GATE_TICK" ] && { ENTERED=yes; break; }
    kill -0 "$APID" 2>/dev/null || break
  done
  echo "[mixed] A entered decode: $ENTERED"
  if [ -n "$GATE_TICK" ]; then
    echo "[mixed] gate tick: $(printf '%s' "$GATE_TICK" | sed 's/^.*loggers/loggers/')"
  fi
  B_STATUS="not-fired"
  if [ "$ENTERED" = "yes" ] && alive; then
    echo "[mixed] firing B: ~${MIXED_B_PROMPT}-token filler + max_tokens=$MIXED_B_GEN ..."
    payload_ns "$MIXED_B_PROMPT" "$SALT-mB" "$MIXED_B_GEN" "$P_MB"
    ( curl -s --max-time 3600 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary @"$P_MB" -o "$P_MB_R" ) &
    BPID=$!
    wait "$APID" 2>/dev/null; wait "$BPID" 2>/dev/null
    B_STATUS="fired"
  else
    wait "$APID" 2>/dev/null
  fi
  ELAPSED_S="$(awk -v n=$(( $(date +%s%N) - t0A )) 'BEGIN{printf "%.1f", n/1e9}')"
  usage_of "$P_MA_R"
  A_COMPL="${COMPL:-0}"; A_FIN="${FINISH:-?}"
  usage_of "$P_MB_R"
  B_COMPL="${COMPL:-0}"; B_FIN="${FINISH:-?}"
  _A="no"; alive && _A="yes"
  echo "[mixed] A: generated=$A_COMPL finish=$A_FIN | B(fired=$B_STATUS): generated=$B_COMPL finish=$B_FIN | wall $ELAPSED_S s | alive=$_A"
  row "mixed-A64k+B96k" "${DECODE_TOKENS}+${MIXED_B_GEN}" "A~60k+B~${MIXED_B_PROMPT}" "$ELAPSED_S" "n/a" "$_A"
  if [ "$_A" != "yes" ]; then
    MIXED_STATUS="engine-died"
    echo "--- OOM snapshot (boot log tail, filtered) ---"
    if [ -n "$ENGL" ]; then
      grep -E 'CUDA out of memory|OutOfMemoryError|_mqa_logits|ERROR' "$ENGL" 2>/dev/null | tail -n 14
    fi
  else
    MIXED_STATUS="survived"
  fi
fi

cat >> "$DETAIL" << EOF
--- summary ---
mixed leg: $MIXED_STATUS
rows appended: $(wc -l < "$TSV") lines total in $TSV
EOF

echo
echo "sweep complete -> rows appended to $TSV (mixed: $MIXED_STATUS)"
