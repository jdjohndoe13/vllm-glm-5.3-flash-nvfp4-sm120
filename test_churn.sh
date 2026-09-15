#!/bin/bash
# test_churn.sh — NON-DESTRUCTIVE interleaved-churn repro against the LIVE
# server (:1025). Simulates OpenCode-style multi-session traffic with
# growing prefixes and prints the KV-LOOKUP trace after every request, to
# catch the tier-archive vanish (the "disease") in the act.
# Usage: bash test_churn.sh          (server must already be up)
set -uo pipefail
F="$(cd "$(dirname "$0")" && pwd)"
C=vllm-glm-5.3-flash-nvfp4
PORT=1025
BASE=http://localhost:$PORT

if ! curl -s --max-time 3 -o /dev/null "$BASE/health"; then
  echo "server not up on :$PORT - aborting (non-destructive test)"; exit 1
fi

log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "baseline gauges:"
curl -s -m 5 "$BASE/metrics" | grep -a 'kv_offload_cpu_' | grep -av '^#'
curl -s -m 5 "$BASE/metrics" | grep -a 'prefix_cache_' | grep -av '^#' | grep -aE 'queries_total|hits_total'

python3 - <<'PYEOF'
import json, random
def para(seed):
    random.seed(seed)
    words = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel",
             "india","juliet","kilo","lima","mike","november","oscar","papa"]
    return " ".join(random.choice(words) for _ in range(48))
def build(n_chars, seed0=0):
    text = ""; i = seed0
    while len(text) < n_chars:
        text += para(i) + "\n"; i += 1
    return text[:n_chars]
P1 = build(640_000)
P2 = build(720_000, 10_000)
P3 = build(780_000, 20_000)
P1G1 = P1 + "\n" + build(40_000, 30_000)   # session1 turn2: P1 prefix + growth
P1G2 = P1G1 + "\n" + build(40_000, 40_000) # session1 turn3: P1G1 prefix + growth
for name, text in [("P1",P1),("P2",P2),("P3",P3),("P1G1",P1G1),("P1G2",P1G2)]:
    body = {"model": "qwen-3.8-flash-next",
            "messages": [{"role": "user", "content": text + "\n\nReply with just OK."}],
            "max_tokens": 8, "temperature": 0}
    with open(f"/tmp/churn-{name}.json", "w") as f:
        json.dump(body, f)
print("payloads written")
PYEOF

for NAME in P1 P2 P1G1 P3 P1G2; do
  T0=$(date +%s.%N)
  CODE=$(curl -s --max-time 600 -o /tmp/churn-last.json -w '%{http_code}' -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/churn-$NAME.json)
  T1=$(date +%s.%N)
  DT=$(python3 -c "print(round($T1-$T0,1))")
  log "$NAME http=$CODE time=${DT}s"
  docker logs --since 60s "$C" 2>&1 | grep -a KV-LOOKUP | tail -1
done

log "final gauges:"
curl -s -m 5 "$BASE/metrics" | grep -a 'kv_offload_cpu_' | grep -av '^#'
curl -s -m 5 "$BASE/metrics" | grep -a 'prefix_cache_' | grep -av '^#' | grep -aE 'queries_total|hits_total'
log "KV-LOOKUP tail:"
docker logs --since 400s "$C" 2>&1 | grep -a KV-LOOKUP | tail -10
