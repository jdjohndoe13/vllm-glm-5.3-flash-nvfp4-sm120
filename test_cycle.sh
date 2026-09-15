#!/bin/bash
# test_cycle.sh — NON-DESTRUCTIVE tier-archive-persistence repro on the LIVE
# server (:1025). Stores a marker session's KV, evicts it from GPU via a
# second session, applies tier pressure with 10 fresh sessions WITHOUT
# exhausting the pool (so legit LRU eviction must NOT fire), then re-tests
# the marker's tier restore. If the marker restore fails while evicted_total
# stayed 0 and no CPU-TIER-EVICT lines appeared, the silent uncounted
# key-removal path (the "disease") is caught live with frozen counters.
set -uo pipefail
C=vllm-glm-5.3-flash-nvfp4
PORT=1025
BASE=http://localhost:$PORT
curl -s --max-time 3 -o /dev/null "$BASE/health" || { echo "server not up"; exit 1; }
log(){ echo "[$(date +%H:%M:%S)] $*"; }

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
P1   = build(640_000, 100_000)
P2   = build(640_000, 200_000)
P1G1 = P1 + "\n" + build(40_000, 300_000)
P1G2 = P1G1 + "\n" + build(40_000, 310_000)
P2G  = P2 + "\n" + build(40_000, 320_000)
names = [("P1",P1),("P2",P2),("P1G1",P1G1),("P1G2",P1G2),("P2G",P2G)]
for i in range(10):
    names.append((f"C{i}", build(640_000, 400_000 + i*1_000)))
for name, text in names:
    body = {"model": "qwen-3.8-flash-next",
            "messages": [{"role": "user", "content": text + "\n\nReply with just OK."}],
            "max_tokens": 8, "temperature": 0}
    with open(f"/tmp/cyc-{name}.json", "w") as f:
        json.dump(body, f)
print("payloads written")
PYEOF

snap(){ curl -s -m 5 "$BASE/metrics" | grep -a -E 'kv_offload_cpu_(allocated|free_list_len|evictable_len|evicted_total|allocation_size_count|allocation_size_sum)|external_prefix_cache_hits_total' | grep -av '^#' | tr '\n' ' ' ; echo; }

fire(){ # $1 = payload name
  T0=$(date +%s.%N)
  CODE=$(curl -s --max-time 600 -o /tmp/cyc-last.json -w '%{http_code}' -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/cyc-$1.json)
  T1=$(date +%s.%N)
  log "$1 http=$CODE time=$(python3 -c "print(round($T1-$T0,1))")s"
  docker logs --since 90s "$C" 2>&1 | grep -a 'KV-LOOKUP' | tail -1 | sed 's/^.*(EngineCore[^)]*) //'
}

log "baseline:"; snap
fire P1
fire P2
log "sanity restore (expect ~2s tier restore):"; fire P1G1
log "pressure phase (10 fresh x ~165k tokens; pool must NOT exhaust):"
for i in 0 1 2 3 4 5 6 7 8 9; do fire C$i; done
log "disease test A - P1 archive after pressure (expect ~2s if healthy):"; fire P1G2
log "disease test B - P2 archive (newest, expect ~2s):"; fire P2G
log "final:"; snap
log "CPU-TIER-EVICT lines (expect none):"
docker logs --since 900s "$C" 2>&1 | grep -a 'CPU-TIER-EVICT' | tail -5
log "KV-LOOKUP tail:"
docker logs --since 900s "$C" 2>&1 | grep -a 'KV-LOOKUP' | tail -16
