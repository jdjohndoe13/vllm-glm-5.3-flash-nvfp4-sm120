#!/bin/bash
# soak_watch.sh — 4.5-hour unattended disease-watch on the LIVE server.
# Each ~3min iteration: M1 (marker session revisit — must tier-restore in
# ~2s; each load touches its keys MRU-fresh), X_i (fresh ~165k churner —
# evicts GPU + fills tier; pool fills at ~iter 14 and legit LRU eviction
# begins, counted via evicted_total), M1G (marker growth — GPU hit ~2s).
# ALERT condition: i>0 and M1 took >10s (tier restore failed) — checked
# against evicted_total and CPU-TIER-EVICT lines to separate legit eviction
# from the silent uncounted removal (the "disease").
# Log: /tmp/soak.log — safe to run alongside real traffic.
set -uo pipefail
C=vllm-glm-5.3-flash-nvfp4
BASE=http://localhost:1025
LOG=/tmp/soak.log
curl -s --max-time 3 -o /dev/null "$BASE/health" || { echo "server not up"; exit 1; }
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

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
M1  = build(640_000, 900_000)
M1G = M1 + "\n" + build(40_000, 910_000)
names = [("M1",M1),("M1G",M1G)]
for i in range(90):
    names.append((f"X{i}", build(640_000, 950_000 + i*1_000)))
for name, text in names:
    body = {"model": "qwen-3.8-flash-next",
            "messages": [{"role": "user", "content": text + "\n\nReply with just OK."}],
            "max_tokens": 8, "temperature": 0}
    with open(f"/tmp/soak-{name}.json", "w") as f:
        json.dump(body, f)
print("payloads written")
PYEOF

log "=== soak start: 90 iterations x ~3min ==="
for i in $(seq 0 89); do
  T0=$(date +%s)
  T_M1=$(curl -s --max-time 600 -o /dev/null -w '%{time_total}' -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/soak-M1.json)
  T_X=$(curl -s --max-time 600 -o /dev/null -w '%{time_total}' -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/soak-X$i.json)
  T_G=$(curl -s --max-time 600 -o /dev/null -w '%{time_total}' -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/soak-M1G.json)
  LK=$(docker logs --since 130s "$C" 2>&1 | grep -a 'KV-LOOKUP' | tail -1 | sed 's/.*(EngineCore[^)]*) //')
  EVT=$(curl -s -m 5 "$BASE/metrics" | grep -a 'kv_offload_cpu_evicted_total' | grep -av '^#' | grep -oa '[0-9.]*$')
  TAG=ok
  if [ "$i" -gt 0 ] && python3 -c "exit(0 if float('$T_M1') > 10 else 1)"; then TAG=ALERT; fi
  log "ITER $i $TAG M1=${T_M1}s X${i}=${T_X}s M1G=${T_G}s evt=${EVT} | ${LK}"
  NOW=$(date +%s); S=$(( 180 - (NOW - T0) )); [ "$S" -gt 5 ] && sleep "$S" || sleep 5
done
log "=== soak end ==="
