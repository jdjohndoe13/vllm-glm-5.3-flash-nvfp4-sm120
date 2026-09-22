#!/bin/bash
# canary v3.1 — native-engine watchdog (testcomp2, localhost:1025, no docker)
# One row per 300 s appended to $ROWLOG (relocated off /tmp; survives reboots);
# state in $STATE. v3 adds shm= (shared-memory broadcast stall), gpu= (GPU lock),
# NEWBOOTLOG rebase. v3.1 adds hb= (1-token completion heartbeat; 000 = 20 s
# timeout -> ALERT: SERVE-WEDGE) and the relocation off /tmp.
set -uo pipefail

LOGDIR=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/logs
CANARY_DIR=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/logs/canary
STATE="$CANARY_DIR/canary.state"
ROWLOG="$CANARY_DIR/canary5m.log"
HB_PAYLOAD="$CANARY_DIR/hb.json"
URL=http://localhost:1025

mkdir -p "$CANARY_DIR" 2>/dev/null
if [ ! -d "$CANARY_DIR" ] || [ ! -w "$CANARY_DIR" ]; then
  echo "canary: cannot create/write $CANARY_DIR - bailing (relay reports CANARY_DEAD)" >&2
  exit 1
fi

P_PAT='(api_server|EngineCore|Worker_TP)'   # p= discriminator; census-deployed N=9
SHM_PAT='No available shared memory broadcast block'
OOM_PAT='OOM on device|out of memory'
FB_PAT='KV-LOOKUP stale-hit fallback'
KV_PAT='KV-LOOKUP'
REP_PAT='repeats='
APC_PAT='APC-HIT'
GPU_PAT='RC watchdog|GPU is probably locked'      # journalctl -k substitute (dmesg restricted)
EV1_PAT='vllm:kv_offload_cpu_evictable_len'       # closest existing to *_evictable_total (no _total form exists)
EV2_PAT='vllm:kv_offload_cpu_evicted_total'
EV3_PAT='vllm:kv_offload_eviction_tombstones'
RMET='vllm:num_requests_running'                  # rmax fallback when 'Running:' marker absent

cnt() { # cnt <file> <ERE> -> cumulative matching-line count (0 on error/absence)
  local c
  c=$(grep -acE "$2" "$1" 2>/dev/null) || true
  echo "${c:-0}"
}
cnti() { # case-insensitive variant (500/oom)
  local c
  c=$(grep -aciE "$2" "$1" 2>/dev/null) || true
  echo "${c:-0}"
}
mval() { # mval <anchored-metric-name> <metrics-text> -> last sample value ('' if absent)
  local v
  v=$(printf '%s\n' "$2" | grep -E "^$1( |\{)" | tail -1 | cut -d' ' -f2)
  echo "${v:-}"
}

while true; do
  TAG=""

  # ---- resolve bootlog: latest.log symlink target if valid, else newest engine-*.log ----
  L=$(readlink -f "$LOGDIR/latest.log" 2>/dev/null)
  [ -z "$L" ] && L=""
  if [ -z "$L" ] || [ ! -f "$L" ]; then
    L=""
    BASE=$(ls -1 "$LOGDIR" 2>/dev/null | sed -n 's/^\(engine-.*\.log\)$/\1/p' | sort | tail -1)
    [ -n "$BASE" ] && L="$LOGDIR/$BASE"
  fi
  if [ -n "$L" ] && [ -f "$L" ]; then
    SZ=$(stat -c%s "$L" 2>/dev/null)
    [ -z "$SZ" ] && SZ=0
  else
    L=""; SZ=0
  fi

  # ---- load state; missing keys initialize from CURRENT values (no false startup alerts) ----
  ST_SHM=""; ST_GPU=""; ST_PH=""; ST_BOOT=""; ST_OFF=""
  if [ -f "$STATE" ]; then
    ST_SHM=$(sed -n 's/^shm_last=//p' "$STATE" | tail -1)
    ST_GPU=$(sed -n 's/^gpu_last=//p' "$STATE" | tail -1)
    ST_PH=$(sed -n 's/^prev_h=//p' "$STATE" | tail -1)
    ST_BOOT=$(sed -n 's/^bootlog=//p' "$STATE" | tail -1)
    ST_OFF=$(sed -n 's/^offset=//p' "$STATE" | tail -1)
  fi

  if [ -n "$L" ]; then
    SHM_C=$(cnt "$L" "$SHM_PAT")
  else
    SHM_C=0
  fi
  GPU_C=$(journalctl -k --no-pager 2>/dev/null | grep -acE "$GPU_PAT")
  [ -z "$GPU_C" ] && GPU_C=0

  [ -z "$ST_SHM" ] && ST_SHM=$SHM_C
  [ -z "$ST_GPU" ] && ST_GPU=$GPU_C
  [ -z "$ST_PH" ] && ST_PH=1
  [ -z "$ST_OFF" ] && ST_OFF=$SZ
  [ -z "$ST_BOOT" ] && ST_BOOT=$L

  # ---- NEWBOOTLOG: resolved log differs from PREVIOUS-row stored path -> rebase ----
  if [ -n "$L" ] && [ "$L" != "$ST_BOOT" ]; then
    TAG=" ALERT: NEWBOOTLOG"
    ST_SHM=$SHM_C
    ST_OFF=$SZ
    ST_BOOT="$L"
  fi
  # in-place truncation guard (size shrank below last offset)
  if [ -n "$L" ] && [ "$SZ" -lt "$ST_OFF" ]; then
    ST_SHM=$SHM_C
    ST_OFF=$SZ
  fi

  # ---- field collection ----
  H=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "$URL/health" 2>/dev/null)
  [ -z "$H" ] && H=000
  P=$(pgrep -c -f "$P_PAT" 2>/dev/null)
  [ -z "$P" ] && P=0
  if [ -n "$L" ]; then
    C500=$(cnti "$L" '500 Internal Server Error')
    COOM=$(cnti "$L" "$OOM_PAT")
    FBV=$(cnt "$L" "$FB_PAT")
    KVV=$(cnt "$L" "$KV_PAT")
    REPV=$(cnt "$L" "$REP_PAT")
    APCV=$(cnt "$L" "$APC_PAT")
    rmv=$(grep -oa 'Running: [0-9]*' "$L" 2>/dev/null | tail -1 | cut -d: -f2)
    rmv=${rmv// /}
  else
    C500=0; COOM=0; FBV=0; KVV=0; REPV=0; APCV=0; rmv=""
  fi
  MET=$(curl -s -m 5 "$URL/metrics" 2>/dev/null)
  EV1=$(mval "$EV1_PAT" "$MET"); [ -z "$EV1" ] && EV1="?"
  EV2=$(mval "$EV2_PAT" "$MET"); [ -z "$EV2" ] && EV2="?"
  EV3=$(mval "$EV3_PAT" "$MET"); [ -z "$EV3" ] && EV3="?"
  if [ -z "$rmv" ] && [ -n "$MET" ]; then
    rmv=$(mval "$RMET" "$MET")
    case "$rmv" in *.0) rmv=${rmv%.0};; esac
  fi
  [ -z "$rmv" ] && rmv="?"

  # ---- deltas ----
  DSHM=$(( SHM_C - ST_SHM )); [ "$DSHM" -lt 0 ] && DSHM=0
  DGPU=$(( GPU_C - ST_GPU )); [ "$DGPU" -lt 0 ] && DGPU=0
  BYTES=$(( SZ - ST_OFF ));   [ "$BYTES" -lt 0 ] && BYTES=0

  # ---- alerts (SHM-STALL only when the previous emitted row had h=200; first row prev_h=1) ----
  if [ "$DSHM" -ge 1 ] && [ "$ST_PH" = 200 ]; then
    TAG="$TAG ALERT: SHM-STALL($DSHM)"
  fi
  if [ "$DGPU" -ge 1 ]; then
    TAG="$TAG ALERT: GPU-LOCK($DGPU)"
  fi

  # ---- heartbeat (v3.1): 1-token completion; 000 = 20 s timeout -> SERVE-WEDGE ----
  HBC=$(curl -s --max-time 20 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d @"$HB_PAYLOAD" "$URL/v1/chat/completions" 2>/dev/null); HBR=$?
  if [ "$HBR" = 28 ]; then
    HBC=000
    TAG="$TAG ALERT: SERVE-WEDGE"
  elif [ -z "$HBC" ]; then
    HBC=000
  fi

  # ---- emit row (hb= sits between gpu= and fb=) ----
  printf '[%s] h=%s p=%s 500=%s oom=%s shm=%s gpu=%s hb=%s fb=%s kv=%s rmax=%s rep=%s apc=%s bytes=+%s ev=%s,%s,%s%s\n' \
    "$(date +%H:%M:%S)" "$H" "$P" "$C500" "$COOM" "$DSHM" "$DGPU" "$HBC" \
    "$FBV" "$KVV" "$rmv" "$REPV" "$APCV" "$BYTES" "$EV1" "$EV2" "$EV3" "$TAG" >> "$ROWLOG"

  # ---- persist state for next row ----
  printf '%s\n' "shm_last=$SHM_C" "gpu_last=$GPU_C" "prev_h=$H" "bootlog=$L" "offset=$SZ" > "$STATE"

  sleep 300
done
