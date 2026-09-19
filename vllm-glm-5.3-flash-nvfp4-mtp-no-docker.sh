#!/usr/bin/env bash
# ============================================================================
# GLM-5.3-Flash-NVFP4 on vLLM — 8x RTX 5090 (sm_120) — MTP VARIANT, NO DOCKER
# with KV CPU offloading enabled (CPU_TIER_GB, default 512 GiB) and NEXTN MTP
# speculative decoding at SPEC_TOKENS depth (default 3).
#
# Byte-identical to the no-docker launcher
# vllm-glm-5.3-flash-nvfp4-no-docker (2026-09-16 build with the
# eviction-tombstone capacity knob), banner + SPEC_TOKENS aside, except:
#   + --speculative-config '{"method":"mtp","num_speculative_tokens":N}'
#   + --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4]}'
# Same port + pidfile/logs as the no-docker launcher (single-tenant kit):
# unlike the docker launchers, a bare-metal start REFUSES while anything
# (docker container or the no-docker engine) holds port 1025 — it never
# stops a running engine by itself. All of the no-docker launcher's notes
# below apply to THIS file, so only the MTP deltas are described here.
#
# This is the BARE-METAL (no-docker) parity launcher, extracted from the
# running container's filesystem. Purpose (2026-09-15): A/B-test whether the
# ~576+ GiB CPU-tier pinning failure ("cudaHostRegister code=2" /
# "NVRM: failed to allocate page table") is container-related, by running the
# EXACT same engine config directly on the host (Ubuntu 24.04, driver
# 590.48.01, 8x RTX 5090) without the container boundary.
#
# Runtime tree (extracted 2026-09-15 from the pinned container
# sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5,
# image cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1, fork commit
# g487ecf187 / PR vllm-project/vllm#54743 — same provenance as the docker
# launcher header):
#   /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin/
#     dist-packages/   <- container /usr/local/lib/python3.12/dist-packages
#                         (vllm 0.1.dev20051+g487ecf187, torch 2.13.0+cu130,
#                         full cu13 pip-wheel stack: cublas 13.1.1.3,
#                         cudnn 9.20.0.48, nccl 2.30.7 wheel / torch-bundled
#                         2.29.7 runtime, nvshmem, cutlass-dsl, ... 16 GB)
#     bin/vllm         <- container /usr/local/bin/vllm console script,
#                         shebang rewritten to the venv python
#     venv/            <- thin venv wrapper (host python 3.12.3 == container
#                         python 3.12.3); venv site-packages is a SYMLINK to
#                         dist-packages so .pth processing matches the
#                         container's site-dir behavior exactly
#     EXTRACTION_INFO.txt
# The 14 files from patched-files/manifest.txt were overlaid into the
# extracted vllm package (byte-identical to the per-file bind mounts the
# docker launcher performs; MD5-verified at extraction AND re-verified below
# on every start).
#
# Parity with the docker launcher (vllm-glm-5.3-flash-nvfp4.sh):
#   * ALL env vars + server args preserved exactly (CPU_TIER_GB default 512,
#     KV_CACHE_MEMORY, port 1025 (API+metrics), VLLM_* envs, the kv-offload
#     envs, tier remount + leak-cleanup pre-flight, JIT cache dirs)
#   * dropped docker-only bits:
#       --ipc=host        -> implicit (engine uses the HOST's /dev/shm, same
#                            tmpfs the tier mmap lives on)
#       --shm-size        -> irrelevant for the same reason
#       --gpus all        -> implicit (host driver present: 590.48.01,
#                            libcuda.so.1 + libnvidia-ml.so.1 in ldconfig)
#       -p/$PORT:$PORT    -> engine binds 0.0.0.0:$PORT directly
#       pinned-image guard-> replaced by a runtime-tree guard +
#                            patched-file parity check below
#       docker stop/rm    -> plain pidfile + process-group management (stop)
#       --restart=unless-stopped -> NOT reproduced (no supervisor); start the
#                            engine manually, stop with: bash <this>.sh stop
#   * NCCL: the docker launcher sets NO NCCL_* envs — the engine uses
#     torch's bundled NCCL 2.29.7 (verified in-container and in the
#     extracted tree). Nothing to mirror; do not add overrides (parity).
#   * --cap-add SYS_NICE: container-only capability; bare metal as non-root
#     the engine may log a warning when lowering nice/affinity but vLLM
#     tolerates it. Run this script as root if you need that bit of parity.
#
# STOP THE DOCKER ENGINE FIRST: this launcher refuses to start while the
# docker container still holds port 1025 (same-port guard below), and the
# /dev/shm pre-flight would otherwise wipe the RUNNING container's tier
# mmap — the loud failure is deliberate protection, not an error.
#
# Adaptive fallbacks (identical to the docker launcher):
#   * kernel JIT cache dir: /mnt/data/shared/models/vllm-moet-cache when
#     usable, else .cache/ next to this script (NOT the HF model cache)
#   * model: local RedHatAI checkpoint when present+readable, else the HF
#     repo id is passed and vLLM downloads it
#
# Usage:
#   bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh            (start; blocks until
#         the readiness gate passes, then returns — engine keeps running)
#   bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh stop       (TERM the process
#         group, escalate to KILL, then run the same /dev/shm wipe as docker
#         rm would have needed)
#   bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh status    (state summary, then
#         live-follows the newest engine log while the engine runs — Ctrl+C
#         stops the follow, NOT the engine; when the engine is down the
#         summary alone is printed)
#   MAX_MODEL_LEN=150000 bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh
#   bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh KV_CACHE_MEMORY=4000000000
#   CPU_TIER_HUGETLB=1 CPU_TIER_GB=800 bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh
#         (opt-in hugetlbfs/2 MiB-page tier backing: the pre-flight reserves
#         host hugepages via tune-host-for-hugepages.sh and the engine env
#         gets VLLM_KV_OFFLOAD_TIER_HUGETLB=1, so the tier file is created in
#         $VLLM_KV_OFFLOAD_HUGETLB_DIR (default /dev/hugepages) instead of
#         /dev/shm. Real 2 MiB pages shrink the NVIDIA driver's per-rank
#         pinned page-table budget ~64x, lifting the >=576 GiB
#         "NVRM: failed to allocate page table" ceiling. Hugepages are
#         auto-reserved at start (tune-host-for-hugepages.sh; hugefree to
#         release); any
#         hugetlb failure falls back to the stock /dev/shm tier loudly.)
# ============================================================================
set -euo pipefail
F="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# Pre-flight / post-stop shared-memory leak cleanup (VERBATIM from the docker
# launcher; broadened 2026-09-15): reclaims ANY engine-owned orphan in the
# HOST's /dev/shm — the tier mmaps (vllm_offload_*.mmap) AND any other
# vllm*/VLLM* leftover, e.g. leaked VLLM_OBJECT_STORAGE_SHM_BUFFER_* ring
# buffers (vllm auto-names these UPPERCASE per process tree — vllm/envs.py
# get_env_or_set_default — which the old vllm_offload_*.mmap-only pattern
# never matched, forcing a manual rm) — plus torch psm_/sem.mp leftovers.
# Every removed file is printed with its size and a removed-count/bytes
# summary closes the run, so misses are never silent. Used before start
# (like the docker launcher's pre-flight), after a non-graceful stop, and
# on the readiness-gate death path. An optional "tier_only" scope (arg 1 of
# do_shm_cleanup) removes only vllm*/VLLM* files — the docker launcher's
# stop subcommand uses it so a live sibling engine's torch files survive.
# ----------------------------------------------------------------------------
SHM_CLEAN_CODE='
import os
import stat
freed = {"tier_mmap": 0, "vllm_other": 0, "torch_psm": 0, "torch_sem": 0}
removed = 0
removed_bytes = 0
denied = 0
denied_files = 0
_scope = os.environ.get("SHM_CLEAN_SCOPE", "all")
clean_dirs = ["/dev/shm"]
_hugetlb_dir = os.environ.get("VLLM_KV_OFFLOAD_HUGETLB_DIR", "/dev/hugepages")
if _hugetlb_dir not in clean_dirs and os.path.isdir(_hugetlb_dir):
    clean_dirs.append(_hugetlb_dir)

def _fmt(b):
    if b >= 1073741824:
        return f"{b / 1073741824:.2f} GiB"
    return f"{b / 1048576:.1f} MiB"

for clean_dir in clean_dirs:
    try:
        names = os.listdir(clean_dir)
    except OSError:
        continue
    for name in names:
        if not (name.startswith("vllm") or name.startswith("VLLM")
                or name.startswith("psm_") or name.startswith("sem.mp-")):
            continue
        path = os.path.join(clean_dir, name)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode):
            continue  # never touch directories, symlinks, sockets, fifos
        size = st.st_size
        cls = ("tier_mmap" if name.startswith("vllm_offload_") and name.endswith(".mmap")
               else "vllm_other" if name.startswith("vllm") or name.startswith("VLLM")
               else "torch_psm" if name.startswith("psm_")
               else "torch_sem" if name.startswith("sem.mp-")
               else None)
        if cls is None or (_scope == "tier_only" and cls not in ("tier_mmap", "vllm_other")):
            continue
        try:
            os.unlink(path)
        except PermissionError:
            denied += size
            denied_files += 1
            continue
        except OSError as exc:
            print(f"  shm cleanup: could not remove {path} ({exc})")
            continue
        freed[cls] += size
        removed += 1
        removed_bytes += size
        print(f"  shm cleanup: removed {path} ({_fmt(size)})")
if removed:
    parts = ", ".join(f"{k}={_fmt(v)}" for k, v in freed.items())
    print(f"Pre-flight shm cleanup: removed {removed} file(s), {_fmt(removed_bytes)} total ({parts})")
else:
    print("Pre-flight shm cleanup: nothing to remove (0 files)")
if denied:
    print(f"Pre-flight shm cleanup: {denied_files} file(s), {_fmt(denied)} skipped: permission denied — run: sudo rm -f /dev/shm/vllm_* /dev/shm/VLLM_* /dev/shm/psm_* /dev/shm/sem.mp-*")
'
do_shm_cleanup() {
  # $1 optional scope: "tier_only" removes only vllm*/VLLM* engine-owned
  # files (the docker launcher's stop subcommand uses it so a live sibling
  # engine's torch psm_/sem.mp files are never touched); default: full wipe.
  local _scope="${1:-all}"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "WARNING: python3 not found — skipped /dev/shm orphan cleanup." >&2
    return 0
  fi
  if [ "$(id -u)" = "0" ]; then
    SHM_CLEAN_SCOPE="$_scope" python3 -c "$SHM_CLEAN_CODE" || true
  elif sudo -n true 2>/dev/null && sudo -n env SHM_CLEAN_SCOPE="$_scope" python3 -c "$SHM_CLEAN_CODE" 2>/dev/null; then
    : # full wipe done via passwordless sudo
  else
    SHM_CLEAN_SCOPE="$_scope" python3 -c "$SHM_CLEAN_CODE" || true
  fi
}

# ============================================================================
# EDITABLE SETTINGS — view/edit right here, or override from the command line:
#   MAX_MODEL_LEN=150000 bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh (env prefix)
#   bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh MAX_MODEL_LEN=150000 (as argument)
#   MAX_MODEL_LEN=150000 ./vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh   (direct exec)
#   KV_CACHE_MEMORY=4000000000 bash vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh
# NOTE: 'bash MAX_MODEL_LEN=150000 <script>.sh' does NOT work — bash treats
#       the assignment as the script's filename.
# NOTE: the 'stop'/'status' subcommands do NOT honor VAR=value arguments.
# ============================================================================

# Process-management paths (defaults set before the subcommand dispatch; the
# VAR=value override loop below still applies to the start path).
: "${PORT:=1025}"                        # host port: API + /metrics
: "${PIDFILE:=$F/vllm-no-docker.pid}"    # engine launch pid == process-group id
: "${LOGDIR:=$F/logs}"                   # engine logs, next to this script

# ----------------------------------------------------------------------------
# stop/status subcommands — plain process management replacing
# `docker container stop/rm`. Stop SIGTERMs the whole engine process group
# (setsid made PGID == the recorded pid), escalates to SIGKILL, then runs the
# same /dev/shm tier-mmap wipe as the docker launcher's pre-flight — guarded
# so it never wipes a tier owned by ANOTHER engine still on $PORT.
# ----------------------------------------------------------------------------
cmd_stop() {
  if [ ! -f "$PIDFILE" ]; then
    echo "no pidfile ($PIDFILE) — no tracked no-docker engine to stop."
    echo "If an engine is actually running: pgrep -af 'vllm-bin/bin/vllm serve'"
    exit 0
  fi
  PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -z "${PID:-}" ] || ! kill -0 "$PID" 2>/dev/null; then
    echo "stale pidfile $PIDFILE (pid ${PID:-?} not alive) — removing."
    rm -f "$PIDFILE"
    exit 0
  fi
  if ! grep -aq vllm "/proc/${PID}/cmdline" 2>/dev/null; then
    echo "pidfile pid $PID is not the vllm engine (cmdline mismatch) — NOT killing; removing stale pidfile." >&2
    rm -f "$PIDFILE"
    exit 1
  fi
  echo "[launcher] stopping engine process group $PID (SIGTERM; SIGKILL after 150 s)..."
  kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  _gone=0
  for _i in $(seq 1 150); do
    if ! kill -0 -- "-$PID" 2>/dev/null; then _gone=1; break; fi
    if [ $((_i % 15)) -eq 0 ]; then
      echo "[launcher] ... '${_i}s elapsed, process group ${PID} still alive — rechecking every 1s, will escalate to SIGKILL at $_i s'"
    fi
    sleep 1
  done
  if [ "$_gone" != 1 ]; then
    echo "[launcher] group $PID still alive after 150 s — sending SIGKILL"
    kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$PIDFILE"
  echo "[launcher] engine process group $PID stopped. (Graceful exit unlinks the"
  echo "           tier mmap; the wipe below catches hard-kill leftovers — same"
  echo "           as the docker launcher's pre-flight.)"
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}\$"; then
    echo "[launcher] port $PORT still in use — skipping /dev/shm wipe (another engine may hold a tier mmap there)."
  else
    do_shm_cleanup
  fi
}

cmd_status() {
  _RUNNING=0
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null \
       && grep -aq vllm "/proc/${PID}/cmdline" 2>/dev/null; then
      _RUNNING=1
      echo "RUNNING: pid/pgid $PID (pidfile $PIDFILE)"
      echo "cmdline: $(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | cut -c1-160)"
    else
      echo "NOT RUNNING (stale pidfile $PIDFILE)"
    fi
  else
    echo "NOT RUNNING (no pidfile $PIDFILE)"
  fi
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}\$"; then
    echo "PORT $PORT: in use"
    ss -ltnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" || true
  else
    echo "PORT $PORT: free"
  fi
  if [ -f "$LOGDIR/latest.log" ]; then
    echo "log: $LOGDIR/latest.log (last 3 lines:)"
    tail -n 3 "$LOGDIR/latest.log" 2>/dev/null || true
  fi
  # Live-follow the newest engine log (newest by mtime) while the engine is
  # up: Ctrl+C exits the follow, the engine itself keeps running. A down
  # engine keeps the plain summary above — no follow of a dead log.
  if [ "$_RUNNING" = 1 ]; then
    _FOLLOW_LOG=$(ls -t "$LOGDIR"/engine-*.log 2>/dev/null | head -n 1 || true)
    if [ -n "${_FOLLOW_LOG:-}" ]; then
      echo "(following engine log — Ctrl+C to stop following; the engine keeps running)"
      tail -n 40 -F "$_FOLLOW_LOG" || true
    else
      echo "(no engine-*.log found in $LOGDIR — nothing to follow)"
    fi
  fi
}

case "${1-}" in
  stop)   cmd_stop;  exit 0 ;;
  status) cmd_status; exit 0 ;;
esac

# VAR=value command-line overrides (bash <script>.sh VAR=value ...):
for arg in "$@"; do
  case "$arg" in
    [A-Za-z_]*=*) export "${arg%%=*}=${arg#*=}" ;;
    *) echo "WARNING: ignoring non 'VAR=value' argument: $arg" >&2 ;;
  esac
done

# MTP depth sanity (MTP variant): positive integer, 1..6.
case "${SPEC_TOKENS:-3}" in
  ''|*[!0-9]*)
    echo "ERROR: SPEC_TOKENS must be a positive integer (got '${SPEC_TOKENS-}')." >&2
    exit 1 ;;
esac
if [ "${SPEC_TOKENS:-3}" -lt 1 ] || [ "${SPEC_TOKENS:-3}" -gt 6 ]; then
  echo "ERROR: SPEC_TOKENS=${SPEC_TOKENS-} is outside [1..6] — 3 is the default matching the" >&2
  echo "       sglang qwen-3.8-flash-next profile; 1 = the no-gain 260910 mtp-1 profile." >&2
  exit 1
fi

# Extracted runtime tree (replaces the docker IMAGE/IMAGE_ID pair):
: "${ENGINE_HOME:=/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin}"

# Model: local checkpoint dir (read-only for the engine) when present+readable,
# else HF_FALLBACK_ID is passed and vLLM downloads the checkpoint (~198 GB)
# into the HuggingFace cache. NOTE: use the RedHatAI checkpoint — the
# LibertAIDAI one emits corrupted tokens on sm_120 (vllm-project/vllm#54150).
: "${HF_LOCAL:=/mnt/huggingface/RedHatAI/GLM-5.3-Flash-NVFP4}"
: "${HF_FALLBACK_ID:=RedHatAI/GLM-5.3-Flash-NVFP4}"

# Shared kernel JIT cache (Triton/deep_gemm/tilelang — NOT the HuggingFace
# model cache). Falls back to <script-dir>/.cache when not usable.
: "${JIT_CACHE:=/mnt/data/shared/models/vllm-moet-cache}"

# Server sizing / behavior:
: "${MAX_MODEL_LEN:=262144}"
: "${MAX_NUM_SEQS:=4}"
# Chunked-prefill scheduler budget per engine step (tokens). Halving this to
# 1024 shrinks the fp8/fp4 mqa-logits transient workspace (~312 MiB/card at
# 2048) that OOM-killed a 2.75e9-KV bare-metal boot on 2026-09-17 07:26, at
# no cost to decode rate (decode batch width is MAX_NUM_SEQS; prefill
# ingestion pays per-step overhead instead — measure with test-prefill-sweep.sh).
: "${MAX_NUM_BATCHED_TOKENS:=1024}"
# GPU KV cache budget in BYTES per engine (fp8 KV). Default 3.3e9 ->
# 414,634 tokens of pool (validated). 4000000000 -> ~502k tokens;
# 5000000000 -> ~628k tokens (proven to boot WITH the tier in a
# 196k-token, 2-concurrent-request test). If a boot with a larger pool
# + tier fails, drop back to the default.
: "${KV_CACHE_MEMORY:=2600000000}"

# ---------------------------------------------------------------------------
# MTP speculative decoding (variant-specific knob, SUBJECT of this profile):
# SPEC_TOKENS = num_speculative_tokens — how many draft tokens the engine
# draws from the checkpoint's single MTP head every decode step, looping it
# EAGLE-style (single-module path in this fork: vllm/config/speculative.py
# use_multi_module_mtp() = min(num_nextn_predict_layers, N) — 1 layer here,
# config.json:1116, so any N stays single-module). Same mechanism as the
# sglang qwen-3.8-flash-next profile's --speculative-num-steps 3.
# Depth 1 caps a step at 2 tokens — measured as no gain (the
# 260910-01-200k-mtp-1.sh profile). Default 3 = the sglang-matching number.
# Confirm MTP engaged via the boot log's SpeculativeConfig print and
# per-spec stats in engine-*.log once decode-time acceptance starts.
# VRAM note (2026-09-16): with MTP depth 3 on these 32 GB cards,
# KV_CACHE_MEMORY=4000000000 CUDA-OOMs at boot (per-request draft KV slots
# + draft CUDA graphs eat the headroom); the no-docker 3.3e9 default
# (~414k tokens) boots clean under MTP-3 — keep the pool at 3.3e9 or below.
# NOTE: MTP x OffloadingConnector (the KV CPU tier): validated 2026-09-16
# in the docker MTP boot (MTP 3 + 32 GiB /dev/shm tier, zero errors, mean
# acceptance length 2.56/4.0); a hugetlbfs-tier MTP boot remains untested.
# ---------------------------------------------------------------------------
: "${SPEC_TOKENS:=3}"

# Names the model is advertised under in the OpenAI-compatible API,
# space-separated (expanded unquoted in the engine command on purpose, so
# that several names word-split into separate --served-model-name tokens).
# Default: "glm-5.3-flash qwen-3.8-flash-next" — BOTH names are served by
# default (2026-09-15; previously only "qwen-3.8-flash-next").
: "${SERVED_MODEL_NAMES:=glm-5.3-flash qwen-3.8-flash-next}"

# Host port for the OpenAI-compatible API (and /metrics on the same port) is
# $PORT (set above, default 1025): the docker launcher binds the SAME port, so
# only ONE of the two launchers can run at a time — the same-port guard below
# fails loudly otherwise. (test.sh assumes the default port.)
# Engine pidfile + logs: $PIDFILE / $LOGDIR (set above, next to this script).

# KV offloading CPU tier budget in GiB — a pinned, fully-preallocated mmap
# in the HOST's /dev/shm. The launcher remounts /dev/shm larger automatically
# when this exceeds the default 50%-of-RAM shm limit (tmpfs size is a cap,
# not a reservation).
# Default 512 GiB — the validated ceiling on this host/driver (2026-09-13:
# 0 cudaHostRegister failures, mHC warmup green). 576 GiB and above fail
# on ALL ranks with the NVIDIA driver's "NVRM: failed to allocate page
# table" — a per-rank pinned-region page-table budget, independent of
# free RAM or compaction. 256 GiB and 128 GiB validated 2026-09-12.
# NOTE: the tier is charged to RAM at boot and PINNED via cudaHostRegister
# (unpageable) — keep ~60+ GiB physical RAM for OS + engine processes;
# lower this (or stop other big software) if the box runs other big jobs.
# THIS LAUNCHER EXISTS TO A/B-TEST EXACTLY THIS FAILURE (576+ GiB
# cudaHostRegister code=2) WITH VS WITHOUT THE CONTAINER.
: "${CPU_TIER_GB:=512}"

# Hugetlbfs backing for the tier. Default 1 (2026-09-18): the pre-flight
# below auto-reserves the hugepages at start; CPU_TIER_HUGETLB=0 forces
# the stock /dev/shm tier.
# CPU_TIER_HUGETLB=1 moves the tier file to a hugetlbfs mount (2 MiB pages):
# the NVIDIA driver's per-rank pinned page-table budget (~537-600 MB/rank
# at 4 KiB pages — the >= 576 GiB "NVRM: failed to allocate page table"
# ceiling) shrinks ~64x, so 800 GiB can pin. The pre-flight below reserves
# the hugepages (tune-host-for-hugepages.sh) and the engine env gets
# VLLM_KV_OFFLOAD_TIER_HUGETLB=1; any hugetlb failure at boot falls back to
# the stock /dev/shm tier with a loud warning (where 576+ GiB still fails).
: "${CPU_TIER_HUGETLB:=1}"   # 2026-09-18: auto-reserve default (see hugefree)
# hugetlbfs mount for the tier (used by tune-host-for-hugepages.sh and the
# engine patch); exported so the leak-cleanup wipe covers it too.
export VLLM_KV_OFFLOAD_HUGETLB_DIR="${VLLM_KV_OFFLOAD_HUGETLB_DIR:-/dev/hugepages}"

# Eviction-tombstone registry capacity (ENTRIES) for the CPU-tier KV
# offload connector (patched v1/kv_offload/cpu/manager.py). Pure
# evicted-then-recomputed ATTRIBUTION: bounded FIFO of block hashes; when
# full the oldest entries silently age out (overflows_total metric climbs)
# and NOTHING in the cache path gates on it. Engine default 262,144; kit
# default 8,000,000 (pinned 2026-09-16 per user; ~1 GB engine RSS, ~30x
# the 2026-09-16 soak churn of ~1.1k tombstones/iter on the 800 GiB tier).
# 0 disables the registry (attribution metrics go dark). NOTE: vllm's
# envs.py unknown-var watchdog prints a harmless WARNING for this env —
# it is a patch-only variable read via os.environ.
: "${EVICTION_TOMBSTONES:=8000000}"
export VLLM_KV_OFFLOAD_EVICTION_TOMBSTONES="${EVICTION_TOMBSTONES}"

# ============================================================================
# Everything below is derived logic — usually no need to touch.
# ============================================================================

# Engine paths inside the extracted runtime tree.
ENGINE_PY="$ENGINE_HOME/venv/bin/python"
ENGINE_CMD="$ENGINE_HOME/bin/vllm"
DIST="$ENGINE_HOME/dist-packages"
VLLM_PKG="$DIST/vllm"

# Runtime-tree guard (replaces the docker launcher's pinned-image guard):
# the extracted tree must exist and contain the engine + its python.
if [ ! -x "$ENGINE_PY" ] || [ ! -x "$ENGINE_CMD" ] || [ ! -f "$VLLM_PKG/__init__.py" ]; then
  echo "ERROR: extracted runtime tree incomplete at $ENGINE_HOME." >&2
  echo "Expected: $ENGINE_PY, $ENGINE_CMD, $VLLM_PKG/__init__.py" >&2
  echo "Re-extract from the container (read-only docker cp, engine may keep running):" >&2
  echo "  docker cp vllm-glm-5.3-flash-nvfp4:/usr/local/lib/python3.12/dist-packages /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin/" >&2
  echo "  docker cp vllm-glm-5.3-flash-nvfp4:/usr/local/bin/vllm /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin/bin/vllm" >&2
  echo "  python3 -m venv --without-pip /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin/venv" >&2
  echo "  ln -s /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin/dist-packages /mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin/venv/lib/python3.12/site-packages" >&2
  echo "  then rewrite bin/vllm shebang to: #!/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin/venv/bin/python" >&2
  exit 1
fi

# Patched-file parity gate: the docker launcher bind-mounts each file from
# patched-files/ over the image's stock vllm package; the extracted tree must
# already carry those EXACT bytes (overlaid at extraction time). Fail loudly
# on drift instead of silently A/B-testing a different engine.
VLLM_PATCH_MANIFEST="$F/patched-files/manifest.txt"
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  if [ ! -f "$F/patched-files/$rel" ]; then
    echo "ERROR: missing patched file: $F/patched-files/$rel" >&2
    exit 1
  fi
  if ! cmp -s "$F/patched-files/$rel" "$VLLM_PKG/$rel"; then
    echo "ERROR: extracted vllm file differs from the patched file:" >&2
    echo "       $F/patched-files/$rel" >&2
    echo "       vs $VLLM_PKG/$rel" >&2
    echo "Re-run the extraction overlay (cp the patched file over the extracted" >&2
    echo "tree) — refusing to start a non-parity engine." >&2
    exit 1
  fi
done < "$VLLM_PATCH_MANIFEST"

# Adaptive model source: local checkpoint when present+readable, else the HF
# repo id (vLLM downloads on first boot; ~198 GB). On the host there is no
# bind mount: vLLM writes the fallback download into $HF_HOME (default
# ~/.cache/huggingface — NOTE: with XDG_CACHE_HOME pointed at the shared JIT
# cache below this lands in the same place the container's bind-mount sent it).
if [ -d "$HF_LOCAL" ] && [ -r "$HF_LOCAL" ] && [ -n "$(ls -A "$HF_LOCAL" 2>/dev/null)" ]; then
  MODEL_ID="$HF_LOCAL"
else
  MODEL_ID="$HF_FALLBACK_ID"
  HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
  mkdir -p "$HF_CACHE/hub"
  echo "NOTE: local checkpoint '$HF_LOCAL' not found/readable — serving HF repo '$MODEL_ID'" >&2
  echo "      first boot downloads ~198 GB into '$HF_CACHE' (pre-seed with: hf download $MODEL_ID)" >&2
fi

# Adaptive kernel-JIT cache fallback: .cache/ next to this script when the
# shared cache dir is not usable.
if ! mkdir -p "$JIT_CACHE/jit" "$JIT_CACHE/tilelang" 2>/dev/null; then
  echo "NOTE: '$JIT_CACHE' missing or not writable — using '$F/.cache' instead" >&2
  JIT_CACHE="$F/.cache"
  mkdir -p "$JIT_CACHE/jit" "$JIT_CACHE/tilelang"
fi

# Offload tier sizing: bytes for the connector config. The real capacity
# gate is the HOST's /dev/shm free space, checked after the pre-flight
# cleanup below (the tier mmap lives there, exactly as with --ipc=host).
CPU_TIER_BYTES=$(( CPU_TIER_GB * 1073741824 ))

# ----------------------------------------------------------------------------
# Same-port / same-pid alive check (replaces the docker launcher's name-conflict
# guard): a second start must fail LOUDLY. Must run BEFORE the shm pre-flight,
# because a live engine (docker or no-docker) owns tier mmaps in /dev/shm that
# the pre-flight would delete.
# ----------------------------------------------------------------------------
if [ -f "$PIDFILE" ]; then
  _OLD_PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "$_OLD_PID" ] && kill -0 "$_OLD_PID" 2>/dev/null \
     && grep -aq vllm "/proc/${_OLD_PID}/cmdline" 2>/dev/null; then
    echo "ERROR: a no-docker engine is ALREADY RUNNING (pid/pgid $_OLD_PID, pidfile $PIDFILE)." >&2
    echo "       NOT starting a duplicate. Stop it first:" >&2
    echo "         bash $F/$(basename "$0") stop" >&2
    exit 1
  fi
  echo "NOTE: stale pidfile $PIDFILE (pid ${_OLD_PID:-?} not alive) — removing." >&2
  rm -f "$PIDFILE"
fi
# Port guard: catches BOTH a second no-docker engine and the docker engine
# still running on this port (docker-proxy listens on 0.0.0.0:$PORT on the host).
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}\$"; then
  echo "ERROR: TCP port $PORT is already in use — NOT starting a duplicate engine." >&2
  echo "Listening sockets on $PORT:" >&2
  ss -ltnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" || true
  echo "If the Docker engine (vllm-glm-5.3-flash-nvfp4) is still up, stop it first:" >&2
  echo "  bash $F/vllm-glm-5.3-flash-nvfp4.sh stop   (or: docker stop vllm-glm-5.3-flash-nvfp4)" >&2
  echo "  — its tier mmap is wiped by this launcher's pre-flight once the port is free." >&2
  exit 1
fi

# ----------------------------------------------------------------------------
# Pre-flight: reclaim leaked host shared-memory files (measured 2026-09-12:
# five orphaned tier mmaps = 412 GiB + torch leftovers filled /dev/shm to 89%,
# which made every subsequent start fail with "Insufficient space in /dev/shm").
# The tier mmap (vllm_offload_*.mmap) lives in the HOST's /dev/shm — identical
# on bare metal (this IS the host's tmpfs; --ipc=host only mattered for the
# container seeing it). The connector unlinks it only on graceful engine exit;
# hard kills leak it, so it is wiped here. Since 2026-09-15 the wipe covers
# EVERY engine-owned leftover (any vllm*/VLLM* regular file — tier mmaps AND
# other vllm-named files such as VLLM_OBJECT_STORAGE_SHM_BUFFER_* ring
# buffers that the old narrower pattern missed), printing each removal.
# Files may be root-owned: non-root callers need passwordless sudo for a
# full wipe. psm_*/sem.mp-* are torch shared-memory leftovers from killed
# TP-rank worker groups (torch maps them then unlinks, so visible files are
# always unreferenced). All are safe to delete at this point: the alive/port
# guards above passed, so no engine (docker or no-docker) is holding a tier
# right now.
# ----------------------------------------------------------------------------
do_shm_cleanup

# tmpfs /dev/shm capacity is a MOUNT OPTION (default: half of RAM), not a
# hardware limit — raise it automatically when the tier needs more.
# With CPU_TIER_HUGETLB=1 the tier file lives on the hugetlbfs mount instead
# of /dev/shm: only torch's psm/sem files need host shm there.
if [ "$CPU_TIER_HUGETLB" = "1" ]; then
  SHM_TIER_BYTES=$(( 8 * 1073741824 ))
else
  SHM_TIER_BYTES=$CPU_TIER_BYTES
fi
SHM_NEED=$(( SHM_TIER_BYTES + 16 * 1073741824 ))  # tier + torch psm/sem headroom
SHM_TOTAL=$(df -B1 --output=size /dev/shm 2>/dev/null | tail -1)
if [ -n "$SHM_TOTAL" ] && [ "$SHM_TOTAL" -lt "$SHM_NEED" ]; then
  NEED_G=$(( (SHM_NEED + 1073741823) / 1073741824 ))
  echo "NOTE: /dev/shm is $(( SHM_TOTAL / 1073741824 )) GiB; remounting to ${NEED_G} GiB for the ${CPU_TIER_GB} GiB tier."
  if [ "$(id -u)" = "0" ]; then
    mount -o remount,size="${NEED_G}G" /dev/shm || true
  elif sudo -n true 2>/dev/null; then
    sudo -n mount -o remount,size="${NEED_G}G" /dev/shm || true
  fi
  SHM_TOTAL=$(df -B1 --output=size /dev/shm 2>/dev/null | tail -1)
fi

# Hard pre-flight gate: the tier mmap needs its full size FREE in host
# /dev/shm, plus headroom for torch's TP-rank shared-memory files.
# With CPU_TIER_HUGETLB=1 the tier file lives on hugetlbfs instead, so only
# torch's psm/sem files need shm here: the gate then requires SHM_TIER_BYTES
# (8 GiB) + headroom, NOT the full CPU_TIER_BYTES (spent on hugetlbfs).
SHM_AVAIL=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -1)
SHM_HEADROOM=$(( 4 * 1073741824 ))
SHM_GATE_NEED=$(( SHM_TIER_BYTES + SHM_HEADROOM ))
if [ -n "$SHM_AVAIL" ] && [ "$SHM_AVAIL" -lt "$SHM_GATE_NEED" ]; then
  if [ -n "$SHM_TOTAL" ] && [ "$SHM_TOTAL" -lt "$SHM_NEED" ]; then
    NEED_G=$(( (SHM_NEED + 1073741823) / 1073741824 ))
    echo "ERROR: /dev/shm is only $(( SHM_TOTAL / 1073741824 )) GiB total and the automatic remount failed (no passwordless sudo?). Run as root:" >&2
    echo "       mount -o remount,size=${NEED_G}G /dev/shm" >&2
    echo "       For persistence across reboots, add to /etc/fstab:" >&2
    echo "         tmpfs /dev/shm tmpfs rw,nosuid,nodev,size=${NEED_G}G 0 0" >&2
  else
    echo "ERROR: /dev/shm has only $(( SHM_AVAIL / 1073741824 )) GiB free; this config" >&2
    echo "       needs $(( (SHM_GATE_NEED + 1073741823) / 1073741824 )) GiB there (CPU_TIER_GB=${CPU_TIER_GB} CPU_TIER_HUGETLB=${CPU_TIER_HUGETLB}). Lower CPU_TIER_GB, or free host" >&2
    echo "       shared memory with:  sudo rm -f /dev/shm/vllm_offload_*.mmap   (leaked tier files)" >&2
    df -h /dev/shm >&2
  fi
  exit 1
fi
if [ "$CPU_TIER_HUGETLB" = "1" ]; then
  echo "KV offload tier: ${CPU_TIER_GB} GiB (${CPU_TIER_BYTES} bytes) on hugetlbfs (${VLLM_KV_OFFLOAD_HUGETLB_DIR:-/dev/hugepages}) — /dev/shm has $(( SHM_AVAIL / 1073741824 )) GiB free (torch psm/sem only)"
else
  echo "KV offload tier: ${CPU_TIER_GB} GiB (${CPU_TIER_BYTES} bytes) — /dev/shm has $(( SHM_AVAIL / 1073741824 )) GiB free"
fi

# Host tuning for the tier (best-effort; skipped with a warning when no
# passwordless sudo): a synchronous memory compaction before launch
# (vm.compact_memory) and shmem THP "advise" mode. On current kernels the
# tier runs 4 KiB pages (shmem refuses 2 MiB folios, verified 2026-09-13),
# so the advise hint is inert there — both settings are harmless and kept
# for kernels that gain shmem-THP support.
if [ -f "$F/tune-host-for-tier.sh" ]; then
  bash "$F/tune-host-for-tier.sh" || true
else
  echo "NOTE: $F/tune-host-for-tier.sh not found — skipping host tuning." >&2
fi

# Opt-in hugetlbfs tier backing (CPU_TIER_HUGETLB=1): reserve host hugepages
# for the full tier so the hugetlbfs tier file gets REAL 2 MiB pages — the
# NVIDIA driver's per-rank pinned page-table budget (~537-600 MB/rank at
# 4 KiB pages, the >=576 GiB pin ceiling) shrinks ~64x. Best-effort: on any
# failure the engine tier falls back to the stock /dev/shm backing with a
# loud warning (where 576+ GiB still fails to pin).
if [ "$CPU_TIER_HUGETLB" = "1" ]; then
  if [ -f "$F/tune-host-for-hugepages.sh" ]; then
    bash "$F/tune-host-for-hugepages.sh" "$CPU_TIER_GB" || true
  else
    echo "NOTE: $F/tune-host-for-hugepages.sh not found — the tier will fall back to the stock /dev/shm backing." >&2
  fi
fi

# ----------------------------------------------------------------------------
# Engine environment (bare-metal parity with the container):
#   * LD_LIBRARY_PATH: pip-wheel CUDA/NCCL/cuDNN lib dirs FIRST (the wheels
#     bundle the whole cu13 stack and torch/vllm resolve them via RPATH too;
#     the container resolved the same wheel libs — its /usr/local/cuda/lib64
#     held the same versions as the wheels). Host toolkit (/usr/local/cuda,
#     13.1) comes only AFTER the wheels so it can never shadow them.
#   * TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR / DG_JIT_CACHE_DIR: same
#     variable names the docker launcher exported; values re-pointed to the
#     HOST directories that the container bind-mounted onto /root/.cache
#     (so the SAME JIT cache files are reused, not forked).
#   * XDG_CACHE_HOME maps the container's /root/.cache wholesale (flashinfer,
#     cutedsl, torch hub caches land exactly where the bind mount put them).
#   * TILELANG_CACHE_DIR maps the container's /root/.tilelang bind mount.
#   * VLLM_ENGINE_READY_TIMEOUT_S: verbatim from the docker launcher.
#   * VLLM_ENABLE_CUDA_COMPATIBILITY / TORCH_CUDA_ARCH_LIST /
#     VLLM_USAGE_SOURCE: image ENV parity (docker inspect of the running
#     container) — runtime-relevant, kept.
#   * NCCL: NO NCCL_* vars were set by the docker launcher or the image;
#     engine uses torch's bundled NCCL 2.29.7. Deliberately NOT overridden.
# ----------------------------------------------------------------------------
cd "$F"
NV_LIBS="$(find "$DIST/nvidia" -type d -name lib -print 2>/dev/null | sort -u | tr '\n' ':')"
NV_LIBS="${NV_LIBS%:}"
export LD_LIBRARY_PATH="$DIST/torch/lib:$NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
mkdir -p "$JIT_CACHE/jit/triton" "$JIT_CACHE/jit/torchinductor" "$JIT_CACHE/jit/deep_gemm" "$JIT_CACHE/tilelang"
export XDG_CACHE_HOME="$JIT_CACHE/jit"                       # == container /root/.cache (bind mount)
export TRITON_CACHE_DIR="$JIT_CACHE/jit/triton"              # docker launcher env, remapped from /root/.cache/triton
export TORCHINDUCTOR_CACHE_DIR="$JIT_CACHE/jit/torchinductor" # docker launcher env, remapped
export DG_JIT_CACHE_DIR="$JIT_CACHE/jit/deep_gemm"           # docker launcher env, remapped
export TILELANG_CACHE_DIR="$JIT_CACHE/tilelang"              # == container /root/.tilelang (bind mount)
export VLLM_ENGINE_READY_TIMEOUT_S=3600                      # docker launcher env, verbatim
export VLLM_ENABLE_CUDA_COMPATIBILITY=0                      # image ENV parity
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0"  # image ENV parity
export VLLM_USAGE_SOURCE=production-docker-image             # image ENV parity
# Opt-in hugetlbfs tier backing (0/1; mount dir via VLLM_KV_OFFLOAD_HUGETLB_DIR,
# exported above). The engine patch falls back to /dev/shm loudly when set but
# unavailable, so 0 and 1 are both safe to boot.
export VLLM_KV_OFFLOAD_TIER_HUGETLB="${CPU_TIER_HUGETLB}"

# EXP-SEG 2026-09-19: reduce CUDA allocator fragmentation on the 32GiB GPUs.
# After ~11h of max-churn huge-context serving the old boot crashed with
# "CUDA out of memory. Tried to allocate 142.00 MiB ... 97.19 MiB free" on
# GPUs 4/5 (246.11 MiB reserved-but-unallocated -> no contiguous 142 MiB
# block); see logs/engine-20260919-041840.log (MPClient shutdown 15:32:21).
# PyTorch's own hint in that error was PYTORCH_CUDA_ALLOC_CONF with
# expandable_segments:True. House style ${VAR:-default} keeps an explicit
# launcher-env override possible; the engine needs a restart to pick this up.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Log file next to this script (timestamped) + a stable "latest" symlink.
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/engine-$(date +%Y%m%d-%H%M%S).log"
ln -sfn "$LOGFILE" "$LOGDIR/latest.log"

# Summary banner (also written into the log as its first lines).
{
  echo "=== no-docker launcher: $(date -Is) ==="
  echo "engine:   $ENGINE_CMD (vllm 0.1.dev20051+g487ecf187, python 3.12.3)"
  echo "model:    $MODEL_ID"
  echo "names:    $SERVED_MODEL_NAMES"
  echo "port:     $PORT (API + metrics)"
  echo "cpu_tier: ${CPU_TIER_GB} GiB (${CPU_TIER_BYTES} bytes)"
  echo "kv_cache: $KV_CACHE_MEMORY bytes (fp8)"
  echo "ld_lib:   $DIST/torch/lib + ${NV_LIBS%:}"
  echo "caches:   triton=$TRITON_CACHE_DIR inductor=$TORCHINDUCTOR_CACHE_DIR deep_gemm=$DG_JIT_CACHE_DIR tilelang=$TILELANG_CACHE_DIR"
} | tee -a "$LOGFILE"
echo "KV offload tier: ${CPU_TIER_GB} GiB (${CPU_TIER_BYTES} bytes) — logs: $LOGFILE"

# Launch the engine DIRECTLY (no docker). setsid gives the engine its own
# session + process group with PGID == ENGINE_PID, so the stop path can kill
# the whole group (vllm TP workers included) exactly once.
# NOTE: $SERVED_MODEL_NAMES is expanded UNQUOTED on purpose — several names
# word-split into separate --served-model-name tokens (default now serves
# BOTH "glm-5.3-flash" and "qwen-3.8-flash-next").
setsid "$ENGINE_CMD" serve "$MODEL_ID" \
  --served-model-name $SERVED_MODEL_NAMES \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-cache-memory "$KV_CACHE_MEMORY" \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":'"$SPEC_TOKENS"'}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4]}' \
  --kernel-config '{"enable_jit_warmup":true,"enable_cutedsl_warmup":true}' \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser deepseek_r1 \
  --block-size 256 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-metrics \
  --enable-mfu-metrics \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics \
  --enable-chunked-prefill \
  --optimization-level 3 \
  --async-scheduling \
  --jit-monitor-verbose \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": '"$CPU_TIER_BYTES"',
      "offload_prompt_only": false
    }
  }' \
  >>"$LOGFILE" 2>&1 < /dev/null &
ENGINE_PID=$!

# Record the launch PID (== the engine process-group id thanks to setsid).
echo "$ENGINE_PID" > "$PIDFILE"
sleep 2
if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
  # setsid forked unexpectedly: recover the real engine pid via pgrep.
  _real=$(pgrep -f "$ENGINE_CMD serve" 2>/dev/null | tail -1 || true)
  if [ -n "${_real:-}" ]; then
    ENGINE_PID="$_real"
    echo "$ENGINE_PID" > "$PIDFILE"
  fi
fi
if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
  echo "ERROR: engine died immediately — log tail:" >&2
  tail -n 40 "$LOGFILE" >&2
  rm -f "$PIDFILE"
  exit 1
fi
echo "[launcher] engine PID=$ENGINE_PID (PGID=$ENGINE_PID), pidfile=$PIDFILE, log=$LOGFILE"
echo "[launcher] follow: tail -f $LOGFILE"

# ----------------------------------------------------------------------------
# Readiness gate (same as the docker kit expects): wait for the engine to bind
# $PORT and print "Application startup complete", then report.
# ----------------------------------------------------------------------------
READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
STARTED=$SECONDS
HEALTH_OK=0
LOG_OK=0
while :; do
  ELAPSED=$(( SECONDS - STARTED ))
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "ERROR: engine process $ENGINE_PID exited during startup (after ${ELAPSED}s) — log tail:" >&2
    tail -n 50 "$LOGFILE" >&2
    rm -f "$PIDFILE"
    do_shm_cleanup
    exit 1
  fi
  _CODE=$(curl -s -o /dev/null -w '%{http_code}' "$HEALTH_URL" 2>/dev/null || true)
  if [ "${_CODE:-}" = "200" ]; then
    HEALTH_OK=1
  fi
  if grep -q 'Application startup complete' "$LOGFILE" 2>/dev/null; then
    LOG_OK=1
  fi
  if [ "$HEALTH_OK" = 1 ] && [ "$LOG_OK" = 1 ]; then
    break
  fi
  if [ "$ELAPSED" -ge "$READY_TIMEOUT_S" ]; then
    echo "ERROR: readiness gate timed out after ${READY_TIMEOUT_S}s" >&2
    echo "       curl $HEALTH_URL -> ${_CODE:-none}; 'Application startup complete'" >&2
    echo "       seen in log: $([ "$LOG_OK" = 1 ] && echo yes || echo no)" >&2
    echo "       The engine process is STILL RUNNING (PID $ENGINE_PID) — inspect:" >&2
    echo "         tail -f $LOGFILE" >&2
    echo "       or stop it: bash $F/$(basename "$0") stop" >&2
    exit 1
  fi
  if [ $(( ELAPSED % 30 )) -lt 5 ] && [ "$ELAPSED" -gt 0 ]; then
    echo "[launcher] waiting for readiness... ${ELAPSED}s elapsed (last log line: $(tail -n 1 "$LOGFILE" 2>/dev/null | cut -c1-120))"
  fi
  sleep 5
done
echo "[launcher] READY: $HEALTH_URL -> HTTP 200 and 'Application startup complete' in log"
echo "[launcher] API + metrics: http://0.0.0.0:${PORT}  (metrics at /metrics)"
echo "[launcher] stop:   bash $F/$(basename "$0") stop"
echo "[launcher] status: bash $F/$(basename "$0") status"
