#!/usr/bin/env bash
# ============================================================================
# GLM-5.3-Flash-NVFP4 on vLLM — 8x RTX 5090 (sm_120) — MTP VARIANT
# with KV CPU offloading enabled (CPU_TIER_GB, default 512 GiB) and
# NEXTN MTP speculative decoding at SPEC_TOKENS depth (default 3).
#
# Byte-identical to vllm-glm-5.3-flash-nvfp4.sh (2026-09-16 build with the
# CPU_TIER_HUGETLB parity work), opening banner aside, except:
#   + --speculative-config '{"method":"mtp","num_speculative_tokens":N}'
#   + --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4]}'
# Same container name + port as the production launcher, so starting this
# stops the production engine automatically (single-tenant swap). All the
# production launcher's notes below still apply to THIS file:
#
# Production launcher provenance. Config verified on testcomp2 on
# 2026-09-12 (tier ceiling re-validated 2026-09-13):
#   * mounts the 14 PATCHED FILES from patched-files/ over the image's stock
#     vllm package (per-file mounts; proven byte-identical to the original
#     full-tree mount: a full diff of the tree vs the image's stock package
#     showed exactly these files differ — see patched-files/manifest.txt
#     and patches/README.md)
#   * KV offloading: OffloadingConnector, kv_both, CPU_TIER_GB CPU budget
#     (default 512 GiB — the validated ceiling; see EDITABLE SETTINGS)
#     (the tier region is allocated once in the HOST's /dev/shm —
#     512 GiB = 8 ranks x 64 GiB pinned each — and registered into every
#     TP rank's GPU context; that is why /dev/shm
#     is auto-sized (SHM_SIZE) and why tier-sized RAM is charged).
#     CPU_TIER_HUGETLB=1 (no-docker parity, added 2026-09-16) instead hosts
#     the tier on a 2 MiB-page hugetlbfs file in $VLLM_KV_OFFLOAD_HUGETLB_DIR
#     (bind-mounted read-write into the container), where the driver's
#     per-rank pinned page-table budget shrinks ~64x so >= 800 GiB tiers
#     pin — see EDITABLE SETTINGS + the hugetlb pre-flight below.
#   * GPU KV pool: KV_CACHE_MEMORY-sized (default 4.0e9 -> ~502k tokens, fp8;
#     raised 2026-09-14 from 3.3e9/414,634 tokens — captured APC-HIT evidence
#     showed concurrent ~140k agent sessions (esp. parallel-turn bursts,
#     MAX_NUM_SEQS=4) evict whole cached chains between turns; 4.0e9 was
#     previously proven to boot standalone)
#   * auto-restarts on boot/reboot (unless-stopped), port 1025
#
# Adaptive fallbacks (fresh-machine friendly):
#   * kernel JIT cache dir: uses /mnt/data/shared/models/vllm-moet-cache
#     when usable, else creates .cache/ next to this script and mounts that
#     instead (this is the kernel JIT cache, NOT the HuggingFace model cache)
#   * model: uses the local RedHatAI checkpoint when present+readable, else
#     passes the HF repo id so vLLM downloads it (into the mounted HF cache)
#
# All tunables live in the EDITABLE SETTINGS block right below the header.
#
# Usage:
#   bash vllm-glm-5.3-flash-nvfp4.sh        (start; engine runs inside docker
#         with --restart=unless-stopped, so it auto-restarts on boot/reboot)
#   bash vllm-glm-5.3-flash-nvfp4.sh stop   (docker container stop, then
#         reclaim leaked engine-owned files from host /dev/shm — vllm*/VLLM*
#         files only, so a live sibling engine's torch psm_/sem.mp files
#         survive a stop)
#
# Provenance / patch refs (see README.md and patches/README.md):
#   image cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
#     pinned sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5
#     (config digest == registry manifest digest for this build)
#   fork commit g487ecf187; PR vllm-project/vllm#54743 (unmerged,
#   head 899699c74ae2b8e8adc8726e5c9d0e355935076a) — preserved on branch
#   pr-54743-kv-offload-prefix-cacheable of the jdjohndoe13/vllm fork.
# ============================================================================
set -euo pipefail
F="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# Pre-flight / post-stop shared-memory leak cleanup — mirrors the no-docker
# launcher's do_shm_cleanup (broadened 2026-09-15). Reclaims ANY engine-owned
# orphan in the HOST's /dev/shm — the tier mmaps (vllm_offload_*.mmap, which
# live here because --ipc=host shares the host tmpfs) AND any other
# vllm*/VLLM* leftover, e.g. leaked VLLM_OBJECT_STORAGE_SHM_BUFFER_* ring
# buffers (vllm auto-names these UPPERCASE per process tree — vllm/envs.py
# get_env_or_set_default — which the old vllm_offload_*.mmap-only pattern
# never matched, forcing a manual rm) — plus torch psm_/sem.mp leftovers.
# Every removed file is printed with its size and a removed-count/bytes
# summary closes the run, so misses are never silent. An optional
# "tier_only" scope (arg 1 of do_shm_cleanup) removes only vllm*/VLLM*
# files — the stop subcommand uses it so a live sibling engine's torch
# files survive. With CPU_TIER_HUGETLB=1 the tier mmap lives in
# $VLLM_KV_OFFLOAD_HUGETLB_DIR (default /dev/hugepages, bind-mounted into
# the container), so the cleanup scans that dir too as soon as it exists —
# same as the no-docker launcher's leak-cleanup (2026-09-16 parity).
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
  # files (the stop subcommand uses it so a live sibling engine's torch
  # psm_/sem.mp files are never touched); default: full wipe.
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

# Docker container name + host port (defaults set HERE, before the stop
# subcommand dispatch, which needs them to target the container/port — the
# VAR=value argument loop below still applies to the start path). Both
# launchers of this kit share the SAME name+port, so starting one stops the
# other automatically. (test.sh assumes the default port.)
: "${CONTAINER_NAME:=vllm-glm-5.3-flash-nvfp4}"
: "${PORT:=1025}"

# ----------------------------------------------------------------------------
# stop subcommand — `docker container stop` for operator convenience, then
# reclaim what docker kills leak: the connector unlinks the tier mmap only
# on graceful engine exit, so every container kill/rm leaves engine-owned
# files in the HOST's /dev/shm until wiped here (docker rm alone never
# cleaned them). Scope is deliberately "tier_only" (vllm*/VLLM* files
# only): torch psm_/sem.mp files are left for the start pre-flight, so a
# live sibling engine on this host (same port — e.g. the no-docker
# launcher's) is never disturbed by a stop.
# ----------------------------------------------------------------------------
cmd_stop() {
  echo "[launcher] stopping container $CONTAINER_NAME (docker container stop)..."
  docker container stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q '^true$'; then
    echo "[launcher] container $CONTAINER_NAME is STILL RUNNING — NOT wiping host /dev/shm." >&2
    exit 1
  fi
  echo "[launcher] container $CONTAINER_NAME confirmed down (or absent)."
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}\$"; then
    echo "[launcher] NOTE: port $PORT is still in use — another engine may be live on this host;"
    echo "           wiping vllm*/VLLM* shm leftovers only (psm_/sem.mp untouched by stop)."
  fi
  do_shm_cleanup tier_only
}

case "${1-}" in
  stop) cmd_stop; exit 0 ;;
esac

# ============================================================================
# EDITABLE SETTINGS — view/edit right here, or override from the command line:
#   MAX_MODEL_LEN=150000 bash vllm-glm-5.3-flash-nvfp4.sh        (env prefix)
#   bash vllm-glm-5.3-flash-nvfp4.sh MAX_MODEL_LEN=150000        (as argument)
#   MAX_MODEL_LEN=150000 ./vllm-glm-5.3-flash-nvfp4.sh           (direct exec)
#   MAX_MODEL_LEN=150000 MAX_NUM_SEQS=6 bash vllm-glm-5.3-flash-nvfp4.sh
#   KV_CACHE_MEMORY=4000000000 bash vllm-glm-5.3-flash-nvfp4.sh  (~502k pool)
# NOTE: 'bash MAX_MODEL_LEN=150000 <script>.sh' does NOT work — bash treats
#       the assignment as the script's filename.
# ============================================================================

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

# Docker image: tag (readable name) + PINNED build (what actually runs).
# The guard below refuses to start when the pinned ID is absent or the tag
# points at a different build.
: "${IMAGE:=cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1}"
: "${IMAGE_ID:=sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5}"

# Model: local checkpoint dir (mounted read-only) when present+readable, else
# HF_FALLBACK_ID is passed and vLLM downloads the checkpoint (~198 GB) into
# the HuggingFace cache on first boot. NOTE: use the RedHatAI checkpoint —
# the LibertAIDAI one emits corrupted tokens on sm_120 (vllm-project/vllm#54150).
: "${HF_LOCAL:=/mnt/huggingface/RedHatAI/GLM-5.3-Flash-NVFP4}"
: "${HF_FALLBACK_ID:=RedHatAI/GLM-5.3-Flash-NVFP4}"

# Shared kernel JIT cache (Triton/deep_gemm/tilelang — NOT the HuggingFace
# model cache). Falls back to <script-dir>/.cache when not usable.
: "${JIT_CACHE:=/mnt/data/shared/models/vllm-moet-cache}"

# Server sizing / behavior:
: "${MAX_MODEL_LEN:=200000}"
: "${MAX_NUM_SEQS:=4}"
# GPU KV cache budget in BYTES per engine (fp8 KV). Default 3.3e9 ->
# 414,634 tokens of pool (validated). 4000000000 -> ~502k tokens;
# 5000000000 -> ~628k tokens (proven to boot WITH the tier in a
# 196k-token, 2-concurrent-request test). If a boot with a larger pool
# + tier fails, drop back to the default.
: "${KV_CACHE_MEMORY:=3000000000}"

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
# NOTE: MTP x OffloadingConnector (the KV CPU tier) is UNVALIDATED on this
# stack — the first boot IS the smoke test; on failure bisect with
# CPU_TIER_GB=0 (tier disabled) to isolate.
# ---------------------------------------------------------------------------
: "${SPEC_TOKENS:=3}"

# Names the model is advertised under in the OpenAI-compatible API,
# space-separated (expanded unquoted in `docker run` on purpose, so that
# several names word-split into separate --served-model-name tokens).
# Default: just "qwen-3.8-flash-next" (unchanged — only the no-docker
# launcher serves BOTH names by default).
: "${SERVED_MODEL_NAMES:=qwen-3.8-flash-next}"

# Docker container name + host port: defaults are set ABOVE (before the
# stop subcommand dispatch, which needs them). Both launchers of this kit
# share the SAME name+port, so starting one stops the other automatically.
# (test.sh assumes the default port.)

# KV offloading CPU tier budget in GiB — a pinned, fully-preallocated mmap
# in the HOST's /dev/shm (shared via --ipc=host). The launcher remounts
# /dev/shm larger automatically when this exceeds the default 50%-of-RAM
# shm limit (tmpfs size is a cap, not a reservation).
# Default 512 GiB — the validated ceiling on this host/driver (2026-09-13:
# 0 cudaHostRegister failures, mHC warmup green). 576 GiB and above fail
# on ALL ranks with the NVIDIA driver's "NVRM: failed to allocate page
# table" — a per-rank pinned-region page-table budget, independent of
# free RAM or compaction. 256 GiB and 128 GiB validated 2026-09-12.
# NOTE: the tier is charged to RAM at boot and PINNED via cudaHostRegister
# (unpageable) — keep ~60+ GiB physical RAM for OS + engine processes;
# lower this (or stop other big software) if the box runs other big jobs.
: "${CPU_TIER_GB:=512}"

# Opt-in hugetlbfs backing for the tier (default 0 = stock /dev/shm tier).
# CPU_TIER_HUGETLB=1 mirrors the no-docker launcher (added 2026-09-16): the
# tier file is created on a hugetlbfs mount (2 MiB pages) in
# $VLLM_KV_OFFLOAD_HUGETLB_DIR (bind-mounted read-write into the container),
# where the NVIDIA driver's per-rank pinned page-table budget (~537-600
# MB/rank at 4 KiB pages — the >= 576 GiB "NVRM: failed to allocate page
# table" ceiling) shrinks ~64x, so 800 GiB tiers can pin at all. The engine
# env gets VLLM_KV_OFFLOAD_TIER_HUGETLB=1; any hugetlb failure at boot
# falls back loudly to the stock /dev/shm tier (where 576+ GiB still fails).
: "${CPU_TIER_HUGETLB:=0}"
# hugetlbfs mount for the tier (used by the engine patch; exported so the
# leak-cleanup wipe covers it too, and bind-mounted into the container).
: "${VLLM_KV_OFFLOAD_HUGETLB_DIR:=/dev/hugepages}"

# Eviction-tombstone registry capacity (ENTRIES) for the CPU-tier KV
# offload connector (VLLM_KV_OFFLOAD_EVICTION_TOMBSTONES; patched
# v1/kv_offload/cpu/manager.py). Pure evicted-then-recomputed
# ATTRIBUTION: a bounded FIFO of block hashes; when full the oldest
# entries silently age out (vllm:kv_offload_eviction_tombstone_overflows_
# total climbs) and NOTHING in the cache path gates on it. Engine
# default is 262,144; kit default 8,000,000 (pinned 2026-09-16 per user)
# = ~30x the sampled soak churn (~1.1k tombstones/iter on the 800 GiB
# tier), so it should never overflow. RAM: ~1 GB extra engine RSS at
# 8M entries (OrderedDict internals + small bytes object per hash —
# one scheduler-side registry, not multiplied per TP rank). 0 disables
# the registry (attribution metrics go dark; engine otherwise intact).
: "${EVICTION_TOMBSTONES:=8000000}"

# Container --shm-size flag (GiB). NOTE: with --ipc=host Docker IGNORES this
# flag — the engine shares the HOST's /dev/shm (host default: half of RAM,
# ~504 GiB here, shared with every other --ipc=host container). This only
# matters if --ipc=host were ever removed. Auto-sized when empty.
: "${SHM_SIZE:=}"

# ============================================================================
# Everything below is derived logic — usually no need to touch.
# ============================================================================

# Guard: pinned image must exist locally, and the tag must not have drifted.
if ! docker image inspect "$IMAGE_ID" >/dev/null 2>&1; then
  echo "ERROR: pinned image $IMAGE_ID is not present locally." >&2
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "The tag '$IMAGE' exists but is NOT the pinned build:" >&2
    docker image inspect "$IMAGE" --format '  tag points to: {{.Id}}' >&2
    echo "Refusing to run a possibly-drifted image. Obtain the pinned one:" >&2
    echo "  docker load < glm-image.tar                     (backup-image.sh tarball)" >&2
    echo "  or docker pull cstechdev/vllm@$IMAGE_ID" >&2
  else
    echo "Obtain it first:" >&2
    echo "  docker load < glm-image.tar                     (backup-image.sh tarball)" >&2
    echo "  or docker pull cstechdev/vllm@$IMAGE_ID" >&2
  fi
  exit 1
fi

# Adaptive model source: local checkpoint when present+readable, else the HF
# repo id (vLLM downloads on first boot into the mounted HF cache; ~198 GB).
MODEL_MOUNTS=()
if [ -d "$HF_LOCAL" ] && [ -r "$HF_LOCAL" ] && [ -n "$(ls -A "$HF_LOCAL" 2>/dev/null)" ]; then
  MODEL_ID="$HF_LOCAL"
  MODEL_MOUNTS+=( -v /mnt/huggingface:/mnt/huggingface:ro )
else
  MODEL_ID="$HF_FALLBACK_ID"
  HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
  mkdir -p "$HF_CACHE/hub"
  MODEL_MOUNTS+=( -v "$HF_CACHE:/root/.cache/huggingface" )
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

# per-file mounts of the patched files over the image's stock vllm package
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
MOUNTS=()
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  if [ ! -f "$F/patched-files/$rel" ]; then
    echo "ERROR: missing patched file: $F/patched-files/$rel" >&2
    exit 1
  fi
  MOUNTS+=( -v "$F/patched-files/$rel:$VLLM_PKG/$rel:ro" )
done < "$F/patched-files/manifest.txt"

# Offload tier sizing: bytes for the connector config. The real capacity
# gate is the HOST's /dev/shm free space, checked after the pre-flight
# cleanup below (the tier mmap lives there — --ipc=host shares it).
CPU_TIER_BYTES=$(( CPU_TIER_GB * 1073741824 ))
if [ -z "$SHM_SIZE" ]; then
  SHM_SIZE=$(( CPU_TIER_GB + 32 > 128 ? CPU_TIER_GB + 32 : 128 ))
fi

# switch guard: stop/remove any previous instance (also used when switching
# to/from the -orig launcher, which uses the same container name)
docker container stop "$CONTAINER_NAME" 2>/dev/null || true
docker container rm -f "$CONTAINER_NAME" 2>/dev/null || true

# name-conflict guard (added 2026-09-15, extended same day per user request):
# `rm -f` right after a stop can fail transiently ("removal of container ...
# is already in progress") — the old code swallowed that with `|| true`, so
# the later `docker run` aborted with 'The container name "/..." is already
# in use by container ...'. Poll until the name is genuinely free: up to 10
# attempts, 5 s apart, each attempt printing progress (so an interactive
# caller can see it working instead of a silent pause) and retrying the
# removal. If the name is STILL occupied after ~50 s, fail loudly instead
# of with the cryptic docker conflict.
_name_free=0
for _attempt in 1 2 3 4 5 6 7 8 9 10; do
  _existing=$(docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.ID}}' 2>/dev/null || true)
  if [ -z "$_existing" ]; then _name_free=1; break; fi
  echo "[launcher] attempt ${_attempt}/10: container name '$CONTAINER_NAME' still in use (${_existing}) — removing and waiting 5 s ..."
  docker container rm -f "$_existing" >/dev/null 2>&1 || true
  sleep 5
done
if [ "$_name_free" -ne 1 ]; then
  echo "ERROR: container name '$CONTAINER_NAME' still in use after 10" \
       "remove attempts over ~50 s — docker daemon may be wedged or another" \
       "engine is holding the name. NOT starting a duplicate." >&2
  exit 1
fi

# ----------------------------------------------------------------------------
# Pre-flight: reclaim leaked host shared-memory files (measured 2026-09-12:
# five orphaned tier mmaps = 412 GiB + torch leftovers filled /dev/shm to 89%,
# which made every subsequent start fail with "Insufficient space in /dev/shm").
# The tier mmap (vllm_offload_*.mmap) lives in the HOST's /dev/shm because
# --ipc=host shares it. The connector unlinks it only on graceful engine exit;
# docker stop/rm SIGKILLs the engine, so EVERY restart leaks the tier file
# until cleaned here. Since 2026-09-15 the wipe covers EVERY engine-owned
# leftover (any vllm*/VLLM* regular file — tier mmaps AND other vllm-named
# files such as VLLM_OBJECT_STORAGE_SHM_BUFFER_* ring buffers that the old
# narrower pattern missed), printing each removal. Files are root-owned:
# non-root callers need passwordless sudo for a full wipe. psm_*/sem.mp-* are
# torch shared-memory leftovers from killed TP-rank worker groups (torch maps
# them then unlinks, so visible files are always unreferenced). All are safe
# to delete at this point: the container was just stopped and nothing else on
# this host uses host shm (the stop subcommand above is deliberately
# narrower — tier_only — for sibling-engine safety).
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
# With CPU_TIER_HUGETLB=1 the tier lives on hugetlbfs instead (8 GiB +
# headroom gate — the full CPU_TIER_BYTES are charged to the hugepages
# pool, checked right below), NOT the full size in /dev/shm.
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
  echo "KV offload tier: ${CPU_TIER_GB} GiB (${CPU_TIER_BYTES} bytes) on hugetlbfs (${VLLM_KV_OFFLOAD_HUGETLB_DIR}) — /dev/shm has $(( SHM_AVAIL / 1073741824 )) GiB free (torch psm/sem only)"
else
  echo "KV offload tier: ${CPU_TIER_GB} GiB (${CPU_TIER_BYTES} bytes) — /dev/shm has $(( SHM_AVAIL / 1073741824 )) GiB free"
fi

# ----------------------------------------------------------------------------
# Hugepages pool sanity + container wiring (hugetlb mode only, PARITY with
# the no-docker launcher). The engine creates vllm_offload_*.mmap itself in
# $VLLM_KV_OFFLOAD_HUGETLB_DIR and falls back loudly to the /dev/shm tier
# when hugetlb cannot back CPU_TIER_BYTES — warn early here instead of
# dying mid-boot. Read-only pool check (the pool is reserved host-wide, NOT
# by this launcher); then bind-mount the dir read-write and feed the engine
# patch its two env vars via docker -v/-e.
# ----------------------------------------------------------------------------
HUGETLB_ARGS=()
if [ "$CPU_TIER_HUGETLB" = "1" ]; then
  if [ ! -d "$VLLM_KV_OFFLOAD_HUGETLB_DIR" ]; then
    echo "ERROR: CPU_TIER_HUGETLB=1 but '$VLLM_KV_OFFLOAD_HUGETLB_DIR' does not exist on the host." >&2
    exit 1
  fi
  _HP_FREE=$(awk '/^HugePages_Free:/ {print $2}' /proc/meminfo 2>/dev/null || true)
  _HP_SIZE_KB=$(awk '/^Hugepagesize:/ {print $2}' /proc/meminfo 2>/dev/null || true)
  if [ -z "$_HP_FREE" ] || [ -z "$_HP_SIZE_KB" ]; then
    echo "WARNING: /proc/meminfo has no HugePages info — cannot pre-check the hugepages pool; relying on the engine's loud hugetlb fallback." >&2
  elif [ "$(( _HP_FREE * _HP_SIZE_KB * 1024 ))" -lt "$CPU_TIER_BYTES" ]; then
    echo "WARNING: hugepages pool has only $(( (_HP_FREE * _HP_SIZE_KB * 1024 + 1073741823) / 1073741824 )) GiB free (< tier ${CPU_TIER_GB} GiB)" >&2
    echo "         — engine still boots, but falls back loudly to the /dev/shm tier." >&2
  fi
  HUGETLB_ARGS+=( -v "$VLLM_KV_OFFLOAD_HUGETLB_DIR":"$VLLM_KV_OFFLOAD_HUGETLB_DIR" \
                  -e VLLM_KV_OFFLOAD_TIER_HUGETLB=1 \
                  -e VLLM_KV_OFFLOAD_HUGETLB_DIR="$VLLM_KV_OFFLOAD_HUGETLB_DIR" )
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

docker run --restart=unless-stopped --gpus all --ipc=host --shm-size "${SHM_SIZE}g" -p "$PORT:$PORT" \
  --name "$CONTAINER_NAME" \
  --cap-add SYS_NICE \
  "${MODEL_MOUNTS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_KV_OFFLOAD_EVICTION_TOMBSTONES="${EVICTION_TOMBSTONES}" \
  "${MOUNTS[@]}" \
  -v "$JIT_CACHE/jit":/root/.cache \
  -v "$JIT_CACHE/tilelang":/root/.tilelang \
  -e DG_JIT_CACHE_DIR=/root/.cache/deep_gemm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  ${HUGETLB_ARGS[@]+"${HUGETLB_ARGS[@]}"} \
  "$IMAGE_ID" "$MODEL_ID" \
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
  --max-num-batched-tokens 2048 \
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
  }'
