#!/usr/bin/env bash
# ============================================================================
# GLM-5.3-Flash-NVFP4 on vLLM — 8x RTX 5090 (sm_120) — PRODUCTION CONFIG
# with KV CPU offloading (64 GiB) enabled.
#
# This is the primary launcher. Config verified on testcomp2 on 2026-09-12:
#   * mounts the 6 PATCHED FILES from patched-files/ over the image's stock
#     vllm package (per-file mounts; proven byte-identical to the original
#     full-tree mount: a full diff of the tree vs the image's stock package
#     showed exactly these 6 files differ — see patched-files/manifest.txt
#     and patches/README.md)
#   * KV offloading: OffloadingConnector, kv_both, CPU_TIER_GB CPU budget
#     (default 256 GiB; see EDITABLE SETTINGS to change it)
#     (total across TP=8 -> ~8 GiB pinned/rank, shared region in /dev/shm;
#     that is why /dev/shm is auto-sized (SHM_SIZE) and why tier-sized RAM
#     is used)
#   * GPU KV pool: KV_CACHE_MEMORY-sized (default 3.3e9 -> 414,634 tokens,
#     fp8; 4000000000 -> ~502k tokens — proven to boot standalone)
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
# GPU KV cache budget in BYTES per engine (fp8 KV). Default 3.3e9 -> 414,634
# tokens of pool (validated 100%). 4000000000 -> ~502k tokens — proven to
# boot in a no-offload launch; the +offload combination adds only tiny GPU
# staging buffers, so it is expected to boot but not yet burn-tested: if a
# boot with 4.0e9 + the tier fails, drop back to the default.
: "${KV_CACHE_MEMORY:=3300000000}"

# Names the model is advertised under in the OpenAI-compatible API,
# space-separated (expanded unquoted in `docker run` on purpose, so that
# several names word-split into separate --served-model-name tokens).
# Default: just "glm-5.3-flash".
: "${SERVED_MODEL_NAMES:=glm-5.3-flash}"

# Docker container name + host port. Both launchers of this kit share the
# SAME name+port, so starting one stops the other automatically.
# (test.sh assumes the default port.)
: "${CONTAINER_NAME:=vllm-glm-5.3-flash-nvfp4}"
: "${PORT:=1025}"

# KV offloading CPU tier budget in GiB — a pinned, fully-preallocated mmap
# in the HOST's /dev/shm (shared via --ipc=host). The launcher remounts
# /dev/shm larger automatically when this exceeds the default 50%-of-RAM
# shm limit (tmpfs size is a cap, not a reservation). Default 256 —
# validated 2026-09-12 (boots, serves, absorbed >2x the old 64-GiB cap;
# upstream vllm-project/vllm#52656 crash reports applied to other stacks).
# NOTE: the tier is charged to RAM at boot and PINNED via cudaHostRegister
# (unpageable) — keep ~60+ GiB physical RAM for OS + engine processes:
# on this 1007-GiB machine that puts the practical ceiling near ~900 GiB,
# lower if the box runs other big software simultaneously.
: "${CPU_TIER_GB:=256}"

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

# ----------------------------------------------------------------------------
# Pre-flight: reclaim leaked host shared-memory files (measured 2026-09-12:
# five orphaned tier mmaps = 412 GiB + torch leftovers filled /dev/shm to 89%,
# which made every subsequent start fail with "Insufficient space in /dev/shm").
# The tier mmap (vllm_offload_*.mmap) lives in the HOST's /dev/shm because
# --ipc=host shares it. The connector unlinks it only on graceful engine exit;
# docker stop/rm SIGKILLs the engine, so EVERY restart leaks the tier file
# until cleaned here. Tier files are root-owned: non-root callers need
# passwordless sudo for a full wipe. psm_*/sem.mp-* are torch shared-memory
# leftovers from killed TP-rank worker groups (torch maps them then unlinks,
# so visible files are always unreferenced). All are safe to delete at this
# point: the container was just stopped and nothing else on this host uses
# host shm.
# ----------------------------------------------------------------------------
SHM_CLEAN_CODE='
import os
freed = {"tier_mmap": 0, "torch_psm": 0, "torch_sem": 0}
denied = 0
for name in os.listdir("/dev/shm"):
    path = os.path.join("/dev/shm", name)
    try:
        size = os.stat(path).st_size
    except OSError:
        continue
    cls = ("tier_mmap" if name.startswith("vllm_offload_") and name.endswith(".mmap")
           else "torch_psm" if name.startswith("psm_")
           else "torch_sem" if name.startswith("sem.mp-")
           else None)
    if cls is None:
        continue
    try:
        os.unlink(path)
        freed[cls] += size
    except PermissionError:
        denied += size
    except OSError:
        pass
msg = ", ".join(f"{k}={v / 2**30:.1f} GiB" for k, v in freed.items())
if denied:
    msg += f" ({denied / 2**30:.1f} GiB skipped: permission denied — run: sudo rm -f /dev/shm/vllm_offload_*.mmap)"
print("Pre-flight shm cleanup: " + msg)
'
if command -v python3 >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    python3 -c "$SHM_CLEAN_CODE" || true
  elif sudo -n true 2>/dev/null && sudo -n python3 -c "$SHM_CLEAN_CODE" 2>/dev/null; then
    : # full wipe done via passwordless sudo
  else
    python3 -c "$SHM_CLEAN_CODE" || true
  fi
else
  echo "WARNING: python3 not found — skipped /dev/shm orphan cleanup." >&2
fi

# tmpfs /dev/shm capacity is a MOUNT OPTION (default: half of RAM), not a
# hardware limit — raise it automatically when the tier needs more.
SHM_NEED=$(( CPU_TIER_BYTES + 16 * 1073741824 ))  # tier + torch psm/sem headroom
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
SHM_AVAIL=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -1)
SHM_HEADROOM=$(( 4 * 1073741824 ))
if [ -n "$SHM_AVAIL" ] && [ "$SHM_AVAIL" -lt $(( CPU_TIER_BYTES + SHM_HEADROOM )) ]; then
  if [ -n "$SHM_TOTAL" ] && [ "$SHM_TOTAL" -lt "$SHM_NEED" ]; then
    NEED_G=$(( (SHM_NEED + 1073741823) / 1073741824 ))
    echo "ERROR: /dev/shm is only $(( SHM_TOTAL / 1073741824 )) GiB total and the automatic remount failed (no passwordless sudo?). Run as root:" >&2
    echo "       mount -o remount,size=${NEED_G}G /dev/shm" >&2
    echo "       For persistence across reboots, add to /etc/fstab:" >&2
    echo "         tmpfs /dev/shm tmpfs rw,nosuid,nodev,size=${NEED_G}G 0 0" >&2
  else
    echo "ERROR: /dev/shm has only $(( SHM_AVAIL / 1073741824 )) GiB free; the ${CPU_TIER_GB} GiB" >&2
    echo "       offload tier needs $(( (CPU_TIER_BYTES + SHM_HEADROOM) / 1073741824 )) GiB. Lower CPU_TIER_GB, or free host" >&2
    echo "       shared memory with:  sudo rm -f /dev/shm/vllm_offload_*.mmap   (leaked tier files)" >&2
    df -h /dev/shm >&2
  fi
  exit 1
fi
echo "KV offload tier: ${CPU_TIER_GB} GiB (${CPU_TIER_BYTES} bytes) — /dev/shm has $(( SHM_AVAIL / 1073741824 )) GiB free"

docker run --restart=unless-stopped --gpus all --ipc=host --shm-size "${SHM_SIZE}g" -p "$PORT:$PORT" \
  --name "$CONTAINER_NAME" \
  --cap-add SYS_NICE \
  "${MODEL_MOUNTS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  "${MOUNTS[@]}" \
  -v "$JIT_CACHE/jit":/root/.cache \
  -v "$JIT_CACHE/tilelang":/root/.tilelang \
  -e DG_JIT_CACHE_DIR=/root/.cache/deep_gemm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
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
      "cpu_bytes_to_use": '"$CPU_TIER_BYTES"'
    }
  }'
