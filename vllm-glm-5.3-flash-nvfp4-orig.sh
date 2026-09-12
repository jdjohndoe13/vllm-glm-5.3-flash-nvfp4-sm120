#!/usr/bin/env bash
# ============================================================================
# GLM-5.3-Flash-NVFP4 on vLLM — 8x RTX 5090 (sm_120) — FALLBACK: no KV offload
#
# Same server as vllm-glm-5.3-flash-nvfp4.sh but WITHOUT KV CPU offloading
# (uses the image's stock vllm package + the two overlay files mounted
# individually instead of the offload-patched set). Use this if you want to
# compare behavior or if the offload config ever misbehaves. Same container
# name + port: stop the other one first (both launchers do this guard
# themselves).
#
# Adaptive fallbacks (identical to the primary launcher):
#   * kernel JIT cache dir: /mnt/data/shared/models/vllm-moet-cache when
#     usable, else .cache/ next to this script (kernel JIT cache — NOT the
#     HuggingFace model cache)
#   * model: local RedHatAI checkpoint when present+readable, else the HF
#     repo id (vLLM downloads on first boot into the mounted HF cache)
#
# NOTE on the two file overlays (shipped under patched-files/, mirroring the
# package tree — the same files the offload launcher mounts, see manifest):
#   * model_executor/layers/quantization/modelopt.py   (SM120 NVFP4 fix)
#   * model_executor/warmup/deepseek_v4_mhc_warmup.py  (mHC kernel warmup)
#
# All tunables live in the EDITABLE SETTINGS block right below the header.
# ============================================================================
set -euo pipefail
F="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# EDITABLE SETTINGS — same override forms as the primary launcher:
#   MAX_MODEL_LEN=150000 bash vllm-glm-5.3-flash-nvfp4-orig.sh    (env prefix)
#   bash vllm-glm-5.3-flash-nvfp4-orig.sh MAX_MODEL_LEN=150000    (as argument)
#   MAX_MODEL_LEN=150000 ./vllm-glm-5.3-flash-nvfp4-orig.sh       (direct exec)
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
# Same pinned build as the offload launcher (see its header).
: "${IMAGE:=cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1}"
: "${IMAGE_ID:=sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5}"

# Model: local checkpoint dir (mounted read-only) when present+readable, else
# HF_FALLBACK_ID is passed and vLLM downloads the checkpoint (~198 GB) into
# the HuggingFace cache on first boot. NOTE: use the RedHatAI checkpoint
# (compressed-tensors) — the LibertAIDAI modelopt checkpoint emits corrupted
# tokens on sm_120 (invalid UTF-8 / U+FFFD, occasional degenerate loops —
# open bug vllm-project/vllm#54150).
: "${HF_LOCAL:=/mnt/huggingface/RedHatAI/GLM-5.3-Flash-NVFP4}"
: "${HF_FALLBACK_ID:=RedHatAI/GLM-5.3-Flash-NVFP4}"

# Shared kernel JIT cache (Triton/deep_gemm/tilelang — NOT the HuggingFace
# model cache). Falls back to <script-dir>/.cache when not usable.
: "${JIT_CACHE:=/mnt/data/shared/models/vllm-moet-cache}"

# Server sizing / behavior:
: "${MAX_MODEL_LEN:=200000}"
: "${MAX_NUM_SEQS:=4}"

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

# switch guard: stop/remove any previous instance (also used when switching
# to/from the offload launcher, which uses the same container name)
docker container stop "$CONTAINER_NAME" 2>/dev/null || true
docker container rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run --restart=unless-stopped --gpus all --ipc=host --shm-size 64g -p "$PORT:$PORT" \
  --name "$CONTAINER_NAME" \
  --cap-add SYS_NICE \
  "${MODEL_MOUNTS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v "$F/patched-files/model_executor/layers/quantization/modelopt.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
  -v "$JIT_CACHE/jit":/root/.cache \
  -v "$JIT_CACHE/tilelang":/root/.tilelang \
  -v "$F/patched-files/model_executor/warmup/deepseek_v4_mhc_warmup.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py:ro \
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
  --kv-cache-memory 3300000000 \
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
  --jit-monitor-verbose
