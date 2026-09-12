#!/usr/bin/env bash
# ============================================================================
# GLM-5.3-Flash-NVFP4 on vLLM — 8x RTX 5090 (sm_120) — FALLBACK: no KV offload
#
# Same server as vllm-glm-5.3-flash-nvfp4.sh but WITHOUT KV CPU offloading
# (uses the image's stock vllm package + two overlay files instead of the
# patched tree). Use this if you want to compare behavior or if the offload
# tree ever misbehaves. Same container name + port: stop the other one first
# (both launchers do this guard themselves).
#
# NOTE on the two file overlays (shipped under patched-files/, mirroring the
# package tree — the same files the offload launcher mounts, see manifest):
#   * model_executor/layers/quantization/modelopt.py   (SM120 NVFP4 fix)
#   * model_executor/warmup/deepseek_v4_mhc_warmup.py  (mHC kernel warmup)
# ============================================================================
set -euo pipefail
F="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Overlay image with the SM120 rope-free sparse-MLA + kpool fixes
# (upstream vLLM cannot run this model on sm_120 yet; see
#  vllm-project/vllm issues #53963, #54150 and PR #53969):
IMAGE="cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1"
# PINNED build — same hash as the offload launcher (see its header):
#   config digest == registry manifest digest for this build
IMAGE_ID="sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5"

if ! docker image inspect "$IMAGE_ID" >/dev/null 2>&1; then
  echo "ERROR: pinned image $IMAGE_ID is not present locally." >&2
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "The tag '$IMAGE' exists but is NOT the pinned build:" >&2
    docker image inspect "$IMAGE" --format '  tag points to: {{.Id}}' >&2
    echo "Refusing to run a possibly-drifted image. Obtain the pinned one:" >&2
    echo "  docker load < glm-image.tar                     (backup-image.sh tarball)" >&2
    echo "  or docker pull cstechdev/vllm@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5" >&2
  else
    echo "Obtain it first:" >&2
    echo "  docker load < glm-image.tar                     (backup-image.sh tarball)" >&2
    echo "  or docker pull cstechdev/vllm@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5" >&2
  fi
  exit 1
fi

# IMPORTANT — checkpoint choice on SM120:
#   Use RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-tensors).
#   LibertAIDAI/GLM-5.3-Flash-NVFP4 (modelopt) emits corrupted tokens
#   on SM120 (invalid UTF-8 / U+FFFD, occasional degenerate loops) —
#   open bug vllm-project/vllm#54150.
MODEL_ID="/mnt/huggingface/RedHatAI/GLM-5.3-Flash-NVFP4"
#MODEL_ID="/mnt/huggingface/LibertAIDAI/GLM-5.3-Flash-NVFP4"

MAX_MODEL_LEN=200000
MAX_NUM_SEQS=4
NAME=vllm-glm-5.3-flash-nvfp4
CACHE=/mnt/data/shared/models/vllm-moet-cache
mkdir -p "$CACHE/jit" "$CACHE/tilelang"

docker container stop "$NAME" 2>/dev/null || true
docker container rm -f "$NAME" 2>/dev/null || true

docker run --restart=unless-stopped --gpus all --ipc=host --shm-size 64g -p 1025:1025 \
  --name "$NAME" \
  --cap-add SYS_NICE \
  -v /mnt/huggingface:/mnt/huggingface:ro \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v "$F/patched-files/model_executor/layers/quantization/modelopt.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
  -v "$CACHE/jit":/root/.cache \
  -v "$CACHE/tilelang":/root/.tilelang \
  -v "$F/patched-files/model_executor/warmup/deepseek_v4_mhc_warmup.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py:ro \
  -e DG_JIT_CACHE_DIR=/root/.cache/deep_gemm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  "$IMAGE_ID" "$MODEL_ID" \
  --served-model-name glm-5.3-flash qwen-3.8-flash-next \
  --host 0.0.0.0 --port 1025 \
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
  --enable-chunked-prefill \
  --optimization-level 3 \
  --async-scheduling \
  --jit-monitor-verbose
