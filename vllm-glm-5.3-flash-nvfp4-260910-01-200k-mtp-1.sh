#!/usr/bin/env bash

# Go to script folder
cd "$(dirname "${BASH_SOURCE[0]}")"

# GLM-5.3-Flash NVFP4 on vLLM — 8x RTX 5090 (sm_120, 32 GB each, TP=8)
#
# Overlay image with the SM120 rope-free sparse-MLA + kpool fixes
# (upstream vLLM cannot run this model on sm_120 yet; see
#  vllm-project/vllm issues #53963, #54150 and PR #53969):
IMAGE="cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1"

# IMPORTANT — checkpoint choice on SM120:
#   Use RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-tensors).
#   LibertAIDAI/GLM-5.3-Flash-NVFP4 (modelopt) emits corrupted tokens
#   on SM120 (invalid UTF-8 / U+FFFD, occasional degenerate loops) —
#   open bug vllm-project/vllm#54150.
MODEL_ID="/mnt/huggingface/RedHatAI/GLM-5.3-Flash-NVFP4"
#MODEL_ID="/mnt/huggingface/LibertAIDAI/GLM-5.3-Flash-NVFP4"

# Pre-download weights (or set HF_TOKEN for gated access); served read-only from cache.
#hf download "$MODEL_ID"

# NOTE: the verified config in #53963 is 4x RTX PRO 6000 96 GB (TP4).
# TP=8 on 8x RTX 5090 (256 GB total vs ~181 GiB of weights) is not a
# community-verified configuration — expect tight memory. If the mamba/
# KDA state cache or KV cache is too small, lower MAX_MODEL_LEN or
# MAX_NUM_SEQS first (the linear-attention state cache is per-request).
MAX_MODEL_LEN=200000
MAX_NUM_SEQS=2

CACHE=/mnt/data/shared/models/vllm-moet-cache
mkdir -p "$CACHE/jit" "$CACHE/tilelang"

docker run --rm --gpus all --ipc=host --shm-size 64g -p 1025:1025 \
  --cap-add SYS_NICE \
  -v /mnt/huggingface:/mnt/huggingface:ro \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v ./patched-files/model_executor/layers/quantization/modelopt.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
  -v "$CACHE/jit":/root/.cache \
  -v "$CACHE/tilelang":/root/.tilelang \
  -e DG_JIT_CACHE_DIR=/root/.cache/deep_gemm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  "$IMAGE" "$MODEL_ID" \
  --served-model-name glm-5.3-flash qwen-3.8-flash-next \
  --host 0.0.0.0 --port 1025 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.945 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4]}' \
  --kernel-config '{"enable_jit_warmup":true,"enable_cutedsl_warmup":true}' \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser deepseek_r1 \
  --block-size 256 \
  --max-num-batched-tokens 4096 \
  --kv-cache-metrics \
  --enable-mfu-metrics \
  --enable-chunked-prefill \
  --optimization-level 3 \
  --async-scheduling \
