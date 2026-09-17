# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
)


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.append(max_auto_tokens)
    return _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ not in (
            "DeepseekV4DecoderLayer",
            "Glm5NextDecoderLayer",
        ):
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_post",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    # Upstream a8887c208 ("[DSV4] aiter mhc support (ROCm)") refactored
    # ``hc_head`` from a free function into the ``HCHeadOp`` CustomOp
    # instance attached to the model as ``hc_head_op``. We call through
    # that instance so the warmup exercises the same dispatched
    # implementation as the inference path.
    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        return

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


def _warmup_layer_mhc_glm(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    """Warm GLM-5 Next (mhc=True) mHC kernels.

    Differs from the DSv4 path in three ways:
      * hc_pre returns (post_mix, res_mix, layer_input); DSv4 returns
        (layer_input, post_mix, comb_mix).
      * serving passes norm_weight/norm_eps into hc_pre, selecting the
        ``*_with_norm`` TileLang kernel variants; warm those too.
      * the inter-layer fused path ``hc_fused_post_pre`` (mhc_fused_tilelang)
        dominates serving; exercise it explicitly.
    """
    max_tokens = max(token_sizes)
    n = int(layer.n)
    hidden_size = int(layer.hidden_size)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        n,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    norm_weight = layer.input_layernorm.weight.data
    norm_eps = float(layer.input_layernorm.variance_epsilon)

    # The serving wrapper derives the TileLang GEMM split factor from the
    # batch size at 64-token granularity:
    #   n_splits = n_sms // ceil(num_tokens/64)  (capped by k-dim blocks)
    # so every distinct ceil(num_tokens/64) bucket maps to a distinct
    # cache_key. The default power-of-2 ladder only reaches ~6 of the ~20
    # buckets reachable with max_num_batched_tokens=2048, leaving real
    # traffic to trip first-use warnings (and their latency spikes) per
    # boot. Probe one token count per 64-token bucket so no serving-time
    # token count can land on a cold key.
    bucket_sizes = sorted(
        set(token_sizes) | {64 * c for c in range(1, 33) if 64 * c <= max_tokens}
    )
    for size in bucket_sizes:
        res_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            post_mix, comb_mix, layer_input = layer.hc_pre(
                res_slice,
                fn,
                scale,
                base,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
            layer.hc_post(layer_input, res_slice, post_mix, comb_mix)
            layer.hc_fused_post_pre(
                layer_input,
                res_slice,
                post_mix,
                comb_mix,
                fn,
                scale,
                base,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )


def _warmup_moe_expert_count_kernels(model: torch.nn.Module) -> None:
    """Pre-compile fused_moe's Triton _count_expert_num_tokens specs.

    The counting kernel is only reached on the eager fused-MoE path (outside
    the CUDA-graph capture sizes), so boot-time dummy runs never compile it
    and the first real chunked prefill pays a JIT spike per boot (observed
    2026-09-17 15:17 synchronously on all 8 ranks). Warm every BLOCK_SIZE
    bucket the wrapper can select (next_power_of_2(min(numel, 1024))), in
    both divisibility variants (tt.divisibility hint depends on numel % 16),
    with and without an expert map, so no serving-time token count can land
    on a cold key. num_experts=16 (divisible by 16) keeps the div-16 layout
    hint set of the real invocation; the grid size does not enter the
    Triton compile key otherwise.
    """
    from vllm.model_executor.layers.fused_moe.utils import (
        count_expert_num_tokens,
    )

    device = next(model.parameters()).device
    num_experts = 16
    ids_full = torch.zeros((1, 1024), dtype=torch.int32, device=device)
    expert_map = torch.arange(num_experts, dtype=torch.int32, device=device)
    for bucket in (128, 256, 512, 1024):
        for numel in (bucket, bucket - 8):
            ids = ids_full[:, :numel]
            count_expert_num_tokens(ids, num_experts, None)
            count_expert_num_tokens(ids, num_experts, expert_map)
    logger.info(
        "Warmup: fused_moe _count_expert_num_tokens Triton specs compiled."
    )


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    # Cheap model-type gate before walking ``model.modules()``. The class
    # walk below is O(num_layers) and shows up in startup time on very
    # large checkpoints; bail out for any model that is not DeepSeek V4.
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type not in ("deepseek_v4", "glm5_next", "glm5_next_text"):
        return

    try:
        _warmup_moe_expert_count_kernels(model)
    except Exception:
        logger.exception(
            "MoE count-expert Triton warmup failed; continuing startup "
            "(first chunked prefill may pay JIT cost)."
        )

    layer = _find_first_mhc_layer(model)
    if layer is None:
        return

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    is_glm = model_type in ("glm5_next", "glm5_next_text")
    deepseek_model = None if is_glm else _find_deepseek_v4_model(model)
    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",
        token_sizes,
    )
    with torch.inference_mode():
        if is_glm:
            try:
                _warmup_layer_mhc_glm(layer, token_sizes)
            except Exception:
                logger.exception(
                    "GLM mHC warmup failed; continuing startup "
                    "(first requests may pay JIT cost)."
                )
        else:
            _warmup_layer_mhc(layer, token_sizes)
        if deepseek_model is not None:
            _warmup_hc_head(deepseek_model, token_sizes)
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
