# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"
    # CPU-TIER-EVICT diagnostics 2026-09-14: true CPU tier pool state +
    # cumulative eviction counter (emitted in manager.get_stats; gauges are
    # snapshots; evicted_total is cumulative-since-boot).
    CPU_ALLOCATED_BLOCKS = "vllm:kv_offload_cpu_allocated"
    CPU_FREE_LIST_LEN = "vllm:kv_offload_cpu_free_list_len"
    CPU_EVICTABLE_LEN = "vllm:kv_offload_cpu_evictable_len"
    CPU_EVICTED_TOTAL = "vllm:kv_offload_cpu_evicted_total"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """
