# Patches — provenance and application notes

The shipped `patched-files/` were produced from the vllm package **extracted
from the docker image** (below), with the modifications applied below.

**Note: the committed kit ships only the resulting changed files**
(`patched-files/` + `manifest.txt`, 14 files — see below), not the full tree.
Equivalence was verified by diffing the full patched tree against the
image's stock package: exactly the 7 manifest files differ, everything else
is byte-identical to the image.

1. PR vllm-project/vllm#54743 — "[KV Offload] Scope offload group configs to
   prefix-cacheable KV cache groups" (**open/unmerged**).
   - Head commit: `899699c74ae2b8e8adc8726e5c9d0e355935076a`
     (fork `nood-co1`, branch `fix/offloading-config-scope-prefix-cacheable`)
   - Base commit: `504bb8b0c39dbde713a7344772bdb7005adbb214`
   - Why: the fork's offloading config crashes on hybrid models —
     `tokens_per_block` of non-prefix-cacheable scratch groups (DSA indexer
     tail_cache ring buffers, block_size=4) is not divisible by
     `tokens_per_hash`, triggering a hard assert. Related context:
     vllm-project/vllm#54831. The PR scopes offload group enumeration to
     prefix-cacheable groups only.
   - Files: `pr54743.diff` (as downloaded from
     `https://github.com/vllm-project/vllm/pull/54743.diff`),
     `pr54743.notests.diff` (same, with `tests/` paths filtered out — that is
     what we apply).
   - Application: `patch -p1 --forward < pr54743.notests.diff` from the tree
     root. **21 of 23 hunks apply cleanly; 2 rejects are expected** — both in
     the offloading `scheduler.py`, both targeting
     `_build_aligned_boundary_store_jobs`, a function that does not exist in
     the fork's older scheduler. Rejecting those hunks is correct and
     intentional; no dangling references are left behind.
   - ⚠ The PR is NOT merged upstream and its head commit is NOT on vllm
     master. It is preserved on branch `pr-54743-kv-offload-prefix-cacheable`
     of the fork `jdjohndoe13/vllm`:
     https://github.com/jdjohndoe13/vllm/tree/pr-54743-kv-offload-prefix-cacheable
     (SSH: `git@github-jdjohndoe13:jdjohndoe13/vllm.git`).

2. **Boot-fix (one line)** — the fork's `KVCacheSpec` classes predate
   upstream's `prefix_cacheable` attribute introduced/referenced by
   vllm-project/vllm#54743. The fork's equivalent property is
   `participates_in_prefix_caching` (returns False exactly for the
   non-prefix-cacheable scratch groups — the fork's own tail_cache
   solution). Without this the server dies at boot with:
   `AttributeError: 'UniformTypeKVCacheSpecs' object has no attribute 'prefix_cacheable'`
   (`.../offloading/config.py:62`).
   - Fix applied by `build-image-tree.sh`:
     ```
     sed -i 's/if group\.kv_cache_spec\.prefix_cacheable/if group.kv_cache_spec.participates_in_prefix_caching/' \
       distributed/kv_transfer/kv_connector/v1/offloading/config.py
     ```

3. **Overlay files** (2 of the 7 entries in `patched-files/` + `manifest.txt`;
   the no-offload launcher mounts these two too):
   - `patched-files/model_executor/layers/quantization/modelopt.py` →
     mounted over `model_executor/layers/quantization/modelopt.py`
     (SM120 NVFP4 serving fix from the cstechdev image's overlay set)
   - `patched-files/model_executor/warmup/deepseek_v4_mhc_warmup.py` →
     mounted over `model_executor/warmup/deepseek_v4_mhc_warmup.py`
     (mHC kernel warmup; removes per-request JIT compile warnings)

4. **CPU offload tier region hint** — `v1/kv_offload/cpu/shared_offload_region.py`
   (mounted over the same path, offload launcher only): issues
   `madvise(MADV_HUGEPAGE)` on the tier mmap before the populate pre-fault,
   so on kernels whose shmem supports THP the tier materializes as
   2 MiB folios. On the current host kernel (6.17.0-20-generic) shmem
   refuses 2 MiB folios in every `shmem_enabled` mode (verified
   2026-09-13 with a 7-variant test matrix), so this is a harmless no-op
   there — kept so the tier benefits automatically if the kernel gains
   shmem-THP support.

5. **MoE shared-experts state-leak fix** — `model_executor/layers/fused_moe/runner/moe_runner.py`
   (mounted over the same path, both launchers, added 2026-09-16):
   `_apply_quant_method` now wraps the shared-expert produce → consume
   sequence in `try/except BaseException` and clears
   `SharedExperts._output[_output_idx]` on the exception path.
   - Why: the slot written by `_maybe_apply_shared_experts` is consumed
     (and cleared) only by the `self._shared_experts.output` property at
     the end of the method. Any exception between produce and consume
     (routine during warmup / CUDA-graph capture / inductor recompile)
     leaves the slot dirty, and the next step's `SharedExperts.forward`
     hits `assert self._output[idx] is None` — permanently poisoning the
     layer. Observed on testcomp2 2026-09-15: first request after boot
     crashed all 8 TP ranks with that assert (same bug class as upstream
     vllm-project/vllm#46857).
   - Normal-path behavior is unchanged: the overlap (aux-stream) path
     stays fully enabled, so decode throughput is unaffected — this only
     unwinds state on the failure path.

## Image / fork provenance

- Image: [`cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`](https://hub.docker.com/r/cstechdev/vllm/tags?page=1&name=glm53-flash-nope-sm120-cu130-20260826-r1)
  on Docker Hub ([repo overview](https://hub.docker.com/r/cstechdev/vllm))
  (cu130 build, SM120 fixes baked in by the cstechdev fork)
  - pinned: `sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5`
    (config digest == registry manifest digest for this build; both
    launchers run this ID and refuse drifted tags)
- The image's vllm is built from the **cstechdev fork, commit `g487ecf187`**
  — NOT an upstream vllm commit. The SM120 overlay fixes (rope-free
  sparse-MLA + kpool; glm5next support per vllm-project/vllm#53906 lineage)
  are part of that fork; upstream vLLM could not run this model on sm_120 at
  image build time.
- The image build sources live at
  https://github.com/chriswritescode-dev/glm-5.3-flash-sm120 , with a
  preservation fork at
  https://github.com/jdjohndoe13/glm-5.3-flash-sm120-docker-image-sources .
- If you need to inspect the fork's history, the shipped files were
  extracted from the image exactly as patched — no need to rebuild anything.

## 2026-09-16 crash notes — kv_offload swap path (crashes #3/#4)

- Signature (both boots): `gpu_worker.py transfer_async` →
  `cuMemcpyBatchAsync failed at index N with error 1`
  (CUDA_ERROR_INVALID_VALUE), always on a GPU->CPU
  preemption/eviction-flush store job under cluster KV pressure. #3: index
  34 inside a single 560-entry batch (uniform sizes). #4 (boot
  054426, during the 3-concurrent-~150k-token storm): index 7 INSIDE a
  <=32-entry chunk — batch size alone is not the discriminator; <=7-entry
  evictions and 560-entry CPU->GPU loads always pass. No dump_input was
  produced; no OOM or MoE asserts involved.
- Analysis: per-item driver rejection inside the batch-memcpy op of the
  deployed `_C_stable_libtorch.abi3.so` (class: vllm #39491 / #49276 on
  Blackwell GeForce builds, driver 590.48.01). Upstream added the
  `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS` env knob in a LATER build than
  ours (verified: the env string is absent from our binary), so chunking
  is owned python-side.
- Mitigation + instrumentation now shipped inside
  `v1/kv_offload/cpu/gpu_worker.py` (md5 `ae6ddad71493985c297ca23ba4e24cd7`,
  elect for `_SwapDiag`):
  every submission capped to `VLLM_KV_OFFLOAD_SWAP_BATCH_CAP` descriptors
  (default 32, `=1` = per-entry everywhere); on rejection the failed chunk
  is probed item-by-item with the same op at n=1, the event is logged to
  `/tmp/vllm_swap_diag/log_swap_T<pid>.jsonl` (env
  `VLLM_KV_OFFLOAD_DIAG_DIR`), and the failed span is completed per-entry
  via `libcuda.so.1 cuMemcpyAsync` on the current stream; afterwards the
  handler goes sticky-per-entry so flushes complete and the engine survives
  while diagnostics accumulate. If a per-entry copy itself fails, the exact
  pointer/size pair is logged and the error re-raised (data-level defect
territory). Handler geometry is logged at init (`handler_init`) so a
   failing item's pointer decodes to block ids offline.

## 2026-09-16 crash notes — crash #5 + driver 615.71.09 (offload stack)

- Crash #5 (boot 072727, after the driver upgrade to 615.71.09) reproduced
  the SAME signature as #3/#4: `cuMemcpyBatchAsync failed at index N with
  error 1` on a GPU->CPU flush store, n=78, first 32-chunk, item 6. The
  pointer values are in the engine log; the JSONL decode data was lost
  again to the tmpfs wipe on reboot.
- CONFIRMED root cause: the defect is item-level data, NOT the driver.
  Upgrading 590.48.01 → 615.71.09 changed nothing (same crash, same code
  path). The poison descriptor MOVES with the job (boot 062647: item
  56/78 size 542720; boot 072727: item 6/78 size 671744) — it is not a
  fixed poison entry, and the eviction path (<=7 entries) never fails.
  The failing triple is a store into the registered CPU tier: a plain
  per-entry `cuMemcpyAsync` on the SAME triple also returns rc=1
  (CUDA_ERROR_INVALID_VALUE), and the batch op at n=1 on the same triple
  fails at index 0 — the driver rejects the descriptor itself, not the
  batch. Chunked `cudaHostRegister` covers the full tier (no tail gap),
  so the poison is a genuinely bad pointer/size the spec pipeline emits
  on large preemption-flush stores, not an unpinned tail.
- The stock `-orig` (docker, no-offload) death in the same window was a
  DIFFERENT failure: the crashed no-docker kit leaked its GPU contexts
  (8x30.5 GiB held after the process died), so the docker engine could not
  allocate GPU memory and died; only the reboot cleared it. Not the
  offload bug.
- Mitigation build deployed (md5 `b4f5aedd267c54580f66bd7e8d846073`):
  * diag dir moved to PERSISTENT storage
    (`/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/swap_diag`, env
    override still wins) and `handler_init` geometry is mirrored to the
    engine log — the decode data now survives reboots.
  * on any per-entry failure the poison descriptor is CLASSIFIED against
    the handler geometry (`in_tier`, `tier_off`, `tier_tail`, `fits`, and
    `cuPointerGetAttribute` MEMORY_TYPE for src and dst) and logged to
    the engine log — the root-cause decode that survives the tmpfs wipe.
  * the per-entry fallback now RETRIES the failed item with the explicit
    direction API (`cuMemcpyDtoHAsync_v2` / `cuMemcpyHtoDAsync_v2`)
    before re-raising. If the explicit call succeeds, the bug is the
    driver's default-direction inference on that pointer and the store
    path can switch to explicit-direction as a clean fix; if it fails
    too, the pointer is genuinely invalid to the driver and the producer
    (spec/`_fill_group_ops` math) must be fixed.
  * re-raise on hard failure is retained (the consumer asserts `success`
    on every store — `offloading/worker.py:229,277` — so there is no safe
    "fail the job" path; the engine must not silently skip a store).
- Next decode needs a kit boot that hits the poison (storm test now at
  persistent `/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/
  test_preempt_flush.py`). The engine-log lines
  `swap_diag per_entry_failed ... class={...}` (or
  `per_entry_explicit_ok`) are the decisive evidence.

## 2026-09-16 crash #6 + instrumentation bug (boot 084156)

- Crash #6 (storm on the new build): same poison — n=78, off=32,
  item 56, size 542720 — AND the explicit `cuMemcpyDtoHAsync_v2` retry
  ALSO failed (rc=1). So the driver rejects the triple even with the
  direction stated explicitly; the `in_tier: False` classification was
  NOT trustworthy (see below).
- INSTRUMENTATION BUG FOUND: `_SWAP_DIAG.register(self)` was called
  BEFORE `dst_blocks_per_chunk` was assigned (attribute defined later in
  `__init__`) — the AttributeError was swallowed by the bare except, so
  ZERO `handler_init` events ever landed and `_classify()` had no
  geometry: every item reported `in_tier: False` as a false negative.
  The real question (is the poison dst inside the registered tier?) is
  still open — boot 084156 gave no usable decode data.
- Fixed build (md5 `339679d7d570e8fbcbeecab8ca203ebb`):
  * `register()` moved AFTER the `*_blocks_per_chunk` assignments and
    now re-raises instead of swallowing (a broken register can never
    silently disable classification again).
  * new `span_dump` event: on every batch rejection, EVERY descriptor in
    the rejected chunk is classified (in_tier? tier offset? fits?) and
    logged — the producer's pattern across the whole chunk, not just the
    poison item.
  * crash-shape note: the storm reproduces the poison at the SAME
    position (chunk 2, item 56/78, size 542720) on every boot — the
    producer is deterministic; the pointer differs per boot (fresh mmap
    base).

## 2026-09-16 crash #7 + ROOT CAUSE FOUND (boot 091048, driver 615.71.09)

- Storm flush with full-span instrumentation: `span_dump off=0 cnt=32
  poison=7`; poison dst 0x7ad081371800 inside the registered tier
  (in-region, mapped, 256B-aligned, src in GPU tensor 1). A stride-aware
  decode (tools/decode_swap_diag.py) + oracle analysis pinned the cause.
- ROOT CAUSE: the poison descriptor was a LEGAL one whose write span
  STRADDLED an internal `cudaHostRegister` chunk seam. Geometry:
  dst = material row 1092, TP2 slot, MLA layer-1 cell =
  region_base + 68,719,099,904; the 64 GiB chunk-1/chunk-2 seam sits at
  region_base + 68,719,476,736; the 671,744-byte write ends
  region_base + 68,719,771,648 = 294,912 bytes PAST the seam (and starts
  376,832 bytes before it). cuMemcpyBatchAsync AND plain cuMemcpyAsync
  AND explicit cuMemcpyDtoHAsync_v2 ALL reject a copy that crosses two
  registered ranges (CUDA_ERROR_INVALID_VALUE) — verified on drivers
  590.48 and 615.71; the ~615.71 upstream fix lifted only the 512-GiB
  page-table cap, not the cross-range rejection. Descriptors are always
  contained in one material row (row pitch = kv_bytes_per_block =
  62,914,560; row-interleaved slots via _worker_offset), so whether an
  descriptor crosses a seam is deterministic per boot (row/layer
  composition): 671,744-byte items can straddle, 33,792-byte items'
  window is 34x narrower and never caught.
- FIX (md5 `952e03e37e6e15aee9f2828c89c9e45d`, gpu_worker.py only):
  * `pin_mmap_region`: pin chunks are row-aligned
    (chunk = (chunk // _row_stride) * _row_stride, guarded to stay
    >= 64 MiB and a 2 MiB multiple) — registration seams land on material
    row starts where no descriptor can cross them (13 chunks still).
  * `_classify`/`register`: stride-aware extents + row/cell decode
    (dst_extent = (rows-1)*stride0 + row_bytes); kills the earlier
    in_tier:false false-negatives from GPU-worker.py assumption of
    contiguous tier views.
  * `_probe_registration`: CU_POINTER_ATTRIBUTE_RANGE_START_ADDR on the
    first/last byte of a failing host span -> split_ranges/boundary/
    bytes_past_boundary evidence (event `probe_registration`, both in
    _recover (pristine) and per-entry failure paths).
  * `_split_retry`: if a failing span straddles a seam even after the
    pin-chunk fix (e.g. non-row-aligned tier or forced chunk override),
    the copy is split AT the seam into two cuda copies on the same stream
    and completed (events `split_retry_begin` / `split_retry` /
    `per_entry_split_recovered`), stream-synced; engine SURVIVES.
    `VLLM_KV_OFFLOAD_DIAG_SPLIT_PROBE=0` disables the split for A/B
    evidence runs.
- Crash-5's poisoned dst (137,532,389,486,592, boot 072727) is the same
  class: re-anchor via that boot's JSONL handler_init region base + the
  mod formula above if needed.
- BEHAVIORALLY CONFIRMED (2026-09-16, boot engine-20260916-104501):
  first green storm ever via the blocking autotest
  (autotest_kit.sh: stop-all -> boot kit -> fingerprint gate -> watchdog
  -> storm -> verdict; watchman restores the dockerized -orig container
  on any crash). 8/8 ranks row-aligned (13 chunks of 68.7 GB = 1,092
  rows), 16 handler_init JSONL events, ZERO batch_rejected /
  probe_registration / per_entry_failed / split_retry, storm ALL DONE
  green, PH2-restore 0.79 s, health 200 retained -> the cross-range
  rejection class is eliminated at the source (rescue path never
  triggered). Engine left serving.
- TIER->GPU RESTORE SPEED (2026-09-16, measured DURING the soak run on
  the fixed build):
  * instrumented aggregate: ~34 GB/s. Prometheus counters (two
    snapshots, both under soak churn): load_size_count=64 and
    load_size_sum=7.32e10 bytes with load_time_total=2.14 s -> 34.1 GB/s
    average per load op (~1.14 GB each, ~33.5 ms); earlier snapshot
    (48 ops, 5.22e10 bytes, 1.55 s) gives 33.7 GB/s.
  * end-to-end single-session restore: ~12.9 GB/s (storm PH2-restore:
    KV-LOOKUP hit=162 blocks = 165,888 tokens = 10.19 GB in 0.79 s wall;
    wall includes scheduling + prefix hashing + first token).
  * KV bytes/token: 61,441 B = 858.97e9-tier / 13,653 blocks / 1024
    tokens per block (block_size="1024" per cache_config_info).
   * soak probes for reference: iter-0 M1G growth restore 1.95 s, iter-1
   M1 GPU-hit 0.56 s (the 21.45 s iter-0 M1 is cold prefill compute,
   NOT a restore).

## 2026-09-17 — log-noise demotion + MoE count-kernel warmup

1. `offloading/scheduler.py` — the three `KV-LOOKUP abort` log lines demoted
   INFO→debug. These are the per-request offload-tier MISS signal (the
   connector "abandons" the tier lookup and the request proceeds on the
   GPU radix cache; nothing is aborted/dropped). At production volume they
   were 23,707 of 70,066 log lines for boot 121436. `KV-LOOKUP hit` and
   `KV-LOOKUP defer` remain INFO.
2. `model_executor/warmup/deepseek_v4_mhc_warmup.py` — folded in
   `_warmup_moe_expert_count_kernels(model)`, called right after the
   model-type gate inside `deepseek_v4_mhc_warmup`, wrapped in
   try/except (never blocks startup). It pre-compiles Triton
   `fused_moe._count_expert_num_tokens` specs at boot — BLOCK_SIZE
   buckets 128/256/512/1024 × divisibility variant (numel % 16 == 0 or
   not — the tt.divisibility hint set differs) × expert_map None/present
   = 10 launches on id-shape (1,numel) int32 zeros, grid 16. Rationale:
   the kernel is only reached on the eager fused-MoE path (outside
   cudagraph capture sizes), so boot dummies never compile it and the
   first real chunked prefill pays the JIT latency spike per boot
   (observed 15:17 on all 8 ranks of boot 121436; jit_monitor hint:
   "consider extending warmup to cover this shape/config"). No launcher
   or manifest change: the file is already overlaid (manifest entry 6)
   and `kernel_warmup()` calls `deepseek_v4_mhc_warmup` unconditionally
   ahead of the `enable_jit_warmup` gate.
   Effective at next engine boot; verified by
   `Warmup: fused_moe _count_expert_num_tokens Triton specs compiled.`
   in the boot log and by the absence of the 15:17-style JIT warnings.

## 2026-09-18 — KV-offload stale-hit re-lookup livelock fix + log demotion

- Bug caught on boot `engine-20260917-223611` during the 40-iteration
  marker-revisit soak (2026-09-18 00:42–01:32 UTC): a ~166k-token request
  hung with NO first Token for 600 s+ while the scheduler repeated the
  identical `APC-HIT final=0` + `KV-LOOKUP hit` pair every step (~3 Hz;
  431k APC-HIT lines total). Soak showed the milder form first: marker
  restore slope 25→56→125→282 s across late iterations. One request kept
  scanning ~3 min past client abort, then self-cleared (zombie at 01:28:35,
  frozen forever after — a photosensor for the same loop, not a leak).
- Root cause (oracle adjudication 2026-09-18): the CPU-tier lookup
  TRUTHFULLY hits (fresh, ready entries kept alive by churn re-stores —
  not stale `is_ready`; H1 transfer-job stalls, H2.1
  `_chunks_being_loaded` leaks, and H2.2 pseudo-hits were all ruled out),
  but converting the full external hit requires an ALL-OR-NOTHING upfront
  GPU allocation for the whole ~165k-token span; under tier-full churn
  (12,412/12,412 blocks, ~1.4k blocks/iteration of eviction pressure,
  mirrored_keys growing 16k→37k) that conversion never lands, and the
  connector re-reports the identical hit every scheduler step forever.
  The connector's third-party Deferred lane parked while the loop burned
  at full rate; swap throughput stayed flat (~33 GB/s), confirming this
  is purely a scheduling-side livelock.
- Fix (`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
  one guard, no new locks, no unbounded logging):
  `get_num_new_matched_tokens` counts CONSECUTIVE identical no-progress
  repeats (same hit token count + same `num_computed_tokens`) within
  2.0 s wall clock. Legitimate async waits (`transfer_jobs`) and
  defer/RETRY outcomes exit above the counter, so only genuinely stuck
  repeats accrue. Past the budget the connector returns `(0, False)` —
  the scheduler then takes the incremental recompute path, which always
  makes progress regardless of who blocked the upfront conversion.
- Knob: `VLLM_KV_OFFLOAD_STALE_HIT_MAX_LOOKUPS` — default 8; `0` disables
  (legacy livelock behavior preserved for A/B); unparsable values fall
  back to 8 with a log warning. Post-restart consequences: bounded
  first-token latency under tier-full churn (≤ ~2.7 s of scan then
  recompute), zombie abort-scan self-cleans within ~2 s of abort, and the
  M1 marker-restore slowness ramp collapses (previously the unconverted
  scan amplified starvation for minutes at a stretch).
- Log demotion (2 files, diagnostic only — metrics keep all data):
  - `v1/core/kv_cache_coordinator.py`: `APC-HIT …` logger is INFO only
    when the final hit is real (final>0), otherwise DEBUG — this was the
    431k-line INFO flood under churn (per-line unchanged in content).
  - `v1/kv_offload/cpu/manager.py`: `CPU-TIER-EVICT …` INFO→DEBUG (the
    eviction rate stays observable via the `CPU_EVICTED_TOTAL` metric).
- Files + md5 (runtime tree at
  `/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/vllm-bin/dist-packages/vllm`,
  mirrored into `patched-files/` + `deployed-sources/`):
  - `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
    `0b9ee2989447ffb2acb1a9bbdd51c701`
    (env knob init at ~:608 + 5 stale-hit fields on the slots dataclass
    at ~:361 + tail fallback guard at ~:1190; fallback log DEBUG→INFO and
    KV-LOOKUP repeat-flood dedup logging, 2026-09-18b/c)
  - `v1/core/kv_cache_coordinator.py`
    `335c0663851b336fc9ed58b6c2c9bc5f` (change-detect APC-HIT dedup at
    ~:854, 2026-09-18c; supersedes the conditional-INFO version
    `e6582917f8f586e77298d1523005fc6d`)
  - `v1/kv_offload/cpu/manager.py`
    `39800ec45f1cc82fb115c87eee652c14` (CPU-TIER-EVICT → DEBUG at ~:288)
  - Pre-patch runtime backups on testcomp2:
    `*.bak-20260918-livelock` (`dffa153df8e7d91d61fd002c72131d73`,
    `c2f421d3772589d1cbbe6ccb5a0e1550`, `931044c6a4dc433fa1a87d54f3aceeba`).
- Verified: `py_compile` all 3 files on the runtime tree; remote visual
  anchor review (6 anchor points); local mirror hashes == runtime post
  patch hashes everywhere. Effective at the NEXT ENGINE RESTART (the
  running engine keeps the old code; restart-owned by the operator).
- Post-restart verification plan (T+boot, marker-revisit storm):
  KV-LOOKUP hit repeats per request ≤ ~8 (was ~2,180); APC-HIT INFO lines
  two-digit scale (was 431k); first token ≤ storm p95 (was 600 s hang);
  tombstone counting unchanged (attribution path untouched); third-party
  Deferred lane untouched (kept out of scope by design); canary baseline
  reset on the new bootlog.
- 2026-09-18b refinements (runtime + kit + mirrors in lock-step, md5 row
  updated above; effective at next engine restart):
  * fallback log `KV-LOOKUP stale-hit fallback` DEBUG→INFO so the
    bounded-rescue oscillation is auditable at default log level (its DEBUG
    lines are invisible in the INFO bootlog — the 06:34:40–48 pool-jam wait
    showed ~461 repeats ending in a 200 after ~8 s, vs the pre-patch 600 s
    hang; the ~9-call repeat staircase per fallback cycle is BY DESIGN).
  * Post-restart probe: M1 clone restored in 0 s wall / 200 / single
    KV-LOOKUP line / zero repeats (tier-warm fast path healthy).
  * Storm runner v1 crashed on a `grep -c || echo 0` arithmetic pitfall
    (EVI=0 count reads "0\n0"); v2 removes the pitfall and raises the
    per-rid repeat hard-alarm to disease-scale (>6000 repeats).
  * Storm v2 (post-A3) iteration 0-8 (06:53:50–07:12:25, all 200s): M1
    restore times STABLE 0.7/36/9.2/27/20/21/5.2/24/40 s — NO ramp vs the
    pre-patch 25→56→125→282 s escalation; tier filled to 12,412 tokens,
    evicted_total ~4.3k, tombstones ~1.2k; 0×500, 0×OOM. The 461-repeat
    residual waits end in 200s (bounded ~8 s pool-jam waits, fallback
    firing as designed — invisible-then; now INFO-auditable).
  * 07:10 "no GPU computing" incident decoded: user's ~113k request
    queued FCFS behind the storm's iter-8 monster prefills (X8=77.7 s +
    M1G=132 s full 165k pool-churn passes); 64,299 repeat lines (~530/s,
    ~30 MB log bloat) ended at the user's own 07:12:25 shutdown. Starved
    queue + stepwise-rescan NOISE under monopoly, not the old livelock.
  * A5 repeat-flood dedup (md5 `0b9ee2989447ffb2acb1a9bbdd51c701`,
    runtime + kit + both local mirrors, py_compile OK, staged at 07:38):
    new slot field `stale_hit_repeat_count`; repeat KV-LOOKUPs (same hit
    + same num_computed_tokens) log DEBUG except every 64th repeat at
    INFO with `repeats=N`; first-hit INFO and the A3 fallback gate are
    unchanged. ~530 repeat-lines/s → ~8 lines/min; 30 MB → ~3 KB per
    jam. canary5m v2 + storm v3 alert on `repeats=` > 5000 (per-rid INFO
    counting is blind once dedup is active).
  * A6 APC-HIT change-detect dedup (md5 `335c0663…`, runtime + kit +
    local `patched-files` mirror, py_compile OK; requested by user after
    they flagged the per-step lines): the 2026-09-14 diagnostic
    (`hit_length > 0` → INFO at ~:854, no request ID in scope) now
    demotes identical-signature repeats — same nhash/max/final/
    per_group/uncached — to DEBUG; INFO only on first occurrence or a
    value change; `hit_length == 0` stays DEBUG (B1 semantics intact).
    Returns the last per-step INFO flood to log-on-change (07:10 jam
    class: identical `final=12288` lines at ~530/s). State lives on a
    coordinator-instance getattr field; no `__init__` patch.
