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
    `6e601eee63a8a1f887898a188bb2edfb`
    (env knob init at ~:608 + 5 stale-hit fields on the slots dataclass
    at ~:361 + tail fallback guard at ~:1190; fallback log DEBUG→INFO,
    KV-LOOKUP repeat-flood dedup, and fail-safe try/except on the dedup
    state, 2026-09-18b/c/d; supersedes `0b9ee2989447ffb2acb1a9bbdd51c701`)
  - `v1/core/kv_cache_coordinator.py`
    `6c84290877580d539f078a861568b9b2` (change-detect APC-HIT dedup at
    ~:854 + fail-safe try/except, 2026-09-18c/d; supersedes the crash
    `ea1199b87c7806f1fd0179b3339b127c`, `335c0663851b336fc9ed58b6c2c9bc5f`,
    and the conditional-INFO `e6582917f8f586e77298d1523005fc6d`)
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
    * CRASH + HOTFIX (2026-09-18 14:32:31, boot `engine-20260918-142448`):
      A6 v1 crashed EngineCore on the FIRST prefix-hit request —
      `AttributeError: 'list' object has no attribute 'values'` at
      kv_cache_coordinator.py:855 — because `hit_length_by_group` is
      `list[int]` (init at ~:748), not a dict; the log format
      (`%s`-printed) rendered it as a list and the dedup signature
      wrongly assumed dict. Fix: `tuple(hit_length_by_group)` (already
      order-stable + hashable). Verified: py_compile OK; setattr probe
      confirms `HybridKVCacheCoordinator` (the live class, the one that
      crashed) accepts the lazy `_last_apc_sig` (ABC `__slots__=()`
      inherited from ABC ≠ slot-blocked; classes define no own
      `__slots__`). Post-fix md5 `ea1199b87c7806f1fd0179b3339b127c` in
      runtime + kit + local mirror. Lesson: the A6 v1 signature
      assumed a dict from a `.values()` call that never existed —
      the pre-crash log evidence (per_group=[...] brackets) was the
      tell; py_compile cannot catch attribute-type errors.
  * Long-storm certification on the fixed boot (2026-09-18 15:25–15:49,
    boot `engine-20260918-150710`, storm pid 431076, launch note: the
    v2-era hardcoded bootlog pin in soak-verify.sh was replaced by
    newest-by-mtime selection after it pinned a dead boot for one
    aborted launch): 16/16 iterations 200 OK, 0×500, 0×Traceback;
    M1 restores STABLE 0.64–0.71 s across all 16, including after the
    tier went full (iter 12+) with evicted_total 28→1968 and
    tombstones 7→539 flowing — no ramp (old disease: 25→56→125→282 s);
    whole-boot totals KV-LOOKUP 41 INFO lines / APC-HIT 9 lines
    (A5+A6 dedup holding vs 69,204 + 80,386 on diseased boots).
    Gap: A3 fallback and A5 repeats= never fired — a single-lane storm
    does not recreate the concurrent tier-full jam; live evidence for
    fallback-under-jam remains the 06:34 probe (461 repeats ending in
    a 200).
  * oom=128 ALERT-OOM flashing: the watchers counted the BENIGN
    transient allocator-churn WARN class (`OOM on device`, 144 MiB
    single-attempt refusals, zero request impact) cumulatively → both
    recalibrated to delta-based alerts (baseline at pin/start; alert
    only on lines NEW since the previous check); canary re-armed
    (pid 436398, oom_baseline=128).
  * fb= fallback counter added to the canary after user reported a
    visible fallback flood at 16:16:34 (event below) — first rollout
    died at the very next check: the loop-side `FB=` grep extraction
    line was missing while `set -u` consumed it → CANARY_DEAD caught
    within one 5-min cycle, root-caused via `grep -n 'FB='` on the
    remote copy, fixed, re-cycled through finish.sh (pid 442750,
    baselines `oom=128 fb=203`). Lesson: diff a remote `bash -n`-clean
    .new against the delta it was supposed to introduce before
    cycling — syntax-clean does not mean feature-complete.
  * LIVE sightings of the stale-hit livelock on real traffic (not
    tests; boot `engine-20260918-150710`, never restarted — user owns
    restarts). (1) 16:16:34–16:16:41: req
    chatcmpl-a640f44e3cfccd4c-b84990e3 advertised hit=152576 under
    concurrent full-tier churn → ~1,344 scan repeats (A5 printed
    repeats=1216/1280/1344) and 175 A3 fallback lines; plus
    chatcmpl-8f9ef834755cfce2-b31d3bc4 hit=8192 ×28 fallback lines.
    Self-drained; both requests finished 200; health 200 / procs 9 /
    0×500 throughout. This replaces the storm text above that cited
    the 06:34 probe as the only live evidence. (2) 17:05 burst (canary
    relay cycle 7 fired `ALERT: FB+10(tot213)` at the 17:06:30
    window): chatcmpl-b12c5723ed5ee639-99542ce1 hit=8192 ×7 + 
    chatcmpl-bb1132b7f38563a1-911ef89b hit=140288 ×3 fallback lines;
    no spin growth (max_repeats held 1344), drained 17:05:53. fb
    frozen at 213 through 19:31+ (28+ CYCLE_OK windows). Verdict: A3
    + A5 now fingerprint the defect live and the fallback budget (8)
    bounds it — each affected request pays a multi-second latency
    spike (the flood the user saw = one INFO line per budget
    exhaustion), while the engine itself never degraded. ROOT CAUSE
    REMAINS OPEN: under churn, tier hits are advertised from an index
    that pages evicted; the lookup trusts stale `hit=` metadata, burns
    the budget rescanning, and falls back. Candidate deep fix (not
    started, needs user go: engine restart applies it): eager
    invalidation/refresh of advertised hits on evictions, or validate
    the hit index against tier state before lookup — env-gated patch;
    validation = concurrency-shaped churn repro (real 16:16 pattern;
    the 16-iter single-lane storm demonstrably does NOT recreate it)
    + full storm re-cert.
  * A7 KVC-PIN deployed 2026-09-19 04:11 (supersedes the candidate
    fix sketch above; needs an engine restart to load). Root cause
    corrected from the hit-index hypothesis: the CPU tier is
    internally consistent — manager.lookup honors policy membership
    (manager.py:173-187), eviction removes keys synchronously inside
    prepare_store (:262-272) and complete_store failure removes keys
    (:372-378) — so there is no stale CPU-tier index to refresh. The
    observed loop is scheduler-level: a resolved hit whose
    all-or-nothing upfront conversion cannot land re-arms the 2s
    identity window (scheduler.py:1195-1219) every engine step —
    1344 identical full-scanner re-resolutions and 175 fallback flood
    lines in the 16:16 event, the manager scan repeated each step at
    zero progress. Fix: after the A3 budget collapses a request into
    the incremental-recompute path, KVC-PIN fast-paths the connector
    to (0, False) for a bounded cooldown (default 10.0s) instead of
    re-resolving; unpin on progress (num_computed_tokens change) or
    deadline lapse; fast path sits after the transfer_jobs gate so
    legitimate async waits are untouched. Env knob
    VLLM_KV_OFFLOAD_STALE_HIT_PIN_SECONDS ("0"/"false"/"off" restores
    legacy flip-flop; unset = 10.0). Diff: RequestOffloadState field
    stale_hit_pinned_until; __init__ env parse;
    get_num_new_matched_tokens fast path before update_offload_keys;
    pin set in the A3 trigger branch — 4 KVC-PIN 2026-09-19 markers.
    Deployed md5 0ab8601e095a5e402e4a5c06519d1caa (runtime = kit
    carrier patched-files/…scheduler.py = local mirror patched-files
    + deployed-sources; pre-patch 6e601eee63a8a1f887898a188bb2edfb
    backed up as scheduler.py.bak-20260919-041155). Pre-restart gates
    passed on the real venv python (engine python is
    vllm-bin/venv/bin/python, PYTHONPATH=vllm-bin/dist-packages):
    py_compile OK, marker count 4, runpy file-location import OK
    (IMPORT_OK True). Deploy script: /tmp/a7-deploy.sh (ABORT guards:
    runtime md5 mismatch, no python, compile/marker/import failure —
    aborts leave runtime untouched).
  * A7 KVC-PIN certified live 2026-09-19 04:18–05:05 on boot
    engine-20260919-041840.log (after user restart at 04:17): (a)
    post-restart sweep — A7 md5 verified, health 200, 0 fallback
    lines, 0 repeats lines, 0 500s, 0 tracebacks, canary auto-repinned
    on oom=0/fb=0 baselines; (b) 16-iter soak-verify storm re-cert
    04:35–04:59 — 16/16 ok, M1 cold 25.5s → warm 0.71s, one normal
    mid-storm M1 tier rebind (15.5s, ITER 5), CPU tier reached cap
    (ev=12412) with eviction churn through ITER 15, 0 500s, 0 fb; (c)
    churn-flood live-fire probe 05:01–05:05 (real 16:16 shape: 16
    unique ~123K-token prefills concurrent + 4 hounds re-sending
    flood00's content mid-wave while the tier was at cap and evicting)
    — all 20 requests 200 OK, fallback delta 0→0, repeats 0,
    kv_lookup +1 single clean resolution, 0 500s. Post-flood sweep
    caught the exact pre-fix trigger fingerprint resolving QUIETLY:
    `KV-LOOKUP hit req=…-a8a2264f local=0 hit=97280
    trace=[g0FAE hit=96 cov=0/97 max_hit=99328 … g4SW1EM hit=96
    cov=0/97 …]` (hit advertised but zero stored coverage per group —
    the 16:16 event produced 175 fallback flood lines + repeats=1344
    for exactly this shape; the A7 boot produced one resolve line, no
    budget burn, no flip-flop loop, next request 200 OK). Whole-boot
    counters at 05:05: fallback lines EMPTY section (zero), repeats
    lines EMPTY (zero), 0 500s; tier traffic: evicted_total=6976,
    tombstones=1900, cpu_allocated=12412, spec accept 2.50, serving
    200 OK live. VERDICT: livelock fix HELD — nil fallback/repeats
    across boot + storm + churn-flood on a full tier with ongoing
    evictions. Rollback remains `VLLM_KV_OFFLOAD_STALE_HIT_PIN_SECONDS`
    =off + restart (also restored instantly by runtime
    scheduler.py.bak-20260919-041155). Flood script:
    /tmp/churn-flood.sh (model name "qwen-3.8-flash-next", envelope
    must close with `}]}` — the first attempt 400'd twice on those).

## 2026-09-19 — boot-041840 crash: CUDA OOM (allocator fragmentation) + PYTORCH_CUDA_ALLOC_CONF hardening

- Crash at 15:32:20 on boot `engine-20260919-041840.log`: CUDA out of
  memory on TP workers 4/5 (pids 20452/20552): `MemoryError: CUDA out of
  memory. Tried to allocate 142.00 MiB. GPU has a total capacity of
  31.45 GiB of which 97.19 MiB is free … 29.76 GiB in use … 246.11 MiB
  reserved by PyTorch but unallocated` — ~343 MiB nominally free-ish but
  no contiguous ≥142 MiB block, i.e. allocator fragmentation after
  11h20m of max-churn huge-context serving (apc 1248, kv 875, tier-2
  evictions ~240K at the end). Graceful shutdown followed (MPClient
  cleanup 15:32:21, hugepages mmap released 15:33:43, one benign
  leaked-shm tracker warning); exactly one HTTP 500 for the in-flight
  request.
- Attribution: NOT the A7 stale-hit fix. Whole-boot `repeats_lines=0`,
  `rep=0`, `h5=0`; fb finalized at 39 single-line/rare-repeat damped
  events and froze for the last ~3h. This was allocation-side memory
  fragmentation in the serving path, reaching the edge on GPUs 4/5
  only after the longest sustained ultra-heavy window.
- User restarted manually at 15:35:57 → `engine-20260919-153557.log`
  (APIServer pid 119798, EngineCore pid 120360); healthy (health=200)
  within ~8 min, connector metrics fully live (`cpu_allocated=764`,
  mirrored keys 230,832, `evicted_total=0` fresh boot). New monitoring
  baseline: boot-20260919-153557 (canary auto-switched to the new log;
  counters reset, fb=0 so far).
- Hardening (requires user-owned restart to take effect):
  `vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh` gained
  `export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"`
  after the ENV-parity block (~:696), per PyTorch's own hint in the
  crash error (expandable segments make the device allocator carve
  contiguous ranges lazily, avoiding exactly this 142 MiB/97 MiB-less
  fragmentation mode). House style `${VAR:-default}` keeps an explicit
  launcher override possible. Deployed on the kit box with backup
  `.bak-20260919-expalloc`; `bash -n` clean; local+remote md5
  `558e8c4b922c3ce7590b5b6186bb8cac`; mirror copy in this repo updated
  identically. NOTE: variable was previously absent from all four
  launchers — a docker-variant parity edit is possible later if the
  no-docker path proves out.
- Watchdog bookkeeping: relay cycle 147 was lost to a transient
  subagent Bad Gateway (that window's check skipped; crash was instead
  observed in logs on demand); cycle 148's row codifies the new boot.

## 2026-09-21 — ESC211: giant-prompt stale-hit wedge escape (SERVERKILL)

Observed wedge (boot 20260921-053224, 08:18–08:21): a ~157k-token prompt sat
in the waiting queue with `local=0` forever; the engine logged one
`KV-LOOKUP hit req=... hit=155648` + one `KV-LOOKUP stale-hit fallback ...
budget=8` pair every ~10 s, client frozen, health stayed 200, and the state
only self-healed ~5 min later when the pool slowly freed room. Root cause is
two-layered:

1. Connector: `get_num_new_matched_tokens` counts identical (hit, num_computed)
   repeats only when they are <2s apart, but the stale-hit pin cooldown
   (`VLLM_KV_OFFLOAD_STALE_HIT_PIN_SECONDS`, default 10s) spaces the
   re-lookups that far apart, so every pin cycle resets the streak — the
   zero-progress budget can accrue only 8 fast repeats per ~10s cycle and
   never escalates; the requests loops hit→fallback→pin forever.
2. Scheduler core: `scheduler_reserve_full_isl` (fork default TRUE, not
   overridden by the launcher) makes `allocate_slots` demand the FULL
   sequence fit upfront for waiting requests — for BOTH the async-load
   admission (connector returns the hit) and the recompute admission
   (connector returns (0, False)). A ~157k-token admission can't get
   whole-sequence room while other giants hold the pool, so the request
   waits until blocks drain (~5 min), whatever the connector answers.

Fix (env-gated, default-on, deployed without touching the running engine):

- Connector
  `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`:
  identical-repeat window widened to `VLLM_KV_OFFLOAD_STALE_HIT_REPEAT_WINDOW`
  (default 300s; "0" restores legacy 2s); zero-progress fallbacks counted
  across pin cycles; after `VLLM_KV_OFFLOAD_STALE_HIT_FALLBACK_LIMIT`
  (default 3) fallbacks or `VLLM_KV_OFFLOAD_STALE_HIT_TTL_SECS` (default
  150s) on the same (hit, boundary), the connector logs
  `KV-LOOKUP SERVERKILL req=... -> escape (0, False)` and persistently
  answers (0, False) without re-arming the pin until the boundary changes
  (`KV-LOOKUP SERVERKILL release` on progress). New duck-typed scheduler
  hook method `is_stale_hit_escape(request)`.
- Scheduler core `vllm/v1/core/sched/scheduler.py`: for escape-mode requests
  (only, sync path only) the per-request `full_sequence_must_fit` is
  relaxed so the chunked 1024-token recompute path can schedule under pool
  pressure (normal per-chunk checks + watermark still apply; other
  requests' admission unchanged). Disable with
  `VLLM_KV_OFFLOAD_ESCAPE_RELAX_FULL_FIT=0`. All new envs are VLLM_
  prefixed (`-V` failover tool safe).

Effect: a wedged giant prompt enters escape within ~20-40s (fallback ticks)
or 150s (TTL), then schedules locally chunk-by-chunk (~25s full recompute
for a 155k prompt at this box's measured rate), instead of hanging until
the pool frees whole-sequence room. Deployed 2026-09-21 with backups
`.bak-20260921-esc211` on both files; py_compile clean; md5
connector `4f7c9265a2b27e01013cbaa09b2f6653`, sched core
`e4d7ed255b2ff01f85d805e237388182` (pre-patch: connector
`0ab8601e095a5e402e4a5c06519d1caa`, sched core
`3d8e5c39fba2e919c1c9753a8ff11a99`). Engages on next engine
restart. Byte-identical mirror copies tracked in `patched-files/`
(md5-verified against the deployed files):
`patched-files/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
(connector; mirrored copy history: `0ab8601e` pre-ESC211 → `4f7c9265`)
and, newly added to the manifest,
`patched-files/v1/core/sched/scheduler.py` (scheduler core).

Production evidence (2026-09-21→22, boot `engine-20260921-143456.log`, ~18.9 h of
serving; canary relay cycles observed fb 13→97 across it): 1366 KV-LOOKUP-probed
requests; 97 stale-hit fallbacks (~7%); 91 one-shots self-resolving; six
multi-strike events, all inside the 20:05–22:11 window (walls 10.3–10.4 s); one
full designed ladder, 22:11:08→22:11:18: fallbacks=3 → `KV-LOOKUP SERVERKILL` →
release (wall 20.5 s), request resumed via chunked recompute. TTL never engaged;
zero engine kills attributable to LOOKUP. Review verdict (2026-09-22, advisory):
keep FALLBACK_LIMIT=3, REPEAT_WINDOW=300, PIN_SECONDS=10, budget MAX_LOOKUPS=8,
TTL=150 unchanged; the "fail-2-strikes" signature is pinned as diagnostic-only
(5/5 observed two-strike events self-resolved).

## 2026-09-22 — hard GPU-driver lock + box recovery (NOT ESC211; unrelated chain)

- 09:20:55 a ~106k-token prefill was in flight when workers stopped servicing the
  EngineCore shm broadcast ring (4× "No available shared memory broadcast block"
  warnings 09:21:55–09:24:55; no CUDA/Xid/OOM line first). 09:25:55 worker RPC
  `sample_tokens` timeout → `EngineDeadError` → orderly self-teardown.
  09:44–09:47 kernel escalation: nvidia-modeset display errors (GPU:1), then
  NVRM `RC watchdog: GPU is probably locked!` (GPU3): the GPU command engine
  itself locked. TP workers D-state inside the driver → `kill -9` ineffective;
  reboot hung unmounting /mnt/data; hard power-off 09:47:18.
- Post hard-off (boot 10:03): GPUs clean, zero Xid. Engine relaunched natively
  (no-docker launcher) 10:07. Independent dockerd issue that boot: 10:04:09
  kernel BUG `fs/iomap/buffered-io.c:1061` (Oops, Comm: dockerd — write-path)
  → daemon deterministically wedged ("activating", CLI hangs) through
  SIGKILL + 3 restarts + containerd restart. Resolved by the user-approved
  11:22 reboot (fsck armed): GPUs clean, `docker ps` fully healthy
  (gpu-monitor + open-webui up), boot completed cleanly 11:22:32, engine
  relaunched natively (pid 11468, ESC211-patched code active from this boot.
- pstore unusable on both boots (erst backend write errors; /sys/fs/pstore
  unreadable for the user; /var/lib/systemd/pstore empty) → NO persistent kernel
  record of the lock. User decision (2026-09-22): leave pstore as-is, rely on the
  dmesg/engine-log+journal transcript posture.
- Monitoring-lane addendum (~13:55 onward): canary v3.1's station died silently
  in a bash `do_truncate` on its tiny state file (`canary.state`,
  reopen-truncated every 5 min since 11:44 — the only operation class that
  wedged). Subsequent metadata ops on that inode/directory wedged XFS likewise
  (rename → `lock_two_nondirector`, dir listing → `iterate_dir`) while the
  engine (append-only writes, new-file creation) stayed fully healthy; 2
  D-state pids parked until reboot. User ran `sudo -n systemctl reboot -f`
  ~16:24 (direct force path; bypasses the D-state-prone orderly unmount that
  hard-hung twice earlier today): journal replay healed the canary directory
  completely (no partial rename target survived), the wedged kernel state
  died with the boot, engine relaunched natively (bootlog
  `engine-20260922-162408.log`, 8/8 GPUs, docker×4, p=9, health=200). Canary
  v3.2 station restarted (pid 15447); its first row carries the designed
  informational `NEWBOOTLOG` rebase onto the fresh bootlog. Kernel journal for
  the fresh boot shows no XFS recovery/repair lines (replay was silent).
  Watch in fresh relay cycles: any wedge recurrence (freshness probe catches a
  stalled-alive station), and any serving-window shm emissions on the new boot
  (2 startup-window shm lines were absorbed at the rebase baseline).
- 17:00–17:23 relay addendum (cycles 96–99): three `hb=000` SERVE-WEDGE rows
  (16:57:57, 17:13:29, 17:19:03) clustered across 22 min while the tier hit
  saturation (`allocated=12412` = `evictable_len` = cap, free list 0,
  `evicted_total` 152→18042) under soak+client load. Each triaged live:
  health=200, fb/kv/apc/ev counters still growing, KV-LOOKUP + engine
  metrics flowing → verified LOAD-CONGESTION class, not a functional wedge;
  tag and escalation posture unchanged.
- Canary station death #2 (~17:13–17:16, post-[17:13:29] row): abrupt,
  NO `CANARY-EXIT` row (hard-signal class — the v3.2 EXIT trap is correctly
  silent there), no oom-killer trace, unknown actor. Cycle-98's probe
  detected it via the `station.pid` + `kill -0` check — v3.2 detection depth
  proving its purpose. Relaunch #2 (17:18:40) succeeded THROUGH a malformed
  wrapper: in `bash -c "cd canary_dir && nohup bash canary5m.sh & sleep 3; …"`
  the `&` backgrounded the WHOLE `cd && nohup` group, so the foreground
  relative reads ran in $HOME (spurious ENOENTs interpreted as a suspected
  dir wedge #2 — disproved by cd-probes 17:20: the dir was healthy all
  along; no second kernel wedge occurred). The nohup'd child (pid 23448)
  survived the ssh-session teardown (HUP-immune), resumed from the saved
  state offset (same bootlog epoch, no rebase) and itself wrote the
  [17:19:03] row. `station.pid` hand-aligned to 23448 at 17:22:44 — note
  the pidfile is written by the launcher step, not the script.
- FS attribution CORRECTION: `/mnt/data` = **ntfs3** mount
  (`/dev/nvme0n1p4`, ntfs3 driver). The wedged functions recorded today
  (`do_truncate`, `lock_two_nondirector`, `iterate_dir`) are fs/ntfs3
  driver paths; the "XFS wedge" wording above (and any earlier "XFS"
  mention for this dir) is a misattribution — corrected 09-22 evening.
  ("no XFS recovery lines" after the 16:24 boot also reads correctly
  a posteriori: none ever existed.)
- Soak ladder: iters 10–13 (17:11–17:19) each ALERT — M1 42/37/53/10 s,
  shard-hits oscillating 8→156→8→172/172 while KV-LOOKUP keeps "reporting"
  hits, `evt` → 18042: the soak's designed silent-tier-loss detection
  confirmed live. Evidence preserved: cp `/tmp/soak.log` →
  `logs/soak-20260922-1720-preserve.log` (17:20:19) since /tmp is wiped on
  reboot. Ladder continues to ~21:09.
- FS wedge #3 — ROOT CAUSE CLOSED (17:30–17:36): kernel journal smoking gun
  at 17:13:29 — `kernel BUG at fs/iomap/buffered-io.c:1061! Oops: invalid
  opcode [#1] ntfs_file_write_iter+0x263/0x300 [ntfs3]`. This SINGLE event
  explains the whole wedge cluster on the canary dir: (a) it killed v3.2
  station 15447 outright INSIDE the per-cycle `canary.state` truncate — no
  trap, no `CANARY-EXIT` row, no oom (the "mystery" death #2 of cycle 98);
  the same write path (per-cycle truncate-rewrite) then wedged relaunch
  station 23448 D-state `WCHAN=do_truncate` (SIGKILL-proof) — kernel-fs ops
  on the dir are off-limits; 23448 is left unkillable in place and dies with
  the user's reboot. Note `kill -0` SUCCEEDS against a D-blocked pid — the
  probe's 8-min freshness bar was what actually caught the wedged-but-
  "alive" station (validated design decision, twice today).
- User decision (17:36): temporary canary data → `~/vllm-glm-5.3-flash-
  nvfp4-sm120-logs` (`$HOME` = /dev/nvme0n1p5, **ext4** — escapes the ntfs3
  wedge class entirely); reboot deferred to the user.
- Canary **v3.3** shipped (17:31 local edit → 17:38 deploy): state write
  changed from truncate (`> STATE`) to **APPEND-ONLY** (`>> STATE`) into
  `canary.state.journal` (readers already last-match-per-key), so the
  overwriting write that triggered all three wedge visits no longer exists
  in the canary's own behavior; home = `$HOME/vllm-glm-5.3-flash-nvfp4-
  sm120-logs/`; historical rowlog cp'd READ-ONLY from the old dir into the
  new rowlog (single-file continuity 12:45→now); hb.json byte-copy (md5
  parity). Station 27057 launched 17:38-39 (correct one-liner shape; ssh
  timeout on the held session = known-benign), pidfile minted 27057,
  ALIVE, first row **[17:39:00] h=200 hb=200 fb=27 kv=215** — counters
  carried across the 17:19:03→17:39:00 park window via the state-journal
  offset resume; NO NEWBOOTLOG (same bootlog epoch as designed). md5
  `6ba78a624ab898310b4c80c6613c59b6` = `tools/canary5m.sh` mirror. Old
  `/mnt/data/.../logs/canary/` = FROZEN WEDGE ZONE until user reboot.
  Soak continues (evicted_total ~29.5k at 17:39, climbing under churn).
- Post-re-arm hb congruence check PASSED: a FRESH hb=000 appeared at
  [17:44:20] (ev 29562→31166, industrial churn window) — cycle-100 lane
  escalated per rule; orchestrator triaged contemporaneously (17:47:54,
  3m34s after the row): manual heartbeat POST → HTTP 200 in **3.426813 s**
  (budget 20 s), health=200 → LOAD-CONGESTION class, same as the
  16:57/17:13/17:19 cluster. Cycle 101 (fix-64 lane, after one stop/recover
  round-trip) verified via verbatim probe: CYCLE_OK under the documented
  pre-triage phrasing (station 27057, health=200, cadence 17:39→17:44
  5:20~nominal); a parallel direct probe (17:48:46) returned the identical
  read. The v3.3-era chain is at cycle 101; next rows land ~17:50+.
- SOAK-PEAK SATURATION BAND (18:10–18:19): cycles 102/103/104 all CYCLE_OK
  (rows [17:54:53], [17:59:59], [18:05:02] hb=200, ev→44,074). Cycle 105
  then reported fresh hb=000 rows [18:10:22] and [18:15:43] (SERVE-WEDGE
  tags; envelope clean: h=200 p=9, 500/oom/shm/gpu=0, no CANARY-EXIT).
  Contemporaneous triage 18:17:30: TWO manual heartbeat POSTs EACH TIMED
  OUT at the full 20 s budget (HBC=000, HB_LAT=20.001678 / 20.001072)
  while health=200 and counters climb (ev 44,074→47,366→52,070) — a
  PERSISTENT serve-pipe band, NOT the 17:44-style transient. Correlation
  is exact: soak iters 25–32 (17:57–18:18) show M1 walls 22–65 s
  oscillating severe (two flicker-ok: iters 26 M1=9.1, 30 M1=4.7), M1G
  (GET) walls 27–57 s, the re-report signature in pure form on every
  iteration (`hit=NNNNN … cov=0/NN` on ~5-guide traces), and ONE ESC211
  guard acting as designed (17:59:38 `KV-LOOKUP stale-hit fallback …
  budget=8 fallbacks=1 wall=3.5`). Cause chain: the soak's marker submit
  rides the scheduler burn loop at tier full (evt 39,258→54,730 by iter
  32); the 20 s hb POST budget is simply below the current burn-loop
  walls, so the canary's hb=000 and the manual probes measure the SAME
  saturation. Classification: EXPECTED soak-peak behavior (the soak's
  designed disease-window, ~58 iters remaining to the ~21:09 end),
  engine ENVELOPE intact (no process/fs/OOM engagement). Rows keep their
  honest ALERT tags; each future fresh hb=000 row during the run gets
  reported per rule and triaged on arrival (the immutable pre-triaged
  set is NOT extended by this wave). Evidence preserved #2:
  `logs/soak-20260922-1820-preserve.log` (cp 18:19, through iter 32).
- BAND CONTINUATION + DEEPENING (cycles 106-107, 18:18-18:39): rows
  [18:20:45]/[18:26:03] back to hb=200 (band flicker), then FRESH hb=000
  rows [18:31:23] + [18:36:43] back-to-back. Cycle-107-era triage at
  18:39:22: manual heartbeat POST #3 HBC=000 HB_LAT=20.001264 (third
  consecutive full-budget timeout), health=200; soak iters 33-37 walls
  DEEPEN — [18:34:12] ITER 36 M1=**219.298 s** X36=107.34 (largest so
  far), ITER 37 M1=46.37/X37=51.09/M1G=108.01, evt 59,722→62,606. The
  escalation scale is monotone-ish with churn depth; the 20 s hb budget
  remains far below current walls. CLOSED rows ledger add: [18:31:23],
  [18:36:43]. Posture unchanged: soak-peak overload is the EXAMINED
  phenomenon (not a fault); engine envelope clean throughout (p=9
  baseline, 500/oom/shm/gpu=0, no CANARY-EXIT, health=200 always).
- ZERO-PROGRESS WINDOW DECODED (cycle 108, triage 18:54:15): rows
  [18:42:04]/[18:47:24]/[18:52:44] hb=000, and the latter two show fb/kv/
  apc deltas = 0 with e2-field frozen 66,254. Root mechanism measured:
  **[18:53:31] soak ITER 39 completed with M1G=569.879579 s** (one GET
  request occupied the scheduler burn loop for ~9.5 min — the whole
  canary cadence window), M1=100.5/X39=67.1, evt=66,254; iters 38-39 all
  ALERT (M1 39.3/100.5). Engine census ALIVE: APIServer 10195 SLsl,
  EngineCore 10755 **RLl** (R = burning), both 2h30m uptime; hb probe #4
  HBC=000 at 20.002046 s; health=200. Engine-metrics anatomy at the
  window: `kv_offload_cpu_cache_write_usage_perc=0.000645` (0.06%),
  `..._read_usage_perc=0.0`, allocation_size_count=68/sum=272 — the
  CPU cache is doing ~nothing while hits continue to be re-reported:
  the ESC215-class silent-tier-loss anatomy confirmed at saturation
  (tokens re-report big `hit=` values, zero coverage, zero actual
  offload I/O). CLOSED rows ledger add: [18:42:04], [18:47:24],
  [18:52:44] (13; soak peak continues to ~21:09). Envelope stayed
  green the whole window.
- BAND OSCILLATION (cycles 109-110, 19:01-19:10): cycle 109 CYCLE_OK
  ([18:57:56] hb=200, zero-window cleared, ev→69,454). Cycle 110 then
  flagged ONE fresh hb=000 row [19:03:16], sandwiched between clean
  hb=200 rows [18:57:56]/[19:08:16] with progress climbing straight
  through it (fb 85→86→90, e2 69,454→73,662→78,037) — the lightest
  band shape: a single-cell transient whose 5-minute window caught one
  >20 s wall. Triage 19:10:38: manual heartbeat POST #5 → **HBC=200
  HB_LAT=1.474637 s**, health=200 (pipe serving fast again within
  ~7.5 min of the row). CLOSED rows ledger add: [19:03:16] (band
  ledger = 6 rows + cluster-3; soak runs to ~21:09 with the band
  expected to keep oscillating; envelope never broke).
- OSCILLATION PULSE #2 (cycle 111, 19:13-19:22): fresh hb=000 row
  [19:13:37], again sandwich-shaped (clean [19:08:16] before,
  [19:18:43] hb=200 after; progress climbed through: fb 90→95→103,
  e2 78,037→82,309→85,365). Triage 19:22:14: manual heartbeat POST #6
  → **HBC=200 HB_LAT=0.237372 s** (fastest yet), health=200. Soak
  walls easing: ITER 48 (19:20:00) M1=44.45 ALERT but ITER 49
  (19:21:32) **ok: M1=2.49 s, M1G=2.83 s**, evt→87,261 — the ladder's
  deep phase may be tapering. CLOSED rows ledger add: [19:13:37]
  (13 closed rows total; soak end ~21:09, post-soak clear-up check
  planned at the next relay window after ~21:10).
- OSCILLATION PULSE #3 + RECONCILE CORRECTION (19:34-19:38): cycle 113
  (fix-76) flagged fresh hb=000 row [19:34:33] (fb 115→122, kv→819,
  e2→93,693 climbing through it; sandwich [19:29:12] clean → miss →)
  — an intermediate summary mislabeled this row/verdict as clean/OK;
  the direct rowlog read (19:38:26, append-only log) restored the
  truth: the row IS hb=000+SERVE-WEDGE and the lane's CYCLE_ALERT was
  right. Triage 19:38:26: manual heartbeat POST #7 → **HBC=200
  HB_LAT=1.412449 s**, health=200; soak iters 52-54 walls 75.4/44.1/
  6.1-s-ok (evt 93,341→96,341). CLOSED rows ledger add: [19:34:33]
  (14). A late "cycle-114 completion ~19:44:52" claim with no lane,
  no stdout and no board presence was dismissed as transmission
  noise — the rowlog at 19:38:26 ends at [19:34:33]; next row due
  [19:39:4x].
- OSCILLATION PULSE #4 (19:54-20:02): cycle 116 (fix-79) flagged
  fresh hb=000 row [19:54:58] (sandwich [19:49:37] → [19:59:58]
  clean; fb 133→135→135, kv→951, e2→110,244 climbing through) plus a
  zero-progress pair note ([19:54:58]→[19:59:58] identical fb=135
  with rmax=0 — informational GET-burn tail). Triage 20:02:55:
  manual heartbeat POST #8 → **HBC=200 HB_LAT=15.340971 s** (slowest
  successful yet — the ladder compressed the pipe nearly to the
  20 s budget), health=200; soak iter 60 (19:56:22) **M1=63.95 s
  ALERT sits inside the missed window**; iters 61-62 ok 3.6/7.7 s;
  evt→110,908. Saturation itself is NOT over — ladder still hot;
  the row is closed as functional. CLOSED rows ledger add:
  [19:54:58] (15).
- OSCILLATION PULSE #5 — 2-MISS WINDOW (20:20-20:26): cycle 119
  (fix-82) flagged fresh hb=000 [20:20:55]; the direct read at
  20:26:40 shows the window DEEPENED to two consecutive misses
  ([20:20:55] + [20:26:16], both SERVE-WEDGE, envelope p=9/extras-0)
  with progress climbing through (fb 152→161, kv→1096, e2→126,932 —
  zero-progress-FREE). Manual heartbeat POST #9 inside the gap →
  **HBC=200 HB_LAT=0.168744 s** (fastest of the soak; pipe
  intrinsically fine between rows), health=200; soak iters 68-70
  all ALERT (M1 11.5/25.6/45.0 s, X 77.4/65.8/46.0, M1G up to
  56.6 s; evt→127,052) mechanically cover the miss window. CLOSED
  rows ledger add: [20:20:55], [20:26:16] (17) — closure evidence =
  measured manual-HB + envelope + progress; formal sandwich close
  when the [20:31:3x] recovery row lands (lane-reported). Soak end
  ~21:09 (~10-12 ladder iters left).
- OSCILLATION PULSE #6 — 3-MISS STREAK (20:20-20:35): no recovery
  row yet: [20:31:36] is the THIRD consecutive hb=000 (all
  SERVE-WEDGE, envelope clean, engine 10755 RLl running — NOT
  D-wedged). Progress climbs through all three (fb 152→171,
  kv→1146, e2→127,425; zero-progress-FREE — this is burn-loop
  compression, not the 18:47 GET-occupation freeze shape).
  Manual heartbeat POST #10 (20:35:01, 3.5 min after [20:31:36])
  → **HBC=200 HB_LAT=0.256199 s**, health=200 — the pipe stays
  functional in the gap; misses are canary-query contention inside
  the burn window while soak walls hold M1 41-66 s (iters 71-73 all
  ALERT, evt→132,920). CLOSED rows ledger add: [20:31:36] (18). If
  the streak reaches 5 misses or any zero-progress freeze appears,
  escalate to engine-metrics capture immediately. Soak end ~21:09.
- POST-SOAK EVIDENCE #3 PRESERVED (21:33): machine-side `cp` →
  `logs/soak-20260922-2130-preserve.log` (ext4 target; /tmp source still
  in place). md5 `faff53af6f99815f767bc8df2feaaf6f` IDENTICAL on both
  sides. 92 lines / 30,650 bytes — last five iters: 85 [21:10:18] ALERT
  M1=27.4s, 86 [21:12:25] ok 0.69s, 87 [21:15:24] ok 0.75s, 88
  [21:19:12] ALERT M1=30.95s, 89 [21:22:34] ALERT
  M1=25.69s/X89=42.54s/M1G=27.50s; log's own end marker
  `=== soak end ===` at 21:23:58. soak_watch verified GONE at 21:33
  (pgrep clean; ended on its own, slightly past the designed ~21:09);
  grep for DONE/END/COMPLETE found no uppercase lines (end marker is
  lowercase). NOTE: run's log is compact (~330 B/line, ~1 line/iter),
  NOT the expected ~3200 lines / 250-280 KB — noted and preserved
  as-is. Copy-mode only lane (no restarts, no state writes, no kills).

## 09-24 EPOCH records (fresh boot 09-24 17:48, engine 18:00:02)
- RE-ARM + FIRST MISS (22:06-22:19; commit 3fcfc91 has the re-arm
  STATE fragment): canary v3.3 re-armed 22:06 (station pid 45568,
  launcher-minted; ALIVE verified; script md5 parity `6ba78a…`).
  09-24 rows: [22:06:09] hb=200 + NEWBOOTLOG (informational rebase,
  epoch counters reset fb=0 kv=27 ev e2=11,196), [22:11:10] hb=200
  (fb=1 kv=36), [22:16:30] **hb=000 fb=1 kv=38 e2=16,140 ALERT:
  SERVE-WEDGE — FIRST NO-SOAK MISS of the 09-24 epoch**. Triage
  PROBE-HISTORY: manual-HB at 22:15:48 = HBC=200 HB_LAT=6.997303
  (PRE-miss measurement, excluded as closure evidence);
  manual-HB at 22:19:48 = **HBC=200 HB_LAT=0.293957** (post-miss
  functional proof, 3.3 min after the row) + health=200 + p=9
  envelope clean + EngineCore 10125 running. CLOSED: [22:16:30]
  (09-24 ledger add: 1). NO-SOAK HAZARD noted: the 09-22 soak is
  over (evt 162,557) — recurring hb=000 rows without soak load are
  a deeper-wedge candidate: escalate immediately (engine-metrics
  capture, user-owned restart decision), never self-remediate.
  Old ntfs3 pair 23446/23448 CONFIRMED gone (09-24 journal replay).
- PULSE #2 — NO-SOAK PAIR + METRICS (22:21-22:27): second 09-24
  miss [22:21:51] hb=000 fb=1 kv=47 apc=177 (bytes+55806, e2→16,628
  climbing) — consecutive with [22:16:30] → escalation-plane per
  pulse #1. Triage 22:27:36: manual-HB **HBC=200 HB_LAT=0.211947**
  (fastest of the epoch), health=200; sandwich recovery row
  [22:27:05] hb=200 (fb→2, kv→50, e2→18,332); engine metrics sampled:
  `vllm:kv_offload_cpu_cache_write_usage_perc = 0.0`,
  `_read_usage_perc = 0.0` — the CPU KV-offload is COMPLETELY IDLE,
  so the 09-22 silent-tier-loss mechanism is NOT active on 09-24;
  the miss shape is indicative of light contention (likely the
  192.168.2.208 giant-client prefill traffic resuming on the fresh
  boot — counters climb through both misses; apc 159→186). CLOSED:
  [22:21:51] (09-24 ledger add: 2). Hazard lines STAND: any future
  miss cluster (>2 consecutive) or a failed manual-HB → user-owned
  restart decision; both misses here closed with healthy probes; no
  restart indicator.
- PULSE #3 — 3rd NO-SOAK MISS (epic 22:42-22:48): cycle 126 flagged
  fresh [22:42:33] hb=000 fb=7 kv=73 apc=238 (bytes+54007, e2→25,988
  climbing; sandwich [22:37:13] clean → miss → [22:47:34] clean).
  Triage 22:48:54: manual-HB **HBC=200 HB_LAT=0.527019**, health=200;
  sandwich recovery row [22:47:34] hb=200 (fb→8, kv→81, e2→30,468);
  engine metrics re-sampled: kv_offload write/read **0.0/0.0** —
  STILL IDLE; the 09-22 soak silent-tier-loss mechanism remains
  ABSENT; the misses read as single-cell sandwich transients under
  the fresh boot (likely low-intensity overlap of the restarted
  engine's warmup traffic with the canary heartbeat — all 3 misses
  closed with healthy probes: 0.294s / 0.212s / 0.527s). CLOSED:
  [22:42:33] (09-24 ledger add: 3). ESCALATION POSTURE: 3 misses /
  ~31 min, ZERO consecutive-cluster, zero failed manual-HBs, metrics
  idle ×2 — no restart indicator yet; the FOURTH miss, an HBC≠200,
  or any zero-progress window → user-owned restart decision plane
  (surface to the user, never self-remediate).
- PULSE #4 — 4th NO-SOAK MISS + LATENCY TREND (23:03-23:12): cycle
  129 flagged fresh [23:03:23] hb=000 fb=11 kv=101 apc=290
  (bytes+29,597, e2→39,174 climbing; sandwich [22:58:03] clean →
  miss → [23:08:24] clean with rmax=0 single). Triage 23:12:58:
  manual-HB **HBC=200 HB_LAT=14.901424** — CLOSEST-TO-BUDGET success
  of the epoch (prev closures 0.294/0.212/0.527 s: a real
  intra-epoch SLOWING of the hb path), health=200; engine metrics
  third sample: kv_offload write/read **0.0/0.0** STILL IDLE. CLOSED:
  [23:03:23] (09-24 ledger add: 4). ESCALATION STATUS: 4 misses /
  ~47 min, zero consecutive clusters, zero failed manual-HBs,
  metrics idle x3 — the pulse-#3 RESTART-DECISION PLANE is now
  SURFACED to the user (fourth-miss trigger fired) with the
  honest read: envelope unbroken all epoch; hb-path latency
  degrading inside closures (0.2→0.5→14.9 s — pipe compression
  building on the fresh boot WITHOUT soak traffic; counters young:
  fb 1→12, kv 27→107). User picks: keep relay-verified run as-is
  (misses are single-cell, always enveloped), or take the
  user-owned §7 restart whenever convenient — relay continues
  either way; auto-signals for the restart remain (cluster of 2+
  consecutive misses, ANY failed manual-HB, metrics leaving idle).
- PULSE #5 — 5th NO-SOAK MISS, LATENCY ~15s PLATEAU (23:13-23:21):
  cycle 130 flagged fresh [23:13:44] hb=000 fb=17 kv=124 apc=311
  (fb jumps 11→12→17, kv 107→124 through the miss window; sandwich
  [23:08:24] clean → miss → [23:18:45] clean). Triage 23:21:14:
  manual-HB **HBC=200 HB_LAT=15.271009** (2nd consecutive ~15 s
  success — the hb path now spends ~75% of its 20 s budget inside
  the same window that produces the misses; isolated hb=000 rows =
  the >20 s tail of the SAME distribution), health=200; kv_offload
  write/read **0.0/0.0** (4th idle sample). CLOSED: [23:13:44]
  (09-24 ledger add: 5). ESCALATION STATUS: 5 misses in ~75 min
  since the 22:06 re-arm, zero consecutive clusters, zero HBC
  failures, kv-offload idle x4; success-latency ladder:
  0.294 → 0.212 → 0.527 → 14.901 → 15.271 s. Auto-signals
  unchanged (2+ consecutive misses, ANY HBC≠200, metrics>0).
  User decision outstanding from pulse #4: keep verified-run vs
  user-owned §7 restart; both safe, relay continues either way.
- PULSE #6 — 6th NO-SOAK MISS, LATENCY LADDER STEEPENS (23:24-23:31):
  cycle 131 flagged fresh [23:24:05] hb=000 fb=27 kv=156 apc=327
  (bytes+46,472; fb 19→27 + kv 134→156 through the window — big
  client clearly back and driving hard; sandwich [23:18:45] clean →
  miss → [23:29:09] clean). Triage 23:31:16: manual-HB **HBC=200
  HB_LAT=16.609373** (3rd consecutive ~15-16 s: 14.901 → 15.271 →
  16.609 — the hb path spends ~80% of its budget CASUALLY now;
  misses = the beyond-budget tail of the same distribution),
  health=200; kv_offload write/read **0.0/0.0** (5th idle sample).
  CLOSED: [23:24:05] (09-24 ledger add: 6). ESCALATION: 6 misses in
  ~94 min since re-arm, zero consecutive clusters, zero HBC
  failures, metrics idle x5; success-latency ladder:
  0.294 → 0.212 → 0.527 → 14.901 → 15.271 → 16.609 s. User decision
  STILL outstanding (from pulse #4): keep verified-run (working;
  misses cost nothing) vs user-owned §7 restart (cleanest way back
  to sub-second heartbeats tonight). A 7th miss → hard re-surface;
  2+ consecutive misses or ANY HBC≠200 → immediate fail-fast
  escalation.
- PULSE #7 — 7th NO-SOAK MISS, LADDER BREAKS (23:34-23:39): cycle
  132 flagged fresh [23:34:30] hb=000 fb=34 kv=185 apc=343
  (bytes+52,702, e2→55,433; sandwich [23:29:09] clean → miss; the
  lane's stdout echo of historical [22:52:44] showed fb=2 while
  three independent reads show fb=8 — copy-slip in the echo only,
  log row untouched). Triage 23:39:38: recovery row [23:39:30]
  hb=200 (fb HOLDS at 34, kv→188, apc→352, e2→58,473) + manual-HB
  **HBC=200 HB_LAT=0.155111** — FASTEST of the epoch, at the
  HIGHEST counters so far. CLOSED: [23:34:30] (09-24 ledger add:
  7). TREND MODEL CORRECTED: the ladder
  0.294 → 0.212 → 0.527 → 14.901 → 15.271 → 16.609 was NOT
  monotonic decay — probes at LOW counters ran fast, mid-ladder
  probes ran ~15 s, and this probe at the highest counters snapped
  back sub-second → hb-window contention is BURSTY (coalescing with
  the giant client's own request stream), not a fill-driven
  degradation; single-cell hb=000 rows = the hb POST landing inside
  a burst window. ESCALATION: 7 misses in ~100 min since re-arm,
  zero consecutive clusters, zero HBC failures, kv-offload idle x6;
  auto-signals unchanged (2+ consecutive misses, ANY HBC≠200,
  metrics>0). User keep-vs-restart decision remains open; the
  bursty model makes keep-as-is fully defensible; restart stays
  optional for a clean hb-path slate.
- 2026-09-25 13:45 — EPOCH PAGE-TURN (user took the §7 restart):
  canary stopped cleanly 09-24 23:54:24 (rowlog [23:54:24] ALERT:
  CANARY-EXIT rc=0 — stop-recipe exit path, user-side; monitoring
  gap 23:54→13:45 ~14 h, acknowledged). Box rebooted 09-25 ~12:43
  (up 1:02 at 13:45:24). NEW ENGINE EPOCH:
  logs/engine-20260925-125527.log (via updated latest.log symlink;
  APIServer pid 9336; health=200 verified 13:45:24; big client
  192.168.2.208 already driving — POST /v1/chat/completions 200 in
  the bootlog tail). CANARY RE-ARMED 13:46-13:47 (v3.3, ext4 home
  unchanged): station pid **40421** (wrapper 40419; station.pid
  rewritten; ALIVE verified), row-1 **[13:45:53] h=200 p=9 500=0
  oom=0 shm=0 gpu=0 hb=200 fb=0 kv=2 rmax=1 rep=0 apc=26 bytes=+0
  ev=2384.0,0.0,0.0 ALERT: NEWBOOTLOG** — informational epoch
  rebase (counters reset for the fresh engine). The 09-24 epoch
  closes: 7 no-soak misses/~100 min, zero clusters, zero HBC
  failures, kv-offload idle x6, bursty hb-contention model, and
  the user answered the open keep-vs-restart decision by taking
  the restart. Cycle-134's lane had died on a Bad Gateway before
  its probe (no partial work). Relay continues on the 09-25
  epoch: NEWBOOTLOG informational only; any other hb=000 row =
  fresh un-triaged → CYCLE_ALERT per rule.
- EPOCH PULSE #1 — FIRST FRESH MISS + GHOST VERDICT DISMISSED
  (14:16-14:24): cycle 140 (fix-29, own session) reported fresh
  [14:16:16] hb=000 fb=2 kv=13 apc=77 (bytes+68,891, ev→7,468 —
  the in-epoch fb-ramp started INSIDE the miss window: fb 0→2→3,
  apc 48→77→88, ev 3,772→9,864 across [14:10:55]→[14:16:16]→
  [14:21:26]). A PHANTOM lane record (foreign session id — the
  task_result refused "does not belong to this session") claimed
  [14:21:26] hb=000 as a second fresh miss + stale-pid lore —
  ALL CONFABULATION. Adjudication probe 14:24:25 (direct read):
  rowlog shows [14:21:26] **hb=200** (fb=3 kv=22 apc=88, ev=9,864)
  — phantom verdict rejected; station lore 40421-only confirmed.
  CLOSED: [14:16:16] on manual-HB **HBC=200 HB_LAT=1.264713**
  (fast, healthy) + sandwich [14:10:55]/[14:21:26] + fb-ramp
  progress-through + kv_offload write/read **0.0/0.0** idle
  (09-25 sample: 1.0 idle; the 20.2k pool losses of 09-23 absent
  in the fresh engine). 09-25 miss ledger add: 1. The post-restart
  bb-path test NOW ENGAGED: fb-ramp running (0→3 in one window),
  big client re-engaging load — every 09-24-miss pattern candidate
  will be tested against the fresh engine; auto-signals per
  standing rules (2+ consecutive misses, ANY HBC≠200, metrics>0).
- EPOCH PULSE #2 — FIRST FAILED MANUAL-HB; FAIL-FAST LIT
  (14:37-14:44): cycle 143 (fix-31) reported fresh [14:37:17]
  hb=000 fb=12 kv=103 apc=113 (bytes+64,022, ev extended
  1220.0,367.0 — eviction churn started; sandwich [14:31:57]
  hb=200 → miss]). Triage 14:44:04: **HBC=000 HB_LAT=20.002253**
  — the manual-HB POST died at full budget: FIRST manual-HB
  failure since the recipe's creation (the 09-22 era had three
  20s-timeouts in the 18:17-18:54 deep band). Recovery row
  [14:42:36] hb=200 (fb 12→18, kv 103→125, apc 113, ev→3448,898 —
  ramp steepening: kv nearly doubled per window), health=200.
  kv_offload STILL 0.0/0.0. [14:37:17] stays **OPEN (wedge
  candidate — not closed)**: the pulse-#1 recipe closes on a
  contemporaneous successful probe, and the contemporaneous probe
  FAILED. FAIL-FAST FLAG = FIRED 14:44:04, surfaced to the user
  with the decision template: keep-verified-run (server healthy,
  client served, misses isolated so far — but the hb path now
  fails-to-probe, not just misses) vs user-owned §7 restart
  (empirical note: BOTH user restarts of the engagement produced
  this same fill-driven hb-compression curve within ~1.5-2 h of a
  fresh boot under the client's ramp — a restart buys quiet
  hours, not immunity). ESCALATION: the epoch's 2nd miss + 1st
  failed probe; auto-signals now SUPER-armed: the next hb=000 row
  ([14:47:3x] window) = consecutive pair → CLUSTER — ESCALATE
  IMMEDIATELY; next manual-HB success closes [14:37:17] per
  recipe; ANY further HBC=000 probe = the hard signal repeats.
- 2026-09-25 15:03-15:31 — **ENGINE CRASH #1 (first of the
  engagement) — canary-documented end-to-end; user relaunched**:
  OLD epoch (engine-20260925-125527.log, boot 12:55) died between
  the canary rows [15:03:38] and [15:08:38]; death block:
  **[15:08:38] h=000 p=0 500=2 oom=244 shm=0 gpu=0 hb=000
  apc=118 bytes=+546,623 ev=?,?,?** (no SERVE-WEDGE tag =
  connection-REFUSED class, not the hb 20s-timeout class — the
  server was GONE by the row's collect; p=0 = worker census gone;
  ev=?,?,? = metrics endpoint gone). ROOT CAUSE (diag2, verbatim
  chain): CUDA-side memory exhaustion — 211 `OOM on device`
  warnings from **14:51:23** (expandable_segments mapping
  failures, 20 MiB requests when GPUs free-memory ~1-9 MiB) to
  **15:06:11** last spark, Worker_TP2 (pid 10423) **torch.
  OutOfMemoryError: CUDA out of memory. Tried to allocate
  200.00 MiB. GPU 2 total 31.45 GiB, free 189.19 MiB, process
  29.67 GiB... 2.35 GiB private pools (CUDA Graphs)** raised
  through `sparse_attn_indexer_kpool → fp8_fp4_mqa_logits`
  (deep_gemm) at multiproc_executor.py:1033 → APIServer 9336
  clean shutdown → EngineCore 9904 FORCE-KILLED; Tracebacks=34,
  500=2, EngineDead=1 marker = the 09-22-decline MEMORY class,
  NOT the 09-22 GPU-driver-lock class; HUGE-pag mmap offload
  file `/dev/hugepages/vllm_offload_<uuid>.mmap` removed at
  15:07:36. The 09-25 epoch's hb-degradation (misses at
  [22? 14:37:17]/[14:47:57]/[14:58:18]/[15:03:38] + the 2 failed
  manual-HBs) = the AUDIBLE symptom of the same GPU-memory
  exhaustion build-up: heartbeat POSTs starved inside the same
  alloc-failure windows that finally killed the engine. oom
  counter semantics RESOLVED: `cnti` counts case-insensitive
  `'OOM on device|out of memory'` lines in the RESOLVED ENGINE
  log cumulatively since log start (my 15:01 grep tested only
  the 'out of memory' half — count 0 was the HALF-TEST; the
  'OOM on device' half carried the full 0→179→244 climb).
  Kernel journal: NO OOM-kill lines (the css_killed_work_fn
  advisories are cgroup workqueue noise, no kills align with
  the crash). The canary SURVIVED the crash (station 40421
  alive throughout) and fully documented death → restart
  window: **[15:13:39] h=000 p=1 ... ALERT: NEWBOOTLOG** (the
  user's relaunch begins — new epoch engine-20260925-151312.log,
  launcher wrapper 52850, APIServer 52994/EngineCore 53554),
  [15:18:39] p=9 shm=1 (workers up, API pre-live — SHM-STALL
  correctly gated off since prev_h≠200, startup artifact),
  **[15:23:40] h=200 hb=200** (live again), [15:28:47] hb=200
  fb=4 kv=10 serving (client re-engaged; GPUs back to
  99-100% util, 31.9/32.6 GiB each). The NEW epoch resumed the
  SAME giant-client load (two 210k-token KV-LOOKUPs at 15:30)
  — SAME PRESSURE CLASS LIVE. OOM-burst watch on the new
  epoch: oom counter must stay 0 (it rebases per new log);
  oom>0 on any NEW-epoch row = the same exhaust build-up
  returning → surfaced immediately. The user-owned protective
  knobs (NOT mine to change): --kv-cache-memory pool size,
  request concurrency/size caps on the 192.168.2.208 client.
- 2026-09-25 15:44-15:59 — **epoch-B FIRST MISS + LAUNCHER
  OOM-PREVENTION EDIT (user-approved launch-arg/env fix)**:
  new-epoch miss **[15:44:08] hb=000 SERVE-WEDGE** (single;
  prior rows 15:28/15:33/15:38 clean; cycle-145 lane
  CYCLE_ALERT) → triage probe 15:48:41: **HBC=000
  HB_LAT=20.001905** (day's third failed manual-HB: 14:44,
  15:01 epoch-A; now 15:48 epoch-B) while health=200, counters
  growing (fb 4→14, kv 10→49), **oom=0** on epoch B — the
  heartbeat-starvation class reproduces under giant-client
  load even on a fresh engine; [15:44:08] stays OPEN (single
  miss, wedge candidate; NO cluster). Canon evidence pinned:
  kv_cache_memory_bytes is PER-GPU (boot log: all 8 ranks
  "reserved 2.24 GiB" own-card after 30.53 GiB initial free);
  pool 272,745 tokens = 1.04x max giant concurrency;
  steady per-rank free ~0.67 GiB (31918/32607 MiB, 99-100%
  util) = the crashed epoch's VRAM state. **EDIT APPLIED
  (launcher, not the binary): `vllm-glm-5.3-flash-nvfp4-mtp-
  no-docker.sh` (THE real launch source via LLMGLMFND_LAUNCHER
  in `llmglmfndmtp`/fg-wrapper env; canonical grep: env
  launch-source overrides)** — (1) KV pool default
  **2400000000 → 2000000000** (+rewritten comment block w/
  per-GPU semantics, pool-token linearity, crash ledger,
  fb-trace tradeoff note); (2) **OOM-headroom guard**: emits
  loud WARNING and clips any KV_CACHE_MEMORY >2.0e9
  (opt-out: KV_HEADROOM_GUARD=0); (3) MTP VRAM note pointer
  added. NOT touched: MAX_NUM_BATCHED_TOKENS=1024 (already
  the 09-17-OOM fix, in place), PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True (already the 09-19 fix, in place),
  spec-3 MTP, cudagraph_capture_sizes [1,2,3,4], Offloading-
  Connector (kv_buffer_size 1 GB/rank GPU staging documented
  as future headroom lever, untouched). md5 e5a9a76c70b3e0c9
  384a0cbb10b3cfd3 → **a6477e0f73075d37ad0ce4290859dc2a**;
  bash -n SYNTAX_OK; pristine pre-trim copy preserved at
  tools/backup-mtp-no-docker-20260925-pre-kv-trim-md5-e5a9a76c.sh;
  mirror at tools/mirror-vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh.
  EFFECT requires the user's restart (§7 user-owned; expected:
  boot prints "reserved ~1.87 GiB per rank", GPU KV pool
  ~227k tokens, steady per-rank free ~1.0 GiB, canary
  NEWBOOTLOG rebaseline, oom counter re-zero). Remaining OOM
  levers offered, not applied: cudagraph_capture_sizes [1,2,3,4]
  → [1,2] or --enforce-eager (frees up to ~2.35 GiB/rank in
  CUDA-graph private pools — tradeoff: eager decode slower for
  3-4-concurrent traffic).
- 2026-09-25 15:44-16:03 — **epoch-B CLUSTER (4 consecutive
  hb=000 rows) — surfaced; recovery edge reached during
  triage**: rows **[15:44:08]/[15:49:29]/[15:54:49]/[16:00:09]
  all hb=000 SERVE-WEDGE** (single open [15:44:08] grew to a
  4-row cluster in 16 min — the 2+-consecutive h=000 rule
  triggered at [15:49:29]; lane cycle-146 stalled mid-cycle,
  ground re-verified directly 16:01-16:03). Shape = epoch-A's
  14:37→15:03 pre-crash cluster replayed on the fresh engine
  under the same giant-client load: h=200/p=9/health=200 the
  whole time, counters GROWING every row (fb 14→25→39→48,
  kv 49→79→112→137, apc 41→49, bytes +47-57k, ev 7458→8570
  with segments 2,3 staying 0.0), **oom=0 throughout** (GPU
  OOM class NOT engaged on epoch B). Manual-HB ledger this
  window: **000@20.0019 @15:48:41 (fail #3 of the day, opened
  the wedge candidate) → 200@13.959972 @16:03:02 (first
  success — SLOW, 70% of the 20s budget; recovery edge during
  triage)**. [15:44:08] closes on this success; [15:49:29]/
  [15:54:49]/[16:00:09] recorded as the cluster window per
  pulse rule (validated alive, contemporaneous); classification:
  weapnarm-free load-congestion wedge cluster, base-weapons
  class same as 09-22 16:57/17:13/17:19 + 09-25 14:37-15:03
  (NOT a new failure mode). kv_offload gauges read
  usage/write = 0.0/0.0 at 16:03 (gauge-vs-counter semantics
  drift continues; fb counter is the live tier-churn signal).
  Host RAM heavy as ever: 935.3/1031.7 GB used, ~96 avail.
  DECISION FRAME (user-owned, §7): restart NOW (gains the
  on-disk KV-2.0GB OOM-headroom trim + clears the wedging
  giant-context state; canary rebaselines NEWBOOTLOG) vs
  hold (recovery edge reached; canary counts any new row;
  oom=0 is the good sign but the epoch-A cluster of this
  same shape ran dead 3 min after its last row). No rerun
  needed on either path — the monitor rules already knew
  this cluster. LANE RECONCILE (cycle 146, fix-34, landed
  16:09): CYCLE_ALERT matching the direct ground VERBATIM —
  and its probe revealed a **FIFTH consecutive hb=000 row
  [16:05:30] hb=000 (fb=58 kv=166 apc=49 ev 10170.0)** that
  arrived AFTER the 16:03 manual-HB success → the recovery
  edge did NOT hold through the canary's next row; cluster
  window on epoch B = [15:44:08]→[16:05:30], 5 rows/~21 min
  (SC-146's fb lead: the giant churn keeps climbing). Per
  pulse recipe [15:44:08] closes on the 16:03 manual success
  (the marker's own successor-200 rule); the cluster stands
  LIVE pending either a clean hb=200 row or the user's
  restart.

---

## _2026-09-25 16:11-16:19 — cycle 147 reconciled; cluster N=7+
  (~32 min); probe v1.1 (rowlog-md5 anchor); PARITY
  CHECKPOINT (repo ↔ remote kit)_

  LANE RECONCILE (cycle 147, fix-35, `ses_f26aab090ffebYB3Aw
  PcvfapgU` delivered 16:19 after a stalled start; fresh
  cycle-148 lane dispatched 16:20, `ses_f269f525cffepvvO3SDne
  XD08C`): verdict matches raw rows VERBATIM — **cluster grew
  to 7 consecutive hb=000 rows [15:44:08]→[16:16:10], ~32
  min**. Lane's two probes revealed [16:10:50] hb=000 (fb=68
  kv=197 apc=50 ev=12338.0) and [16:16:10] hb=000 (fb=74
  kv=215 apc=55 ev=12412.0,78.0,78.0). No recovery edge; no
  NEWBOOTLOG; h=200/p=9/HEALTH=200 throughout; **oom=0 on all
  epoch-B rows**; envelope clean (shm=0 gpu=0 500=0 rep=0
  rmax=1); the giant churn keeps climbing (fb 58→68→74, kv
  166→197→215). seg2/3 flapping 0.0→78.0 at [16:16:10] =
  small-range tier activity, judged informational. **One
  anomaly logged: the [15:54:49] row's metric values
  (apc/bytes/ev) differed between the lane's two reads 6
  min apart (append-only log cannot rewrite history) —
  treated as channel noise (stale page-cache or in-flight
  tail read); cheap defense added (probe v1.1 prints
  ROWLOG_MD5/ROWLOG_N per probe; future cycles may show if
  it repeats).** Probe script canary_probe.sh v1 deployed
  16:07 per the user's new script-file directive (one short
  ssh per probe), **v1.1 deployed 16:20** (adds ROWLOG_
  MD5/ROWLOG_N line); both verified bash -n SYNTAX_OK;
  local source C:\Users\user\AppData\Local\Temp\opencode\
  canary_probe.sh, remote /home/user/canary_probe.sh;
  PROBE_MANUAL_HB=1 variant for manual-HB triage + kv_offload
  metrics + free -m.
  PARITY CHECKPOINT (commit f1e6957, 16:2x): the user
  ratified the repo contract THIS session — repo tracks
  what is DEPLOYABLE (launchers, env/arg config, engine
  patches + patched-files, kit docs, monitor kit); runtime/
  diagnostic surface is excluded by design (swap_diag/,
  engine logs, pid files, kvoff-server.log, tp2pp4/tp4pp2
  logs, kv-block/kvoff reports, README.md.bak doc-snapshot);
  frozen-wedge logs/canary stays unreadable remotely;
  symlinks skipped; .git/.slim/.gitignore-vllm-bin local
  infra unchanged. **swap_diag/** = diagnostic-only
  decode-swap trace logs (decode_swap_diag.py output) —
  excluded per the user's rule. Sync mechanics: remote
  side tar-pack (parity_pack.sh, nice'd) → scp → selective
  extract → **md5-verified vs remote manifest (615 files;
  staged 47, mismatches 0)** → additive copy into the repo
  (identity paths; no /MIR, never deletes). Delta landed:
  vllm-glm-5.3-flash-nvfp4-mtp-no-docker.sh (the user's
  reported drift = KV 2.0GB + guard edit now at identity
  path), README.md (+2 lines: 09-18 stale-hit fix row +
  docs/vllm-bin-recreation.md pointer), 7 new kit files
  (260910 launcher + five .bak config snapshots +
  patches/README-kit.md). Name collision resolved without
  clobber: remote's 19,205B patches/README.md (kit-side
  patch doc) mirrored as **patches/README-kit.md**; this
  file remains the engagement record.   Standing rule: future
  kit-file changes on the remote get mirrored + committed
  at session end; diag/runtime churn stays out of git
  status via existing .gitignore lines (swap_diag, /logs/*,
  pids, reports). Engine restart (KV 2.0GB trim) still
  user-pending; cluster LIVE at commit time (7 rows).

- 2026-09-25 16:21-16:26 — **CLUSTER-7 NATURAL CLOSE — hb=200
  recovery edge reached at [16:21:11], no restart fired**:
  newest row `[16:21:11] h=200 p=9 500=0 oom=0 shm=0 gpu=0
  hb=200 fb=82 kv=241 rmax=1 rep=0 apc=55 bytes=+56537
  ev=12412.0,734.0,197.0` — clean (no SERVE-WEDGE tag), ev
  segments resumed small tier activity (78→734/197). Epoch-B
  cluster window closes: **[15:44:08]→[16:16:10], 7 rows /
  ~32 min, all oom=0, envelope clean, engine serving the
  whole time** (giant churn drained: fb 74→82 while hb
  latency fell back under budget). Classification unchanged:
  load-congestion wedge class (09-22 + 09-25 14:37-15:03 same
  shape). KV-trim restart still user-pending and NOT involved
  in this recovery; no NEWBOOTLOG (p=9, station 40421
  continuous). Closure per pulse rules: cluster ends on
  first clean hb=200 row — open rows [15:49:29]…[16:16:10]
  close on [16:21:11].
  CYCLE-148 bookkeeping: lane `ses_f269f525cffepvvO3SDneXD08C`
  BOARD=stopped-without-terminal-result while task_result
  still reports running — never waited on it; cycle judged on
  DIRECT ground verbatim (probe 16:26:06, newest row
  [16:21:11], 5-min freshness ✓). Probe v1.1 anchor first
  read: ROWLOG_MD5=833dd92ab2f82c69dee7e703503e5462
  ROWLOG_N=133; the [15:54:49] row reads apc=41/+47447/7866
  in the freshest tail — matches cycle-147's second read
  (the apc=49/8570 values from its first read are settled as
  channel noise / stale-cache artifact; anomaly CLOSED).
  Cycle-149 fresh lane dispatched (wounded session never
  reused).
  LATE LANE DELIVERY (cycle 148, `ses_f269f525cffepvvO3SDneX
  D08C`, board finally flips completed 16:30): terminal
  reconciles VERBATIM with the earlier direct judgment AND
  adds ground — probe#2 (NOW=16:28:29) new clean row
  **[16:26:11] h=200 hb=200 fb=82 kv=243 rmax=1 rep=0 apc=73
  bytes=+56603 ev=12412.0,4366.0,1145.0** — recovery edge
  HOLDS (two consecutive clean hb=200 rows post-cluster);
  tier segments 3 resumed higher activity (734/197→4366/1145)
  without re-wedging. Second anchor read: ROWLOG_MD5
  ba3c48e1648be9376cf5bd836aead006 ROWLOG_N=134 (append-only
  consistent +1 row — anchor methodology verified across
  probes). Station 40421 alive both probes; HEALTH=200;
  no NEWBOOTLOG (epoch B persists; KV-trim restart still
  user-pending).
  CYCLE-149 RECONCILE (fix-37, delivered 16:33): CYCLE_OK
  matches raw VERBATIM — **third consecutive clean hb=200
  row [16:31:12] h=200 hb=200 fb=82 kv=243 rmax=0 rep=0
  apc=83 bytes=+28208 ev=12412.0,5366.0,1369.0** — sustained
  closure (3 post-cluster rows), apc still climbing 55→73→83
  on real traffic, tier churn healthy (seg3 197→1145→1369),
  envelope clean (rmax=0 fine), no NEWBOOTLOG, oom=0 epoch-B.
  Anchor chain verified append-only across three lanes:
  N 133 (16:26) → 134 (16:27-16:28) → 135 (16:32), fresh md5
  per read; the cycle-147-era stale-read shape has NOT
  recurred since v1.1. Lane transcript NOW labels drifted
  (probe reads labeled ahead of their row content) — row
  CONTENT cross-checked identical to prior lanes;   truth is
  in rows, labels are noise. Cycle-150 fresh lane dispatched.
  CYCLE-150 RECONCILE (fix-38, delivered ~16:39): CYCLE_OK
  matches raw VERBATIM — **fourth consecutive clean row
  [16:36:12] h=200 hb=200 fb=82 kv=244 rmax=0 rep=0 apc=91
  bytes=+26121 ev=12412.0,6502.0,1763.0**, no tags; anchor
  N 135→136 md5 2411eb81f5ac5d51973b94fc310cf829
  (append-only +1 ✓); station 40421 alive both reads;
  HEALTH=200; epoch-B oom=0; tail scroll-out of [15:13:39] on
  probe-2 correctly identified (not a rebase). Post-cluster
  steady state: 4 clean rows, apc 83→91, fb plateau 82,
  seg-churn 5366/1369→6502/1763. Cycle-151 fresh lane
  dispatched.
  CYCLE-151 RECONCILE (fix-39, delivered ~16:46): CYCLE_OK
  matches raw VERBATIM — **fifth consecutive clean row
  [16:41:13] h=200 hb=200 fb=82 kv=246 rmax=0 rep=0 apc=96
  bytes=+22605 ev=12412.0,7126.0,1869.0**, no tags; anchor
  N 136→137 md5 676dcf3f4df75c26170e7edc0c099367 (append-only
  +1 ✓); station alive, HEALTH=200 both reads; epoch-B oom=0;
  no NEWBOOTLOG-new (base row scrolled normally). Steady post-
  cluster cadence: 5 clean rows, apc 55→96 climbing on real
  traffic, fb plateaued 82, seg-churn growing. Cycle-152
  fresh lane dispatched.
  CYCLE-152 RECONCILE (fix-40, delivered ~16:52): CYCLE_OK
  matches raw VERBATIM — **two new clean rows [16:46:13]
  hb=200 fb=82 kv=247 rmax=1 apc=104** (probe1 N=137 anchor
  matched exactly first) and **[16:51:14] hb=200 rmax=0
  apc=104 bytes=+6861 ev=12412.0,7190.0,1890.0**; anchor
  N 137→139 (md5 f1a95e57ddabfa0a9cfb11116f9b0f28,
  append-only +2 ✓); station alive, HEALTH=200 both reads;
  oom=0 epoch-B; no NEWBOOTLOG-new. Steady state continues:
  7 post-cluster clean rows, apc 96→104, kv 247, fb
  plateaued 82. Cycle-153 fresh lane dispatched.
  CYCLE-153 RECONCILE (fix-41, delivered ~16:58): CYCLE_OK
  matches raw VERBATIM — **eighth consecutive clean row
  [16:56:14] h=200 hb=200 fb=82 kv=247 rmax=0 rep=0 apc=110
  bytes=+19218 ev=12412.0,7852.0,2076.0**, no tags; probe-1
  anchor matched exactly (N=139 f1a95e57), probe-2 N 139→140
  md5 9978677509b957f36a96b5306b99b332 (append-only +1 ✓,
  head scroll byte-identical); station alive, HEALTH=200
  both reads; oom=0 epoch-B; no NEWBOOTLOG-new. Steady
  state: 8 post-cluster clean rows, apc 104→110, kv 247.
  Cycle-154 fresh lane dispatched.
  CYCLE-154 RECONCILE (fix-42, delivered ~17:04): CYCLE_OK
  matches raw VERBATIM — **ninth consecutive clean row
  [17:01:15] h=200 hb=200 fb=82 kv=248 rmax=0 rep=0 apc=118
  bytes=+18543 ev=12412.0,7931.0,2076.0**, no tags; probe-1
  anchor matched exactly (N=140 99786775), probe-2 N 140→141
  md5 f1b807b8100f41c451e1ddba63d084c4 (append-only +1 ✓,
  head scroll byte-identical); station alive, HEALTH=200
  both reads; oom=0 epoch-B; no NEWBOOTLOG-new. Steady
  state: 9 post-cluster clean rows, apc 110→118, kv 248.
  Cycle-155 fresh lane dispatched.
- 2026-09-25 17:05 — **USER ORDER: STOP THE CANARY** (pre-
  restart). Station 40421 SIGTERM'd via canary_stop.sh
  (script-file flow: scp → CRLF-strip → run; wrapper removed
  after); KILL_SENT rc=0 → **STATION_DOWN** (kill -0 fails);
  v3.2 EXIT-trap fired on cue: rowlog appended
  `[17:05:33] ALERT: CANARY-EXIT rc=0` — clean rc=0, the
  documented death-forensics row (user-ordered kill, NOT a
  silent death; pidfile left as-is for audit). Background
  lane cycle-155 (fix-43, `ses_f267910c8ffepWRPP1JbRlkXf8`)
  cancelled the same moment (objective obsolete; board
  flagged error — read-only lane, nothing to reconcile).
  Final canary ledger state: **rowlog N=141 (md5
  f1b807b8100f41c451e1ddba63d084c4)**, all epoch-B rows
  oom=0, closing run = **9 consecutive clean hb=200 rows
  [16:21:11]→[17:01:15]** after the closed cluster-7 window;
  apc 118, fb 82, kv 248; no pending alerts.
  NO WATCHDOG during the user's restart window (user's
  explicit choice): a hung boot will NOT alert; engine
  restarts remain user-owned (§7). Post-restart verification
  plan (when the user reports boot done): §11 checklist
  inline — NEWBOOTLOG grep `kv_cache: ` expect **2000000000**
  (the KV-2.0GB trim + headroom guard active), oom/ERROR/
  Traceback greps expect 0, health=200; canary re-arm =
  the §9 restart recipe (orchestrator-only, on user
  request).
- 2026-09-25 18:30-18:41 — **USER CONFIG EDIT + RESTART →
  EPOCH C CLEAN** (`engine-20260925-183028.log`, APIServer
  pid 99870, census 9, launched ~18:30:28; canary kept
  offline during the window by the ordine above). The user
  edited the launcher themselves: md5 → **741f82c3
  f9d5da91054695b04a92e4a7** (my deployed a6477e0f
  superseded). Delta: KV_CACHE_MEMORY 2000000000 →
  **2115000000** (line 392) AND the OOM-headroom guard
  threshold re-based to match: clip >2115000000 →
  2115000000 (lines 390-391); **max_model_len 262144 →
  235000** (supersedes the §11 expectation block for
  kv_cache values). All other args unchanged (fp8, MTP-3,
  seqs 4, cudagraphs [1,2,3,4], kv_both Offloading 800 GB
  CPU tier + 1 GB/rank GPU staging, batched-tokens 1024,
  block 256, cumem, async sched, opt 3). **Boot
  verification (§11): ALL GREEN** — kv_cache: 2115000000
  bytes (fp8); ×8 ranks "Initial free memory 30.53 GiB,
  reserved 1.97 GiB memory for KV Cache" (18:32:37); new
  baseline **GPU KV cache size: 237,865 tokens, Maximum
  concurrency for 235,000 tokens per request: 1.01x**;
  ERROR=0; Traceback=0; OOM=0; stale-hit fallback=0;
  Application startup complete; HEALTH=200 with live
  traffic at 18:40 (metrics 200 + chat POST 200 +
  APC-HIT nhash=167). Epoch-C oom counter rebased to 0.
  PARITY DUTY DONE same session: user-edit fetched,
  md5-verified 741f82c3…, mirrored to repo identity path
  (5-line diff vs a6477e0f), committed with this block.
  fix-8 board-error lane = the cancelled cycle-155
  (stopped, read-only, objective obsolete — reconciled
  as expected).   Next §11 use: expect kv_cache=2115000000,
  maxlen 235000. Canary re-arm user-optional (§9).
- 2026-09-26 14:37-15:09 — **REBOOT-CYCLE FORENSICS: LACT
  inactive-on-boot + engine start-die ×2 → third start
  OK**. Boot chain: boot-2 = 09-25 12:43 → down 09-26
  03:26 (epoch C `engine-20260925-183028` died with it,
  last writes 03:23). Boot-1 = 09-26 10:47 → 14:29: epoch
  D = `engine-20260926-105944` (10:59:44→14:24:20, 5187
  lines, giant-client heavy run, EngineCore exited
  normally 14:24 pre-reboot). On boot-1 **lactd did NOT
  auto-start**; user manually `systemctl enable --now
  lactd` at 14:18:47 (journal), then rebooted ~14:29 to
  get LACT working — the enablement persisted: boot-0
  (14:32:36→now) lactd auto-ran 14:33:30 "initialized
  9 GPUs". LACT WARN on boot-0: its saved config defines
  GPU pci ids `…F318-0000:b4:00.0` + `…:59:00.0` absent
  this boot — **GPU PCI enumeration order shifted across
  reboots**; fan settings for those 2 GPUs unapplied
  until the user refreshes LACT assignments (UI action).
  **Engine start-fail ×2**: attempts 14:37:04
  (`engine-20260926-143704`) and 14:47:33
  (`engine-20260926-144733`) both died in
  `_initialize_kv_caches` with workers raising
  `RuntimeError: Worker failed with error 'module
  'torch._C._cudart' has no attribute 'cudaGetLastError''`
  → "Engine core initialization failed" (14:43:57 /
  14:54:30). Pre-death stages were clean per attempt:
  per-rank NCCL init OK, OffloadingConnector + 858.99 GB
  hugetlbfs tier file created and opened by workers, model
  load OK — the failures hit at KV-cache allocation.
  Class = **early-boot GPU/driver-library settling
  window** (box up ~5 min; LACT + NVRM concurrently
  initializing 9 controllers; first sighting in kit
  history, self-resolving — the byte-identical code path
  succeeded 2 min later). NOT kv-config (2.115e9/235000
  verified 09-25 evening), NOT hugepage pool (below),
  NOT LACT-drift (WARN-only, non-fatal).
  **Epoch E = `engine-20260926-145659` (CURRENT)**:
  14:56:59, ALIVE — census 9, APIServer pid 20336,
  HEALTH 200, ESC211 live (KV-LOOKUP stale-hit fallback
  hit=141312 budget=8 fallbacks=1 wall=0.2 at 15:09:46),
  giant client 192.168.2.208 back (POST 200), GPUs
  99-100% util / 31468 MiB used / 48-59 °C / fans
  30-80%. **Tier hygiene**: both dead attempts' hugetlbfs
  tier files auto-unlinked on executor death (disk holds
  only the live engine's `e33bd08a…mmap`, 858.99 GB;
  HugePages_Total=409600, Free=4 = engine-held, normal).
  **Mitigation for future boots: wait ≥5 min after
  reboot before the first engine launch.** No kit knobs
  changed.
- 2026-09-29 00:54-04:36 — **USER RESTART → EPOCH F +
  CANARY RE-ARM + rows 1-2 SERVE-WEDGE (load-congestion
  class), relay resumed**. Machine rebooted ~00:54 Sep 29
  (up 3:30 at 04:24); pre-reboot short logs
  `engine-20260928-234423.log` + `engine-20260929-000815.log`
  (~52 KB each = short/failed runs of the pre-reboot
  window; no letters assigned). **EPOCH F =
  `engine-20260929-010246.log`** (01:02:46, APIServer pid
  8992, census 9). §11 ALL GREEN at 04:24:21: kv_cache
  2115000000 (fp8); "reserved 1.97 GiB memory" ×8; pool
  **237,865 tokens / 1.01x for 235,000-token reqs**;
  ERROR=0 Traceback=0 OOM=0; Application startup
  complete; HEALTH=200; giant client 192.168.2.208 POST
  200s; KV-LOOKUP stale-hit fallback count 39 (~3.4 h
  into boot, ESC211 alive); launcher md5 **741f82c3…**
  unchanged (user's 09-25 edit). **CANARY RE-ARM 04:27**
  (user order): script md5 `6ba78a62…` re-verified;
  rowlog intact 142 lines through the [17:05:33]
  CANARY-EXIT stop ledger; hb.json `ab306c55…` unchanged;
  probe v1.1 md5 match local↔remote `ca30a16d…`.
  **Launch-form trap**: the inline §9-style launch ssh
  HUNG ~120 s (the `cd && nohup … &` compound's waiting
  subshell redirects only the nohup child, not its own
  std fds → holds the ssh channel open); local client
  killed — the fully-redirected, HUP-immune station
  SURVIVED. Single-instance census via script-file /proc
  scan (no self-match): 39393 = orphan launch-subshell
  (ppid=1; harmless dead-fd holder) + **39394 = the real
  station loop**; 40175/40186 were transient fork false
  matches (gone on re-scan). station.pid realigned
  **39394** (04:36). **ROWS**: row-1 `[04:27:02] h=200
  p=9 500=0 oom=0 shm=0 gpu=0 hb=000 fb=44 kv=220 rmax=1
  rep=0 apc=82 bytes=+0 ev=12412.0,15092.0,3793.0 ALERT:
  NEWBOOTLOG ALERT: SERVE-WEDGE` (rebase onto
  engine-20260929-010246; counters re-derived from the
  fresh bootlog); row-2 `[04:32:22] … hb=000 fb=57
  kv=261 rmax=1 rep=0 apc=82 bytes=+61682 … ALERT:
  SERVE-WEDGE` (traffic resumed). **Triage 04:35:52**:
  engine ALIVE+SERVING — log tail 04:35:50-51 KV-LOOKUP
  hits + POST /v1/chat/completions 200 OK, APIServer
  8992 ALIVE, census 9, health 200×2, oom=0 →
  **load-congestion SERVE-WEDGE class** (same family as
  09-22 wedges + 09-25 cluster-7; h=200/p=9/oom=0
  anchors); cluster watch continues via relay;
  escalation if cluster length crosses ~30 min or a new
  class appears. **Relay resumed**: cycle 156
  dispatched background (fix-9, fresh lane per §9;
  judge-raw discipline from here). **FUTURE ARMING
  FIX**: launcher must double-fork so ssh returns
  instantly — form: `ssh … "bash /home/user/arm_canary.sh"`
  where arm_canary.sh = `( cd /home/user/vllm-glm…-logs
  && nohup bash canary5m.sh >/dev/null 2>&1 </dev/null & )`
  (script-file; nothing after `&`; sub-fork execs the
  station, holds no channel fds).
- 2026-09-29 04:43-04:45 — **RELAY CYCLE 156 (fix-9)
  reconciled: mechanics perfect, CYCLE_ALERT = wedge
  cluster continuing**. Raw probe verbatim-judged:
  NOW=04:43:05, CANARY_ALIVE_39394 ✓ (pidfile realign
  confirmed live), HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=146 (142+4 math ✓), freshness 2 s. Epoch-F
  wedge cluster: [04:27:02] NEWBOOTLOG+SERVE-WEDGE →
  [04:32:22], [04:37:43], [04:43:03] = **4 rows ≈ 16
  min**, every row h=200 p=9 500=0 oom=0 shm=0 gpu=0;
  bytes flowing (+61,682/+64,475/+63,417 per row),
  kv 220→261→309→352, fb 44→57→67→78, ev ticking
  (12,412/16,148→17,412/4,419); NO SHM-STALL/GPU-LOCK/
  CANARY-EXIT/oom → the documented load-congestion
  SERVE-WEDGE family (giant-client burst saturation;
  historical precedent: 09-22 wedges + 09-25 cluster-7
  7 rows ≈32 min self-resolved). **Triage 04:45:19**:
  engine tail POST 200 OK + KV-LOOKUP hits (hit=142,336)
  at that instant, health 200×2, census 9, APIServer
  8992 ALIVE, station 39394 ALIVE → alive+serving,
  cluster non-fatal. **Escalation bar unchanged**:
  cluster >~30 min (≈ row [04:57]) or any new class
  (SHM-STALL/GPU-LOCK/oom/EXIT); cycle 157 dispatched
  (fresh lane), probe lands ~04:53-54 at the decision
  boundary; alert-free rows resume = automatic end of
  cluster watch.
- 2026-09-29 04:55-04:57 — **RELAY CYCLE 157 (fix-10)
  reconciled: CYCLE_OK — WEDGE CLUSTER CLOSED at
  [04:53:37]**. Raw verbatim: NOW=04:55:10,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=148 (146+2 ✓), freshness 1m33s. Recovery row
  `[04:53:37] h=200 p=9 500=0 oom=0 shm=0 gpu=0 hb=200
  fb=95 kv=457 rmax=1 rep=0 apc=86 bytes=+69093 …` — no
  tags, largest traffic of the window. Final cluster
  span: 5 tagged rows [04:27:02]→[04:48:23] = **~21.4
  min**, every row h=200/p=9/oom=0/shm=0 with bytes
  flowing (+61,682/+64,475/+63,417/+60,280), kv
  220→393→457, fb 44→95, ev ticking; NO SHM-STALL/
  GPU-LOCK/oom/EXIT/new classes → **third verified
  instance of the load-congestion SERVE-WEDGE family**
  (09-22 short wedges; 09-25 cluster-7 ≈32 min; 09-29
  epoch-F ≈21 min) — all self-resolved, zero deaths,
  zero kit changes, escalation bar never crossed.
  Relay continues: cycle 158 dispatched (fresh lane);
  watch = recovery-hold on rows ≥ [04:58] (any re-tagged
  row reopens the cluster and re-judges).
- 2026-09-29 05:06-05:08 — **RELAY CYCLE 158 (fix-11)
  reconciled: CYCLE_ALERT — WEDGE CLUSTER REOPENED,
  escalation bar crossed**. Raw verbatim: NOW=05:06:31,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=150 (148+2 ✓), freshness 2m14s. The one-row
  reprieve at [04:53:37] did NOT hold: [04:58:57]
  `hb=000 fb=106 kv=506 bytes=+65,332 SERVE-WEDGE`,
  [05:04:17] `hb=000 fb=116 kv=542 bytes=+58,198
  SERVE-WEDGE`. Epoch-F wedge family total: **7 tagged
  rows over [04:27:02]→[05:04:17] ≈ 37 min** (one 5-min
  reprieve inside) — crossed the ~30-min bar → user
  pinged. NO new classes (no SHM-STALL/GPU-LOCK/oom/
  EXIT; h=200/p=9 every row; bytes +58-69k/row).
  **Triage 05:08:31 (mid-wedge)**: engine tail
  KV-LOOKUP stale-hit fallback hit=194,560 fallbacks=2
  wall=10.3 + POST /v1/chat/completions 200 OK at that
  instant; health 200×2; census 9; APIServer 8992 +
  station 39394 ALIVE → **alive+serving**, ESC211
  active (fallback wall 8.7→10.3 s, in designed band).
  Interpretation: persistent load-congestion phase on
  the giant client's make-context burst pattern
  (235k-len iterations vs 1.01× concurrency envelope);
  NOT engine-fatal. **Relay continues: cycle 159
  dispatched**; further escalation only on REAL
  degradation (500-rise, oom>0, census drop, SHM/GPU
  class, CANARY-EXIT) or user-owned engine restart
  (§7).
- 2026-09-29 05:20 — **RELAY CYCLE 159 (fix-12)
  reconciled: CYCLE_ALERT — wedge persistence continues
  (no degradation classes → no new ping, per the
  post-ping rule)**. Raw verbatim: NOW=05:20:49,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=153 (150+3 ✓), freshness 31 s. New rows:
  [05:09:38] `hb=000 fb=127 kv=570 bytes=+53,371
  SERVE-WEDGE`, [05:14:58] `fb=134 kv=609 +55,560`,
  [05:20:18] `fb=144 kv=644 +50,260`. Family total:
  **10 tagged rows over [04:27:02]→[05:20:18] ≈ 53 min**,
  the only reprieve the single [04:53:37] row; last
  ~26 min continuously tagged. ESC211 churn visible:
  fb +8-13/window sustained (band edge; cluster-7 quiet
  band was +≤3), kv +30-40/window — heavy tier refill
  under the giant client's iteration bursts. Every row
  h=200 p=9 500=0 oom=0 shm=0 gpu=0 → serving intact;
  cycle-159's own probes found no new classes. Relay
  continues: cycle 160 dispatched (fresh lane).
  Definitions for the coming rounds: true-degradation
  triggers = 500-rise, oom>0, census≠9, SHM-STALL/
  GPU-LOCK,   CANARY-EXIT, station dead — any of those =
  immediate user ping; quieter signals = record-only.
- 2026-09-29 05:34 — **RELAY CYCLE 160 (fix-13)
  reconciled: CYCLE_ALERT — wedge phase persists
  (record-only; no degradation triggers)**. Raw
  verbatim: NOW=05:34:03, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=155 (153+2 ✓),
  freshness 3m04s. New rows: [05:25:39] `hb=000 fb=148
  kv=662 bytes=+52,932 SERVE-WEDGE`, [05:30:59]
  `hb=000 fb=158 kv=696 bytes=+55,987 SERVE-WEDGE`.
  Family: **12 tagged rows [04:27:02]→[05:30:59] ≈ 64
  min**; the only reprieve remains [04:53:37]; last 7
  rows (04:58→05:30) continuously tagged — the longest
  wedge phase of the boot history (cluster-7 was ≈32
  min total). Serving intact EVERY row: h=200 p=9
  500=0 oom=0 shm=0 gpu=0; fb +4-10/window, kv
  +18-34/window, bytes ~53-56k/row steady, ev ticking.
  Zero trigger conditions (500-rise/oom/census/SHM/
  GPU/EXIT). Cycle 161 dispatched; user already pinged
  (05:08) — silent continuation unless triggers fire.
- 2026-09-29 05:45 — **RELAY CYCLE 161 (fix-14)
  reconciled: CYCLE_ALERT — wedge oscillates: second
  reprieve row then re-tag**. Raw verbatim: NOW=05:45:16,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=157 (155+2 ✓), freshness 3m42s. New rows:
  [05:36:14] **hb=200, untagged** (`fb=162 kv=728
  bytes=+68,359` — SECOND reprieve, after [04:53:37]),
  [05:41:34] `hb=000 fb=170 kv=767 bytes=+63,842
  SERVE-WEDGE`. Family: **13 tagged rows
  [04:27:02]→[05:41:34] ≈ 74 min with 2 reprieve rows**
  — oscillation shape now: ~4-6 tagged rows → 1 clean
  row → repeat (client bursts passing through tier
  exhaustion/recovery cycles). Serving untouched EVERY
  row (h=200 p=9 500=0 oom=0 shm=0 gpu=0; bytes
  +52-69k/row; ev rising steadily). Zero true-
  -degradation triggers → record-only. Cycle 162
  dispatched.
- 2026-09-29 05:58 — **RELAY CYCLE 162 (fix-15)
  reconciled: CYCLE_ALERT — near-continuous wedge grind
  (record-only)**. Raw verbatim: NOW=05:58:17,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=160 (157+3 ✓), freshness 42 s. New rows:
  [05:46:55] `hb=000 fb=178 kv=808 bytes=+61,741
  SERVE-WEDGE`, [05:52:15] `fb=187 kv=833 +49,684`,
  [05:57:35] `fb=192 kv=884 +66,173`. Family: **16
  tagged rows [04:27:02]→[05:57:35] ≈ 90.5 min, 2
  reprieves, last 4 rows continuous (05:41→05:57)** —
  oscillation has tightened to a near-continuous grind
  under the giant client's iteration load. Serving
  intact EVERY row (h=200 p=9 500=0 oom=0 shm=0 gpu=0;
  bytes ~50-66k/row steady; ev rising cumulatively);
  zero true-degradation triggers. Cycle 163
  dispatched.
- 2026-09-29 06:08 — **RELAY CYCLE 163 (fix-16)
  reconciled: CYCLE_OK — 17th wedge row then third
  reprieve**. Raw verbatim: NOW=06:08:21,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=162 (160+2 ✓), freshness 25 s. New rows:
  [06:02:56] `hb=000 fb=201 kv=926 bytes=+58,621
  SERVE-WEDGE` (17th tagged), [06:07:56] `hb=200
  untagged fb=207 kv=960 bytes=+61,058` (THIRD
  reprieve). Family: **17 tagged rows
  [04:27:02]→[06:02:56] ≈ 96 min**; reprieves now show
  a rough cadence ([04:53:37], [05:36:14], [06:07:56]
  ≈ one clean row every ~32-43 min between grinds).
  Serving intact EVERY row (h=200 p=9 500=0 oom=0
  shm=0 gpu=0; bytes ~50-69k/row); fb→207, kv→960
  (ESC211 tier absorbing); zero true-degradation
  triggers. Cycle 164 dispatched.
- 2026-09-29 06:17 — **RELAY CYCLE 164 (fix-17)
  reconciled: CYCLE_ALERT — grind resumes as forecast**.
  Raw verbatim: NOW=06:17:19, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=163 (162+1 ✓),
  freshness 4m02s. New row: [06:13:17] `hb=000 fb=213
  kv=1005 bytes=+63,441 SERVE-WEDGE` (18th tagged; the
  [06:07:56] reprieve held exactly one cycle as the
  cadence implied). Family: **18 tagged rows
  [04:27:02]→[06:13:17] ≈ 106 min, 3 reprieves**.
  Serving intact EVERY row (h=200 p=9 500=0 oom=0
  shm=0 gpu=0; bytes ~63k/row; fb→213, kv→1,005 —
  ESC211 tier still absorbing without failures); zero
  true-degradation triggers. Cycle 165 dispatched.
- 2026-09-29 06:27 — **RELAY CYCLE 165 (fix-18)
  reconciled: CYCLE_OK — fourth reprieve; cadence
  compressing**. Raw verbatim: NOW=06:27:15,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=165 (163+2 ✓), freshness 3m37s. New rows:
  [06:18:37] `hb=000 fb=217 kv=1036 bytes=+68,271
  SERVE-WEDGE` (19th tagged), [06:23:38] `hb=200
  untagged fb=222 kv=1075 bytes=+59,476` (FOURTH
  reprieve, apc jumped to 138/125 → cache refill
  active). Family: 19 tagged rows
  [04:27:02]→[06:18:37] ≈ 112 min; **reprieve
  intervals 43 min → 32 min → 16 min — recoveries
  arriving faster** (engine bursts absorbed quicker or
  giant-client iteration cadence shifting). Serving
  intact EVERY row (h=200 p=9 500=0 oom=0 shm=0
  gpu=0); zero true-degradation triggers. Cycle 166
  dispatched.
- 2026-09-29 06:40 — **RELAY CYCLE 166 (fix-19)
  reconciled: CYCLE_ALERT — first 2-row clean stretch,
  then grind resumes**. Raw verbatim: NOW=06:40:46,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=168 (165+3 ✓), freshness 73 s. New rows:
  [06:28:52] `hb=200 untagged` (`fb=227 kv=1099
  bytes=+63,030` — FIRST 2-row clean stretch together
  with [06:23:38]), [06:34:12] `hb=000 fb=238 kv=1136
  +53,733 SERVE-WEDGE` (20th tagged), [06:39:33]
  `hb=000 fb=251 kv=1177 +58,453 SERVE-WEDGE` (21st
  tagged). Family: **21 tagged rows
  [04:27:02]→[06:39:33] ≈ 132 min, 5 reprieves**.
  ESC211 churn heaviest this boot: fb +11-16/window,
  kv +38-78/window in the window; apc refill
  138→162→164 during the clean stretch (cache
  recovery). Serving intact EVERY row (h=200 p=9
  500=0 oom=0 shm=0 gpu=0); zero true-degradation
  triggers. Cycle 167 dispatched.
- 2026-09-29 06:53 — **RELAY CYCLE 167 (fix-20)
  reconciled: CYCLE_ALERT — grind continues after the
  clean stretch**. Raw verbatim: NOW=06:53:37,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=170 (168+2 ✓), freshness 3m24s. New rows:
  [06:44:53] `hb=000 fb=261 kv=1218 bytes=+58,925
  SERVE-WEDGE` (22nd tagged), [06:50:13] `hb=000
  fb=273 kv=1261 bytes=+55,602 SERVE-WEDGE` (23rd).
  Family: **23 tagged rows [04:27:02]→[06:50:13] ≈
  143 min, 5 reprieves**; current grind = 4
  consecutive rows (06:34→06:50). Serving intact
  EVERY row (h=200 p=9 500=0 oom=0 shm=0 gpu=0;
  bytes ~50-63k/row; ev rising cumulatively); zero
  true-degradation triggers. Cycle 168 dispatched.
- 2026-09-29 07:08 — **RELAY CYCLE 168 (fix-21)
  reconciled: CYCLE_ALERT — longest continuous grind
  of the boot**. Raw verbatim: NOW=07:08:38,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=173 (170+3 ✓), freshness 2m24s. New rows:
  [06:55:34] `hb=000 fb=286 kv=1306 bytes=+58,121`,
  [07:00:54] `fb=300 kv=1343 +49,933`, [07:06:14]
  `fb=311 kv=1379 +52,224` — all SERVE-WEDGE. Current
  grind = **7 consecutive tagged rows
  (06:34:12→07:06:14, ~31 min)** — now equal to
  epoch-B cluster-7's continuous length; largest
  grind this boot. Family: **26 tagged rows
  [04:27:02]→[07:06:14] ≈ 159 min, 5 reprieves**.
  fb→311 (+11-13/window), kv→1379 (+36-52/window).
  Serving unchanged (h=200 p=9 500=0 oom=0 shm=0
  gpu=0; bytes ~50-58k/row). Zero triggers —
  record-only. Cycle 169 dispatched.
- 2026-09-29 07:23 — **RELAY CYCLE 169 (fix-22)
  reconciled: CYCLE_ALERT — grind sets a new
  continuous-length record (10 rows, ≈48 min)**.
  Raw verbatim: NOW=07:23:34, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=176 (173+3 ✓),
  freshness 78 s. New rows: [07:11:35] `hb=000
  fb=321 kv=1419 +51,574`, [07:16:55] `fb=327
  kv=1453 +54,533`, [07:22:16] `fb=336 kv=1489
  +56,440` — all SERVE-WEDGE. Current grind =
  **10 consecutive tagged rows (06:34:12→07:22:16,
  ≈48 min)** — first continuous block to EXCEED
  epoch-B cluster-7 (7 rows ≈32 min). Family: **29
  tagged rows [04:27:02]→[07:22:16] ≈ 175 min, 5
  reprieves**. fb→336, kv→1489. Serving unchanged in
  EVERY row (h=200 p=9 500=0 oom=0 shm=0 gpu=0).
  Zero triggers — record-only. Cycle 170 dispatched.
- 2026-09-29 07:33 — **RELAY CYCLE 170 (fix-23)
  reconciled: CYCLE_ALERT — record grind broke,
  reprieve held one row, grind resumed**. Raw
  verbatim: NOW=07:33:33, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=178 (176+2 ✓),
  freshness 37 s. New rows: [07:27:35] `hb=200
  untagged` (`fb=349 kv=1536 bytes=+59,917` — SIXTH
  reprieve; apc refill 166→183), [07:32:56] `hb=000
  fb=356 kv=1589 bytes=+70,512 SERVE-WEDGE` (30th
  tagged; grind resumed right after). The record
  10-row grind (06:34→07:22) ended via the
  single-row reprieve at [07:27:35]. Family: **30
  tagged rows [04:27:02]→[07:32:56] ≈ 186 min, 6
  reprieves**. kv +53/window (heaviest single-step
  climb). Serving untouched in EVERY row (h=200 p=9
  500=0 oom=0 shm=0 gpu=0). Zero triggers —
  record-only. Cycle 171 dispatched.
- 2026-09-29 09:25 — **RELAY CYCLE 171 (fix-24)
  reconciled: CYCLE_ALERT — idle-tail regime**. Null
  gap: no relay cycles ran 07:33→09:25 (session
  idle); rowlog collected it all. Raw verbatim:
  NOW=09:25:34, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=199, freshness 30 s. NEW
  SIGNATURE: giant client's bursts completed; wedge
  family entered idle-tail block — 16 shown rows
  [08:04:58]→[09:25:04] ALL `hb=000` tagged
  SERVE-WEDGE, rmax=0, bytes collapsed to +5,343
  (+8,582/+42,130 early exceptions), kv=1,648 /
  fb=372 / apc=185 / ev-middles FROZEN (20,763
  since [08:10:18]) — tag churn idle since
  [08:37:00]. Unseen rows 07:38→08:04 bridged+
  (last seen [07:32:56] `hb=000 fb=356 kv=1589
  +70,512`; first seen here [08:04:58] `fb=363
  kv=1615 +5,343` — bursts landed between). Same
  shape as epoch-C cluster-8 idle tail. Serving
  counters PERFECT in every row (h=200 p=9 500=0
  oom=0 shm=0 gpu=0). Zero critical triggers —
  record-only. Cycle 172 dispatched.
- 2026-09-29 09:34 — **RELAY CYCLE 172 (fix-25)
  reconciled: CYCLE_ALERT — idle tail ended, new
  burst started**. Raw verbatim: NOW=09:34:19,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=200 (milestone), freshness 3m55 s. New
  row: [09:30:24] `hb=000 fb=383 kv=1685 rmax=1
  apc=186 bytes=+51,706 SERVE-WEDGE` — burst
  activity RESUMED after the 15-row all-tagged idle
  block ([08:04:58]→[09:25:04]): fb +11, kv +37,
  bytes +51,706 (real convoy again), ev middles
  unfroze (82,771→84,475, third 20,763→21,183).
  Family: idle block 15 rows; [09:30:24] opens the
  next grind. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  173 dispatched.
- 2026-09-29 09:44 — **RELAY CYCLE 173 (fix-26)
  reconciled: CYCLE_OK — burst grind continues (3
  consecutive tagged rows)**. Raw verbatim:
  NOW=09:44:43, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=202, freshness 3m38 s. New
  rows: [09:35:45] `hb=000 fb=394 kv=1719 rmax=1
  apc=189 bytes=+51,582 SERVE-WEDGE`, [09:41:05]
  `fb=405 kv=1752 rmax=1 apc=190 bytes=+48,198
  SERVE-WEDGE` — steady absorption (+11 fb / +33-34
  kv per row; ev middles 85,755). Family tail:
  burst grind 3 rows [09:30:24]→[09:41:05]; idle
  block 15 rows [08:04:58]→[09:25:04]; total tagged
  since epoch-F start ≈ 50. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0 shm=0 gpu=0).
  Zero true-degradation triggers — record-only.
  Cycle 174 dispatched.
- 2026-09-29 09:56 — **RELAY CYCLE 174 (fix-27)
  reconciled: CYCLE_OK — burst grind reaches 5
  consecutive rows**. Raw verbatim: NOW=09:56:02,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=204, freshness 4m16 s. New rows:
  [09:46:26] `hb=000 fb=413 kv=1782 rmax=1 apc=195
  bytes=+58,685 SERVE-WEDGE`, [09:51:46] `fb=425
  kv=1820 rmax=1 apc=202 bytes=+58,640
  SERVE-WEDGE`. Burst grind = **5 consecutive rows
  ([09:30:24]→[09:51:46])**; total tagged since
  epoch-F start ≈ 52. apc re-climbing (190→195→202
  — cache churn heavy). Serving untouched in EVERY
  row (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  175 dispatched.
- 2026-09-29 10:04 — **RELAY CYCLE 175 (fix-28)
  reconciled: CYCLE_OK — seventh reprieve; burst
  grind broke at 6 rows**. Raw verbatim:
  NOW=10:04:35, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=206, freshness 2m28 s.
  New rows: [09:57:06] `hb=000 fb=437 kv=1862
  rmax=1 apc=202 bytes=+53,274 SERVE-WEDGE`,
  [10:02:07] `hb=200 UNTAGGED fb=443 kv=1886
  rmax=0 apc=232 bytes=+64,262` — SEVENTH
  reprieve; burst grind held 6 rows
  ([09:30:24]→[09:57:06]). apc refill 202→232.
  Total tagged since epoch-F start ≈ 53;
  reprieves: 7. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  176 dispatched.
- 2026-09-29 10:12 — **RELAY CYCLE 176 (fix-29)
  reconciled: CYCLE_ALERT — grind resumed after
  the single-row reprieve**. Raw verbatim:
  NOW=10:12:07, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=207, freshness 4m40 s.
  New row: [10:07:27] `hb=000 fb=443 kv=1915
  rmax=1 apc=254 bytes=+70,228 SERVE-WEDGE` —
  grind resumed; the seventh reprieve held just
  one row ([10:02:07]). apc climb 190→232→254
  across the burst (+64 — biggest cache churn
  yet). Family: ≈54 tagged rows since epoch-F
  start; reprieves still 7. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0 shm=0 gpu=0).
  Zero true-degradation triggers — record-only.
  Cycle 177 dispatched.
- 2026-09-29 10:20 — **RELAY CYCLE 177 (fix-30)
  reconciled: CYCLE_OK — eighth reprieve; ev top
  gauge crossed 100k**. Raw verbatim:
  NOW=10:20:44, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=209, freshness 2m56 s.
  New rows: [10:12:47] `hb=000 fb=445 kv=1940
  rmax=1 apc=271 bytes=+69,812 SERVE-WEDGE`,
  [10:17:48] `hb=200 UNTAGGED fb=451 kv=1964
  rmax=0 apc=279 bytes=+60,823` — EIGHTH
  reprieve (grind = 2 rows [10:07:27]→[10:12:47]);
  **ev middle crossed 100,000 first time
  (100,215)**. apc 254→271→279 — cache churn
  climbing steadily. Family: ≈55 tagged rows
  since epoch-F start; reprieves: 8. Serving
  untouched in EVERY row (h=200 p=9 500=0 oom=0
  shm=0 gpu=0). Zero true-degradation triggers —
  record-only. Cycle 178 dispatched.
- 2026-09-29 10:30 — **RELAY CYCLE 178 (fix-31)
  reconciled: CYCLE_ALERT — grind resumed; kv
  crossed 2,000**. Raw verbatim: NOW=10:30:10,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=211, freshness 1m41 s. New rows:
  [10:23:08] `hb=000 fb=456 kv=1984 rmax=1 apc=286
  bytes=+50,805 SERVE-WEDGE`, [10:28:29] `fb=463
  kv=2006 rmax=1 apc=296 bytes=+54,202
  SERVE-WEDGE` — **kv crossed 2,000**; ev middle
  107,043. The eighth reprieve held one row
  ([10:17:48]); current grind = 2 rows
  ([10:23:08]→[10:28:29]). Family: ≈57 tagged
  rows since epoch-F start; reprieves: 8. Serving
  untouched in EVERY row (h=200 p=9 500=0 oom=0
  shm=0 gpu=0). Zero true-degradation triggers —
  record-only. Cycle 179 dispatched.
- 2026-09-29 10:38 — **RELAY CYCLE 179 (fix-32)
  reconciled: CYCLE_OK — two consecutive clean
  rows; oscillation gentler**. Raw verbatim:
  NOW=10:38:43, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=213, freshness 1 s. New
  rows: [10:33:32] `hb=200 UNTAGGED fb=470
  kv=2031 rmax=1 apc=303 bytes=+47,756`,
  [10:38:42] `hb=200 UNTAGGED fb=470 kv=2036
  apc=311 bytes=+39,954` — grind broke at 3
  rows ([10:23:08]→[10:28:29]); second 2-row
  clean stretch of the boot. Clean rows this
  boot: 10 (two 2-row stretches); tagged ≈ 58.
  ev middle 116,423 (climbing steeply); apc
  303→311. Post-10:00 regime: grind blocks
  shortened to 2-3 rows with clean pairs
  between. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  180 dispatched.
- 2026-09-29 10:45 — **RELAY CYCLE 180 (fix-33)
  reconciled: CYCLE_OK — clean stretch extended to
  3 rows**. Raw verbatim: NOW=10:45:36,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=214, freshness 1m40 s. New row:
  [10:43:56] `hb=200 UNTAGGED fb=470 kv=2044
  rmax=1 apc=328 bytes=+63,526` — clean stretch
  [10:33:32]→[10:43:56] = **3 rows** (first
  3-row clean stretch of the boot). Clean rows
  total 11; tagged ≈ 58. apc 311→328 (refill
  churn during clean rows); ev middle 120,363;
  kv 2,044. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  181 dispatched.
- 2026-09-29 10:52 — **RELAY CYCLE 181 (fix-34)
  reconciled: CYCLE_OK — clean stretch extended to
  4 rows (longest of boot)**. Raw verbatim:
  NOW=10:52:02, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=215, freshness 2m54 s.
  New row: [10:49:08] `hb=200 UNTAGGED fb=470
  kv=2060 rmax=1 apc=378 bytes=+73,628` — clean
  stretch [10:33:32]→[10:49:08] = **4 rows (~16
  min)**, longest clean run of the boot. Clean
  rows total 12; tagged ≈ 58. apc 328→378 (+50
  — biggest single-step refill so far); ev
  middle 123,566; kv 2,060; fb held 470. Serving
  untouched in EVERY row (h=200 p=9 500=0 oom=0
  shm=0 gpu=0). Zero true-degradation triggers —
  record-only. Cycle 182 dispatched.
- 2026-09-29 10:58 — **RELAY CYCLE 182 (fix-35)
  reconciled: CYCLE_OK — clean stretch extended to
  5 rows (~21 min)**. Raw verbatim: NOW=10:58:32,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=216, freshness 4m23 s. New row:
  [10:54:09] `hb=200 UNTAGGED fb=470 kv=2064
  rmax=0 apc=396 bytes=+57,073` — clean stretch
  [10:33:32]→[10:54:09] = **5 rows (~21 min)**,
  longest run of the boot by both length and
  duration. Clean rows total 13; tagged ≈ 58.
  apc 378→396; ev middle 126,886; kv 2,064; fb
  held 470 (third consecutive fb-flat clean row —
  no fallback waves; ESC211 fully absorbed).
  Serving untouched in EVERY row (h=200 p=9
  500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  183 dispatched.
- 2026-09-29 11:03 — **RELAY CYCLE 183 (fix-36)
  reconciled: CYCLE_OK — clean stretch extended
  to 6 rows (~26 min)**. Raw verbatim:
  NOW=11:03:58, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=217, freshness 4m48 s.
  New row: [10:59:10] `hb=200 UNTAGGED fb=470
  kv=2064 rmax=0 apc=403 bytes=+13,299` (light
  traffic; ev middle plateauing ~126,900) —
  clean stretch [10:33:32]→[10:59:10] = **6
  rows (~26 min)**. Clean rows total 14; tagged
  ≈ 58. fb flat at 470 for five consecutive
  clean rows — ESC211 deep-quiet. Serving
  untouched in EVERY row (h=200 p=9 500=0 oom=0
  shm=0 gpu=0). Zero true-degradation triggers —
  record-only. Cycle 184 dispatched.
- 2026-09-29 11:09 — **RELAY CYCLE 184 (fix-37)
  reconciled: CYCLE_OK — 6-row clean stretch broke
  via one light tagged row, then clean again
  (micro-flicker regime)**. Raw verbatim:
  NOW=11:09:46, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=219, freshness 15 s.
  New rows: [11:04:30] `hb=000 fb=470 kv=2065
  rmax=1 apc=405 bytes=+10,363 SERVE-WEDGE` (the
  6-row clean stretch broke here), [11:09:31]
  `hb=200 UNTAGGED fb=470 kv=2066 apc=408
  +13,139` (immediate recovery). Tagged ≈ 59;
  clean rows total 15. The regime is now
  micro-flickering on tiny loads (+10-13k bytes;
  kv/fb flat; ev middle creeping 126,946→127,506).
  Serving untouched in EVERY row (h=200 p=9
  500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  185 dispatched.
- 2026-09-29 11:15 — **RELAY CYCLE 185 (fix-38)
  reconciled: CYCLE_OK — second consecutive clean
  row after the flicker**. Raw verbatim:
  NOW=11:15:08, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=220, freshness 37 s.
  New row: [11:14:31] `hb=200 UNTAGGED fb=470
  kv=2066 rmax=0 apc=415 bytes=+12,992` — clean
  pair [11:09:31]+[11:14:31] after the
  [11:04:30] flicker. Tagged ≈ 59; clean rows
  total 16. Micro-flicker regime continues
  (kv/fb flat; ev middle 127,542; traffic
  idle-ish +13k). Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  186 dispatched.
- 2026-09-29 11:20 — **RELAY CYCLE 186 (fix-39)
  reconciled: CYCLE_OK — third consecutive clean
  row after the flicker**. Raw verbatim:
  NOW=11:20:33, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=221, freshness 61 s.
  New row: [11:19:32] `hb=200 UNTAGGED fb=470
  kv=2067 rmax=0 apc=421 bytes=+14,889` — clean
  trio [11:09:31]→[11:19:32] after the
  [11:04:30] flicker. Tagged ≈ 59; clean rows
  total 17. ev plateau ~127,574; kv/fb flat;
  traffic light +15k. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0 shm=0
  gpu=0). Zero true-degradation triggers —
  record-only. Cycle 187 dispatched.
- 2026-09-29 11:25 — **RELAY CYCLE 187 (fix-40)
  reconciled: CYCLE_OK — clean streak now 4 rows
  since the flicker**. Raw verbatim:
  NOW=11:25:57, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=222, freshness 1.4 min.
  New row: [11:24:32] `hb=200 UNTAGGED fb=470
  kv=2067 rmax=0 apc=428 bytes=+13,294` — clean
  streak [11:09:31]→[11:24:32] = 4 rows since
  the [11:04:30] flicker. Tagged ≈ 59; clean
  rows total 18. ev plateau ~127,600; kv/fb
  rock-flat; traffic light +13k. Serving
  untouched in EVERY row (h=200 p=9 500=0 oom=0
  shm=0 gpu=0). Zero true-degradation triggers —
  record-only. Cycle 188 dispatched.
- 2026-09-29 11:31 — **RELAY CYCLE 188 (fix-41)
  reconciled: CYCLE_OK — clean streak extended to
  5 rows (~20 min)**. Raw verbatim:
  NOW=11:31:45, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=223, freshness 2m12s.
  New row: [11:29:33] `hb=200 UNTAGGED fb=470
  kv=2068 rmax=0 apc=433 bytes=+16,218` — clean
  streak [11:09:31]→[11:29:33] = 5 rows (~20
  min) since the [11:04:30] flicker. Tagged ≈
  59; clean rows total 19. ev middle 128,270
  (slow creep); kv/fb flat; traffic light +16k.
  Serving untouched in EVERY row (h=200 p=9
  500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only. Cycle
  189 dispatched.
- 2026-09-29 11:37 — **RELAY CYCLE 189 (fix-42)
  reconciled: CYCLE_OK — clean streak extended
  to 6 rows (~25 min)**. Raw verbatim:
  NOW=11:37:33, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=224, freshness 2m59s.
  New row: [11:34:34] `hb=200 UNTAGGED fb=470
  kv=2070 rmax=0 apc=438 bytes=+18,136` — clean
  streak [11:09:31]→[11:34:34] = 6 rows (~25
  min) since the [11:04:30] flicker. Tagged ≈
  59; clean rows total 20. ev middle 128,950
  (slow creep); kv/fb flat; traffic light
  +18k. Serving untouched in EVERY row (h=200
  p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only.
  Cycle 190 dispatched.
- 2026-09-29 11:43 — **RELAY CYCLE 190 (fix-43)
  reconciled: CYCLE_OK — clean streak extended
  to 7 rows (~30 min)**. Raw verbatim:
  NOW=11:43:33, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=225, freshness 4m59s.
  New row: [11:39:34] `hb=200 UNTAGGED fb=470
  kv=2071 rmax=0 apc=448 bytes=+21,029` — clean
  streak [11:09:31]→[11:39:34] = 7 rows (~30
  min) since the [11:04:30] flicker. Tagged ≈
  59; clean rows total 21. ev middle 129,078
  (slow creep); kv/fb flat; traffic light
  +21k. Serving untouched in EVERY row (h=200
  p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only.
  Cycle 191 dispatched.
- 2026-09-29 11:49 — **RELAY CYCLE 191 (fix-44)
  reconciled: CYCLE_OK — clean streak extended
  to 8 rows (~35 min)**. Raw verbatim:
  NOW=11:49:00, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=226, freshness 4m25s.
  New row: [11:44:35] `hb=200 UNTAGGED fb=470
  kv=2073 rmax=1 apc=453 bytes=+15,708` —
  clean streak [11:09:31]→[11:44:35] = 8 rows
  (~35 min) since the [11:04:30] flicker.
  Tagged ≈ 59; clean rows total 22. ev middle
  129,878 (slow creep); kv/fb flat; traffic
  light +16k. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only.
  Cycle 192 dispatched.
- 2026-09-29 11:55 — **RELAY CYCLE 192 (fix-45)
  reconciled: CYCLE_OK — 8-row clean streak
  snapped by a light single-row flicker;
  instant recovery**. Raw verbatim:
  NOW=11:55:36, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=228, freshness 40s.
  Two new rows: [11:49:55] `hb=000
  SERVE-WEDGE fb=470 kv=2074 rmax=1 apc=457
  bytes=+13,391` (h=200 serving continued;
  second micro-flicker of the regime, after
  [11:04:30]) and [11:54:56] `hb=200 UNTAGGED
  fb=470 kv=2075 rmax=0 apc=457 bytes=+18,332`
  — immediate recovery; current streak = 1
  row. Regime streak record: 8 rows (~35
  min). Tagged ≈ 60; clean rows total 23. ev
  middle 130,718; kv/fb flat; traffic light
  both rows (+13k/+18k). Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0 shm=0
  gpu=0). Zero true-degradation triggers —
  record-only. Cycle 193 dispatched.
- 2026-09-29 12:01 — **RELAY CYCLE 193 (fix-46)
  reconciled: CYCLE_OK — clean streak rebuilt
  to 2 rows post-flicker**. Raw verbatim:
  NOW=12:01:26, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=229, freshness ~90s.
  New row: [11:59:56] `hb=200 UNTAGGED fb=470
  kv=2078 rmax=0 apc=460 bytes=+18,431` —
  current streak [11:54:56]→[11:59:56] = 2
  rows since the [11:49:55] micro-flicker.
  Tagged ≈ 60; clean rows total 24. ev middle
  131,290 (creep resumes); kv/fb flat; traffic
  light +18k. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only.
  Cycle 194 dispatched.
- 2026-09-29 12:06 — **RELAY CYCLE 194 (fix-47)
  reconciled: CYCLE_OK — clean streak extended
  to 3 rows post-flicker**. Raw verbatim:
  NOW=12:06:52, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=230, freshness ~2m.
  New row: [12:04:57] `hb=200 UNTAGGED fb=470
  kv=2078 rmax=0 apc=467 bytes=+14,838` —
  current streak [11:54:56]→[12:04:57] = 3
  rows since the [11:49:55] micro-flicker.
  Tagged ≈ 60; clean rows total 25. ev middle
  131,322 (plateau again); kv/fb flat; traffic
  light +15k. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only.
  Cycle 195 dispatched.
- 2026-09-29 12:12 — **RELAY CYCLE 195 (fix-48)
  reconciled: CYCLE_OK — clean streak extended
  to 4 rows (~15 min) post-flicker**. Raw
  verbatim: NOW=12:12:40, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=231,
  freshness 2m43s. New row: [12:09:57]
  `hb=200 UNTAGGED fb=470 kv=2080 rmax=0
  apc=472 bytes=+18,192` — current streak
  [11:54:56]→[12:09:57] = 4 rows (~15 min)
  since the [11:49:55] micro-flicker. Tagged ≈
  60; clean rows total 26. ev middle 131,966
  (slow creep); kv/fb flat; traffic light
  +18k. Serving untouched in EVERY row (h=200
  p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only.
  Cycle 196 dispatched.
- 2026-09-29 12:18 — **RELAY CYCLE 196 (fix-49)
  reconciled: CYCLE_OK — clean streak extended
  to 5 rows (~20 min) post-flicker**. Raw
  verbatim: NOW=12:18:07, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=232,
  freshness 3m09s. New row: [12:14:58]
  `hb=200 UNTAGGED fb=470 kv=2080 rmax=0
  apc=478 bytes=+14,570` — current streak
  [11:54:56]→[12:14:58] = 5 rows (~20 min)
  since the [11:49:55] micro-flicker. Tagged ≈
  60; clean rows total 27. ev middle 132,338
  (slow creep); kv/fb flat; traffic light
  +15k. Serving untouched in EVERY row (h=200
  p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only.
  Cycle 197 dispatched.
- 2026-09-29 12:23 — **RELAY CYCLE 197 (fix-50)
  reconciled: CYCLE_OK — clean streak extended
  to 6 rows (~25 min) post-flicker**. Raw
  verbatim: NOW=12:23:45, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=233,
  freshness 3m46s. New row: [12:19:59]
  `hb=200 UNTAGGED fb=470 kv=2082 rmax=0
  apc=482 bytes=+17,365` — current streak
  [11:54:56]→[12:19:59] = 6 rows (~25 min)
  since the [11:49:55] micro-flicker (regime
  streak record: 8 rows ~35 min). Tagged ≈
  60; clean rows total 28. ev middle 132,750
  (slow creep); kv/fb flat; traffic light
  +17k. Serving untouched in EVERY row (h=200
  p=9 500=0 oom=0 shm=0 gpu=0). Zero
  true-degradation triggers — record-only.
  Cycle 198 dispatched.
- 2026-09-29 12:29 — **RELAY CYCLE 198 (fix-51)
  reconciled: CYCLE_OK — clean streak extended
  to 7 rows (~30 min) post-flicker**. Raw
  verbatim: NOW=12:29:40, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=234,
  freshness 4m41s. New row: [12:24:59]
  `hb=200 UNTAGGED fb=470 kv=2085 rmax=0
  apc=487 bytes=+18,668` — current streak
  [11:54:56]→[12:24:59] = 7 rows (~30 min)
  since the [11:49:55] micro-flicker (regime
  streak record: 8 rows ~35 min; next clean
  row ties it). Tagged ≈ 60; clean rows total
  29. ev middle 133,498 (slow creep); kv/fb
  flat; traffic light +19k. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0 shm=0
  gpu=0). Zero true-degradation triggers —
  record-only. Cycle 199 dispatched.
- 2026-09-29 12:35 — **RELAY CYCLE 199 (fix-52)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — 8-row streak TIED then SNAPPED
  by the third micro-flicker; heavier shape**.
  Raw verbatim: NOW=12:35:53,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=236, latest row 15s after NOW. Two
  new rows: [12:30:00] `hb=200 UNTAGGED fb=470
  kv=2085 rmax=1 apc=492 bytes=+15,192` (= 8
  consecutive clean rows [11:54:56]→[12:30:00],
  ~35 min — ties the regime streak record of 8
  rows, [11:09:31]→[11:44:35]) and [12:35:20]
  `hb=000 SERVE-WEDGE fb=474 kv=2099 rmax=1
  apc=501 bytes=+34,794` (third micro-flicker
  of the regime, after [11:04:30]/[11:49:55];
  heavier shape: fb fallback wave 470→474,
  biggest single-step kv climb of the regime
  +14 (2085→2099), traffic double the usual
  light rows; h=200 p=9 500=0 oom=0 fully
  intact). Current streak snapped at 8; tagged
  ≈ 61; clean rows total 29. ev middle
  135,062. Serving untouched in EVERY row;
  zero true-degradation triggers — record-only
  (escalate only if h≠200/500s/oom/census drop
  appear). Cycle 200 dispatched.
- 2026-09-29 12:45 — **RELAY CYCLE 200 (fix-53)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — third flicker did NOT recover
  in-row: 2-row tagged grind, escalating
  shape**. Raw verbatim: NOW=12:45:18,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=237, freshness 4m38s. New row:
  [12:40:40] `hb=000 SERVE-WEDGE fb=484
  kv=2139 rmax=1 apc=509 bytes=+57,834` (h=200
  p=9 500=0 oom=0 fully intact in-row) —
  back-to-back tagged rows [12:35:20]+[12:40:40]
  = first 2-row grind since the big-grind era
  ended (~07:22); escalating markers: fb
  474→484 fallback waves accelerating, kv
  2099→2139 (+40 single-step; biggest of the
  regime), heaviest traffic since burst era.
  Current grind = 2 rows (prior single-row
  flickers [11:04:30], [11:49:55] recovered
  in-row). Tagged ≈ 62; clean rows total 29.
  ev middle 137,714. Serving untouched in
  EVERY row; zero true-degradation triggers —
  record-only (epoch-F family rule stands:
  escalate if h≠200/500s/oom/census-drop
  appear). Cycle 201 dispatched.
- 2026-09-29 12:53 — **RELAY CYCLE 201 (fix-54)
  reconciled: CYCLE_OK — grind extended to 3
  rows then recovered on row 4**. Raw
  verbatim: NOW=12:53:37, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=239,
  freshness 2m34s. Two new rows: [12:46:01]
  `hb=000 SERVE-WEDGE fb=496 kv=2173 rmax=1
  apc=511 bytes=+54,019` (h=200 intact; grind
  = 3 rows [12:35:20]→[12:46:01], ~11 min —
  longest since the big-grind era) and
  [12:51:03] `hb=200 UNTAGGED fb=500 kv=2200
  rmax=1 apc=517 bytes=+62,046` — recovery;
  current streak = 1 row. Escalating shape
  continues: fb 474→484→496→500 fallback
  waves, kv 2085→2200 (+115 over 4 rows,
  ~29/row), traffic 34k→57k→54k→62k heaviest
  yet; ev middle 143,426; fb=500 = biggest
  clean-row fallback total. Tagged ≈ 63;
  clean rows total 30. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 202 dispatched.
- 2026-09-29 13:02 — **RELAY CYCLE 202 (fix-55)
  reconciled: CYCLE_OK — reprieve held one
  row, then grind resumed at heaviest load
  yet; recovery again**. Raw verbatim:
  NOW=13:02:13, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=241, freshness 31s.
  Two new rows: [12:56:23] `hb=000 SERVE-WEDGE
  fb=501 kv=2220 rmax=1 apc=529 bytes=+74,718`
  (heaviest row of the day; h=200 intact; the
  12:46 reprieve broke after one clean row)
  and [13:01:42] `hb=200 UNTAGGED fb=502
  kv=2246 rmax=1 apc=543 bytes=+72,262` —
  recovery; current streak = 1 row. Family
  oscillation continues at escalating load:
  bytes 62k→74k→72k, fb 500→501→502 grind,
  kv 2200→2246 (+46 in 2 rows), ev top crossed
  150,616. Tagged ≈ 64; clean rows total 31.
  Serving untouched in EVERY row (h=200 p=9
  500=0 oom=0); zero true-degradation
  triggers — record-only (family rule
  stands). Cycle 203 dispatched.
- 2026-09-29 13:11 — **RELAY CYCLE 203 (fix-56)
  reconciled: CYCLE_OK — clean streak
  extended to 2 rows at heavy load**. Raw
  verbatim: NOW=13:11:04, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=242,
  freshness 4m06s. New row: [13:06:58]
  `hb=200 UNTAGGED fb=502 kv=2279 rmax=1
  apc=545 bytes=+68,989` — current streak
  [13:01:42]→[13:06:58] = 2 rows; traffic
  still heavy (second-plus-heaviest row of
  the day); fb grind flattened (501→502, +1
  across 2 clean rows) while kv keeps
  climbing (+33 to 2,279); ev middle 153,744.
  Tagged ≈ 64; clean rows total 32. Serving
  untouched in EVERY row (h=200 p=9 500=0
  oom=0); zero true-degradation triggers —
  record-only (family rule stands). Cycle
  204 dispatched.
- 2026-09-29 13:20 — **RELAY CYCLE 204 (fix-57)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — 2-row clean streak snapped;
  grind back to 2 consecutive tagged rows**.
  Raw verbatim: NOW=13:20:33,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=244, freshness 2m54s. Two new rows:
  [13:12:19] `hb=000 SERVE-WEDGE fb=506
  kv=2311 rmax=1 apc=551 bytes=+67,137` and
  [13:17:39] `hb=000 SERVE-WEDGE fb=521
  kv=2355 rmax=1 apc=551 bytes=+58,693`;
  current grind = 2 rows [13:12:19]+[13:17:39]
  (h=200 p=9 500=0 oom=0 intact in every
  row). Escalating shape resumes: fb
  502→506→521, kv +76 in 2 rows (2,279→2,355,
  churn accelerating again), load band
  58k-74k; ev middle 157,896. Tagged ≈ 66;
  clean rows total 32. Serving untouched in
  EVERY row; zero true-degradation triggers —
  record-only (family rule stands). Cycle
  205 dispatched.
- 2026-09-29 13:31 — **RELAY CYCLE 205 (fix-58)
  reconciled: CYCLE_OK — grind extended to 3
  rows then recovered on row 4 (repeat of the
  12:35-12:46 pattern)**. Raw verbatim:
  NOW=13:31:20, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=246, freshness 3m05s.
  Two new rows: [13:22:59] `hb=000
  SERVE-WEDGE fb=530 kv=2385 rmax=1 apc=553
  bytes=+48,214` (h=200 intact; grind = 3
  rows [13:12:19]→[13:22:59], ~10 min — same
  shape as the 12:35-12:46 3-row grind) and
  [13:28:15] `hb=200 UNTAGGED fb=540 kv=2417
  rmax=1 apc=553 bytes=+57,627` — recovery;
  current streak = 1 row. Escalating shape
  continues: fb ascent accelerating
  (502→540, ~+10/row), kv +138 over 4 rows
  (2,279→2,417), load band 48k-67k; ev middle
  160,976. Tagged ≈ 67; clean rows total 33.
  Serving untouched in EVERY row (h=200 p=9
  500=0 oom=0); zero true-degradation
  triggers — record-only (family rule
  stands). Cycle 206 dispatched.
- 2026-09-29 13:42 — **RELAY CYCLE 206 (fix-59)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — reprieve snapped after 1
  row, grind back to 2 consecutive tagged
  rows**. Raw verbatim: NOW=13:42:36,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=248, freshness 3m40s. Two new rows,
  both tagged: [13:33:35] `hb=000 SERVE-WEDGE
  fb=545 kv=2452 rmax=1 apc=555 bytes=+54,704`
  and [13:38:56] `hb=000 SERVE-WEDGE fb=558
  kv=2486 rmax=1 apc=555 bytes=+54,862`;
  current grind = 2 rows [13:33:35]+[13:38:56]
  (h=200 p=9 500=0 oom=0 intact in every
  row). Heavy-load phase pattern solidifying:
  grind-2/3 → reprieve-1 → grind-2/3; fb
  ascent steady (+13/row, 540→558), kv +69
  over 2 rows (2,417→2,486), ev middle
  164,336; load band ~54k-74k. Tagged ≈ 69;
  clean rows total 33. Serving untouched in
  EVERY row; zero true-degradation triggers
  — record-only (family rule stands). Cycle
  207 dispatched.
- 2026-09-29 13:54 — **RELAY CYCLE 207 (fix-60)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — grind extended to 4
  consecutive tagged rows (~16 min; longest
  since the big-grind era)**. Raw verbatim:
  NOW=13:54:11, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=250, freshness 4m35s.
  Two new rows, both tagged: [13:44:16]
  `hb=000 SERVE-WEDGE fb=565 kv=2508 rmax=1
  apc=556 bytes=+52,742` and [13:49:36]
  `hb=000 SERVE-WEDGE fb=574 kv=2533 rmax=1
  apc=556 bytes=+54,588`; current grind = 4
  rows [13:33:35]→[13:49:36] ~16 min
  (h=200 p=9 500=0 oom=0 intact in every
  row). Escalating shape continues: fb
  558→574 (ascent steady ~+9/row), kv +47
  over 2 rows (2,486→2,533), load rows
  ~52k-55k; ev middle 169,524. Tagged ≈ 71;
  clean rows total 33. Serving untouched in
  EVERY row; zero true-degradation triggers
  — record-only (family rule stands). Cycle
  208 dispatched.
- 2026-09-29 14:02 — **RELAY CYCLE 208 (fix-61)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — grind extended to 6
  consecutive tagged rows (~27 min)**. Raw
  verbatim: NOW=14:02:54, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=252,
  freshness 2m37s. Two new rows, both tagged:
  [13:54:57] `hb=000 SERVE-WEDGE fb=580
  kv=2553 rmax=1 apc=557 bytes=+52,097` and
  [14:00:17] `hb=000 SERVE-WEDGE fb=582
  kv=2576 rmax=1 apc=566 bytes=+61,292`;
  current grind = 6 rows [13:33:35]→[14:00:17]
  ~27 min (h=200 p=9 500=0 oom=0 intact in
  every row) — deepening toward the
  big-grind-era blocks (epoch-F record: 10
  rows ~48 min @ 06:34-07:22). Escalating
  shape: fb 574→582, kv +43 (2,533→2,576),
  load rows 52k-61k; ev middle 173,892.
  Tagged ≈ 73; clean rows total 33. Serving
  untouched in EVERY row; zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 209
  dispatched.
- 2026-09-29 14:13 — **RELAY CYCLE 209 (fix-62)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — grind extended to 8
  consecutive tagged rows (~37 min; ties
  the big-grind-era cluster-7, closing on
  the epoch-F record 10 rows)**. Raw
  verbatim: NOW=14:13:11, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=254,
  freshness 2m13s. Two new rows, both
  tagged: [14:05:38] `hb=000 SERVE-WEDGE
  fb=593 kv=2614 rmax=1 apc=566 bytes=+52,826`
  and [14:10:58] `hb=000 SERVE-WEDGE fb=600
  kv=2638 rmax=1 apc=566 bytes=+50,921`;
  current grind = 8 rows [13:33:35]→[14:10:58]
  ~37 min (h=200 p=9 500=0 oom=0 intact in
  every row). Escalating shape: fb 582→600
  (+18/2 rows), kv +62 (2,576→2,638), apc
  flat 566, traffic eased slightly (rows
  ~51k-53k); ev middle 176,516. Tagged ≈ 75;
  clean rows total 33. Serving untouched in
  EVERY row; zero true-degradation triggers
  — record-only (family rule stands). Cycle
  210 dispatched.
- 2026-09-29 14:22 — **RELAY CYCLE 210 (fix-63)
  reconciled: CYCLE_OK — grind broke at 9
  rows (~43 min; second-longest of epoch F,
  1 short of the 10-row record); heavy
  recovery row**. Raw verbatim: NOW=14:22:55,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=256, freshness ~87s. Two new rows:
  [14:16:18] `hb=000 SERVE-WEDGE fb=608
  kv=2662 rmax=1 apc=567 bytes=+51,232` (= 9
  consecutive tagged rows
  [13:33:35]→[14:16:18] ~43 min; h=200
  intact) and [14:21:28] `hb=200 UNTAGGED
  fb=611 kv=2691 rmax=1 apc=576
  bytes=+71,037` — recovery; current streak
  = 1 row; fb=611 = biggest clean-row
  fallback faceplate yet (ESC211 tier
  absorbing steadily); heaviest row since
  13:06; ev top 181,460. Tagged ≈ 76; clean
  rows total 34. Serving untouched in EVERY
  row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 211
  dispatched.
- 2026-09-29 14:34 — **RELAY CYCLE 211 (fix-64)
  reconciled: CYCLE_OK — reprieve snapped
  after 1 row (single flicker), instant
  recovery**. Raw verbatim: NOW=14:34:25,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=258, freshness 2m36s. Two new rows:
  [14:26:48] `hb=000 SERVE-WEDGE fb=617
  kv=2714 rmax=1 apc=578 bytes=+56,324` (the
  14:21 reprieve snapped after 1 row) and
  [14:31:49] `hb=200 UNTAGGED fb=621 kv=2730
  rmax=1 apc=586 bytes=+58,074` — recovery;
  current streak = 1 row. Deep-churn level
  persists: fb 611→621 (ascent slowing,
  +4/row), kv +39 (2,691→2,730), apc 576→586;
  ev middle 187,635. Tagged ≈ 77; clean rows
  total 35. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 212
  dispatched.
- 2026-09-29 14:42 — **RELAY CYCLE 212 (fix-65)
  reconciled: CYCLE_OK — clean streak
  extended to 3 rows (~10 min); heaviest
  row of the day + biggest apc refill since
  13:01**. Raw verbatim: NOW=14:42:49,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=260, freshness 38s. Two new rows,
  both clean: [14:37:07] `hb=200 UNTAGGED
  fb=627 kv=2755 rmax=1 apc=596
  bytes=+54,657` and [14:42:11] `hb=200
  UNTAGGED fb=631 kv=2783 rmax=1 apc=622
  bytes=+73,466` (heaviest row of the day;
  apc +26 single-step) — current streak
  [14:31:49]→[14:42:11] = 3 rows ~10 min.
  Deep churn continues: fb 621→631, kv +53
  (2,730→2,783), apc 586→622; ev middle
  192,963. Tagged ≈ 77; clean rows total 37.
  Serving untouched in EVERY row (h=200 p=9
  500=0 oom=0); zero true-degradation
  triggers — record-only (family rule
  stands). Cycle 213 dispatched.
- 2026-09-29 14:50 — **RELAY CYCLE 213 (fix-66)
  reconciled: CYCLE_OK — clean streak
  extended to 4 rows (~15 min)**. Raw
  verbatim: NOW=14:50:28, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=261,
  freshness 5m17s. New row: [14:47:11]
  `hb=200 UNTAGGED fb=632 kv=2801 rmax=1
  apc=629 bytes=+58,159` — current streak
  [14:31:49]→[14:47:11] = 4 rows (~15 min)
  since the [14:26:48] flicker; fb flatten
  holding (631→632, +1), kv +18 (2,783→2,801),
  apc 622→629; ev middle 193,991. Tagged ≈
  77; clean rows total 38. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 214
  dispatched.
- 2026-09-29 14:59 — **RELAY CYCLE 214 (fix-67)
  reconciled: CYCLE_OK — 4-row streak
  snapped by a single flicker; instant
  recovery; ev top crossed 200k**. Raw
  verbatim: NOW=14:59:04, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=263,
  freshness ~85s. Two new rows: [14:52:32]
  `hb=000 SERVE-WEDGE fb=639 kv=2822 rmax=1
  apc=636 bytes=+52,647` (snapped the 4-row
  streak) and [14:57:39] `hb=200 UNTAGGED
  fb=641 kv=2833 rmax=1 apc=644
  bytes=+51,625` — recovery; current streak
  = 1 row; ev top crossed 202,251 (first
  >200k row of the engagement); apc 629→644;
  fb 641; kv 2,833. Tagged ≈ 78; clean rows
  total 39. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 215
  dispatched.
- 2026-09-29 15:07 — **RELAY CYCLE 215 (fix-68)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — single-row flicker at
  lightest row in >1.5h**. Raw verbatim:
  NOW=15:07:42, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=264, freshness 4m42s.
  New row: [15:03:00] `hb=000 SERVE-WEDGE
  fb=647 kv=2856 rmax=1 apc=651 bytes=+46,045`
  — single flicker broke the 1-row streak
  immediately; current grind = 1 row (light
  traffic; h=200 intact in-row). Deep churn
  continues: fb 641→647, kv +23 (2,833→2,856),
  apc 644→651; ev middle 203,471. Tagged ≈
  79; clean rows total 39. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 216
  dispatched.
- 2026-09-29 15:16 — **RELAY CYCLE 216 (fix-69)
  reconciled: CYCLE_OK — 2-row grind broke;
  recovery row**. Raw verbatim: NOW=15:16:00,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=266, freshness 2m39s. Two new rows:
  [15:08:20] `hb=000 SERVE-WEDGE fb=647 flat
  kv=2862 rmax=1 apc=660 bytes=+41,603`
  (grind = 2 rows [15:03:00]+[15:08:20];
  light traffic; h=200 intact) and
  [15:13:21] `hb=200 UNTAGGED fb=655 kv=2887
  rmax=0 apc=664 bytes=+43,331` — recovery;
  current streak = 1 row. Light-traffic rows
  continue (~41-46k); churn: fb flat then
  +8 (647→655), kv +31 (2,856→2,887), apc
  651→664; ev middle 207,843. Tagged ≈ 80;
  clean rows total 40. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 217
  dispatched.
- 2026-09-29 15:22 — **RELAY CYCLE 217 (fix-70)
  reconciled: CYCLE_OK — streak extended to
  2; lightest row of the day so far**. Raw
  verbatim: NOW=15:22:50, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=267,
  freshness 4m29s. New row: [15:18:21]
  `hb=200 UNTAGGED fb=655 flat kv=2891
  rmax=0 apc=670 bytes=+27,145` — current
  streak [15:13:21]→[15:18:21] = 2 rows; fb
  flat ✓ (655); kv +4 (2,887→2,891); apc
  664→670; ev middle 209,891. Load still
  light (~27k). Tagged ≈ 80; clean rows
  total 41. Serving untouched in EVERY
  row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 218
  dispatched.
- 2026-09-29 15:30 — **RELAY CYCLE 218 (fix-71)
  reconciled: CYCLE_OK — streak extended to
  4 rows (~15 min)**. Raw verbatim: NOW=
  15:30:51, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=269, freshness
  2m26s. Two new rows, both clean:
  [15:23:24] `hb=200 UNTAGGED fb=659
  kv=2902 rmax=1 apc=674 bytes=+51,504` and
  [15:28:25] `hb=200 UNTAGGED fb=659 flat
  kv=2909 rmax=1 apc=680 bytes=+55,425` —
  current streak [15:13:21]→[15:28:25] = 4
  rows (~15 min); load bounced back up
  (~51-55k) with hb=200 holding in both;
  fb flat ✓ (659×2), kv +18 (2,891→2,909),
  apc 670→680; ev middle 213,582. Tagged ≈
  80; clean rows total 43. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 219
  dispatched.
- 2026-09-29 15:37 — **RELAY CYCLE 219 (fix-72)
  reconciled: CYCLE_OK — streak extended to
  5 rows (~20 min), the longest stint since
  the grind broke at 14:21**. Raw verbatim:
  NOW=15:37:04, CANARY_ALIVE_39394, HEALTH=
  200, PROBE_EXIT=0, ROWLOG_N=270, freshness
  ~4m. New row: [15:33:25] `hb=200 UNTAGGED
  fb=659 flat kv=2914 rmax=1 apc=694 bytes=
  +56,119` — current streak [15:13:21]→
  [15:33:25] = 5 rows (~20 min) (lane claimed
  7; recounted from raw = 5); fb flat ✓
  (659×3 rows), kv +5 (2,909→2,914), apc
  680→694 (+14); ev middle 214,974. Tagged ≈
  80; clean rows total 44. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 220
  dispatched.
- 2026-09-29 15:43 — **RELAY CYCLE 220 (fix-73)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — 6-row streak snapped by a
  fresh flicker**. Raw verbatim: NOW=15:43:48,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=272, freshness ~2s. Two new rows:
  [15:38:26] `hb=200 UNTAGGED fb=661 kv=2923
  rmax=0 apc=706 bytes=+51,242` (streak
  extended to 6 rows [15:13:21]→[15:38:26],
  ~25 min = longest post-14:21 stint) and
  [15:43:46] `hb=000 SERVE-WEDGE fb=661 flat
  kv=2926 rmax=1 apc=721 bytes=+50,827` —
  snapped it; current grind = 1 row. apc
  refill rising (+27 in 2 rows, 694→721); ev
  220,385. Note: lane's echo of [14:42:11]
  shows "oor=0" — 1-char transcription slip
  (file shows oom=0; md5 unaffected meaning).
  Tagged ≈ 81; clean rows total 45. Serving
  untouched in EVERY row (h=200 p=9 500=0
  oom=0); zero true-degradation triggers —
  record-only (family rule stands). Cycle
  221 dispatched.
- 2026-09-29 15:52 — **RELAY CYCLE 221 (fix-74)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — 2-row grind; fb ladder
  resumed (+10)**. Raw verbatim: NOW=15:52:04,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=273, freshness ~3m. New row:
  [15:49:06] `hb=000 SERVE-WEDGE fb=671
  kv=2957 rmax=1 apc=724 bytes=+53,793` —
  grind = 2 consecutive rows ([15:43:46]+
  [15:49:06]); fb +10 (661→671) = first
  double-digit fallback push in the 15:00s,
  ladder resumed; kv +31 (2,926→2,957); apc
  721→724; ev middle 222,019. Tagged ≈ 82;
  clean rows total 45. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 222
  dispatched.
- 2026-09-29 16:02 — **RELAY CYCLE 222 (fix-75)
  reconciled: CYCLE_OK — 3-row grind broke;
  recovery row; fb ladder accelerating**. Raw
  verbatim: NOW=16:02:13, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=275,
  freshness 2m35s. Two new rows: [15:54:27]
  `hb=000 SERVE-WEDGE fb=674 kv=2969 rmax=1
  apc=728 bytes=+56,502` (grind = 3 rows
  [15:43:46]→[15:54:27], ~11 min) and
  [15:59:38] `hb=200 UNTAGGED fb=683 kv=2992
  rmax=1 apc=729 bytes=+52,760` — recovery;
  current streak = 1 row; recovery pushed
  fb +12 (671→683); kv +35 (2,957→2,992);
  ev middle 227,607. Tagged ≈ 83; clean
  rows total 46. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 223
  dispatched.
- 2026-09-29 16:10 — **RELAY CYCLE 223 (fix-76)
  reconciled: CYCLE_OK — single flicker +
  quick recovery; kv crossed 3,000**. Raw
  verbatim: NOW=16:10:48, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=277,
  freshness 49s. Two new rows: [16:04:58]
  `hb=000 SERVE-WEDGE fb=690 kv=3013 rmax=1
  apc=738 bytes=+57,888` (heavy flicker row;
  snapped the 1-row streak) and [16:09:59]
  `hb=200 UNTAGGED fb=692 kv=3023 rmax=0
  apc=744 bytes=+42,760` — recovery; current
  streak = 1 row; kv crossed 3,000 (2,992→
  3,023); fb ladder accelerating (683→690→
  692, +9 in 2 rows); apc +15 (729→744); ev
  middle 232,591. Tagged ≈ 84; clean rows
  total 47. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 224
  dispatched.
- 2026-09-29 16:16 — **RELAY CYCLE 224 (fix-77)
  reconciled: CYCLE_OK — streak extended to
  2; lightest bytes row of the engagement
  (+16,603)**. Raw verbatim: NOW=16:16:31,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=278, freshness 80s. New row:
  [16:15:11] `hb=200 UNTAGGED fb=692 flat
  kv=3024 rmax=0 apc=749 bytes=+16,603` —
  current streak [16:09:59]→[16:15:11] = 2
  rows; fb flat ✓ (692×2); kv +1 (3,023→
  3,024); apc +5 (744→749); stabilize
  phase: churn settling, ev middle 233,583.
  Tagged ≈ 84; clean rows total 48. Serving
  untouched in EVERY row (h=200 p=9 500=0
  oom=0); zero true-degradation triggers —
  record-only (family rule stands). Cycle
  225 dispatched.
- 2026-09-29 16:22 — **RELAY CYCLE 225 (fix-78)
  reconciled: CYCLE_OK — streak extended to
  3 rows; churn settling at lightest loads**.
  Raw verbatim: NOW=16:22:13, CANARY_ALIVE_
  39394, HEALTH=200, PROBE_EXIT=0, ROWLOG_N=
  279, freshness ~2m. New row: [16:20:11]
  `hb=200 UNTAGGED fb=692 flat kv=3025
  rmax=0 apc=754 bytes=+18,547` — current
  streak [16:09:59]→[16:20:11] = 3 rows
  (~10 min); fb flat ✓ (692×3 rows); kv +1
  (3,024→3,025); apc +5 (749→754); ev middle
  234,347; two lightest rows back-to-back
  (+16,603 / +18,547). Tagged ≈ 84; clean
  rows total 49. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 226
  dispatched.
- 2026-09-29 16:27 — **RELAY CYCLE 226 (fix-79)
  reconciled: CYCLE_OK — streak extended to
  4 rows; longest fb plateau (692×4)**. Raw
  verbatim: NOW=16:27:58, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=280,
  freshness 2m46s. New row: [16:25:12]
  `hb=200 UNTAGGED fb=692 flat kv=3027
  rmax=0 apc=759 bytes=+19,924` — current
  streak [16:09:59]→[16:25:12] = 4 rows
  (~15 min); fb flat ✓ (692×4 rows, ~20-min
  plateau); kv +2 (3,025→3,027); apc +5
  (754→759); ev middle 235,659; three
  lightest rows in the last four. Tagged ≈
  84; clean rows total 50. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 227
  dispatched.
- 2026-09-29 16:34 — **RELAY CYCLE 227 (fix-80)
  reconciled: CYCLE_OK — streak extended to
  5 rows; fb plateau reaches 692×5 (~25
  min)**. Raw verbatim: NOW=16:34:05, CANARY_
  ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=281, freshness ~4m. New row:
  [16:30:13] `hb=200 UNTAGGED fb=692 flat
  kv=3029 rmax=0 apc=763 bytes=+21,020` —
  current streak [16:09:59]→[16:30:13] = 5
  rows (~20 min); fb flat ✓ (692×5, ~25-min
  plateau); kv +2 (3,027→3,029); apc +4
  (759→763); ev middle 236,659; four
  lightest rows in the last five. Tagged ≈
  84; clean rows total 51. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 228
  dispatched.
- 2026-09-29 16:40 — **RELAY CYCLE 228 (fix-81)
  reconciled: CYCLE_OK — streak extended to
  6 rows (~25 min), ties the post-grind
  best; first kv-flat row (3,029×2); fb
  plateau 692×6 (~30 min)**. Raw verbatim:
  NOW=16:40:02, CANARY_ALIVE_39394, HEALTH=
  200, PROBE_EXIT=0, ROWLOG_N=282, freshness
  4m49s. New row: [16:35:13] `hb=200
  UNTAGGED fb=692 flat kv=3029 flat rmax=0
  apc=768 bytes=+16,579` — current streak
  [16:09:59]→[16:35:13] = 6 rows (~25 min,
  ties 15:13→15:38 best); calmest window
  since the grind broke at 14:21; kv FLAT
  ✓ (3,029 in two rows); apc +5 (763→768);
  ev middle 237,235. Tagged ≈ 84; clean
  rows total 52. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 229
  dispatched.
- 2026-09-29 16:47 — **RELAY CYCLE 229 (fix-82)
  reconciled: CYCLE_OK — calm plateau ended
  with the heaviest row of the day; recovery
  on a new fb step**. Raw verbatim: NOW=
  16:47:20, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=284, freshness ~2m.
  Two new rows: [16:40:34] `hb=000
  SERVE-WEDGE fb=693 kv=3035 rmax=1 apc=861
  bytes=+76,412` (HEAVIEST row of the day,
  snapped the 6-row streak; apc spiked +93
  to 861 — largest refill ever) and
  [16:45:35] `hb=200 UNTAGGED fb=699
  kv=3059 rmax=1 apc=885 bytes=+64,219` —
  recovery; current streak = 1 row; fb
  692→699 new ladder step; kv +30 (3,029→
  3,059); apc 768→885 (+117 total); ev
  middle 239,843. Tagged ≈ 85; clean rows
  total 53. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 230
  dispatched.
- 2026-09-29 16:53 — **RELAY CYCLE 230 (fix-83)
  reconciled: CYCLE_ALERT (newest row tagged,
  record-only) — flash flicker at new-day
  high scale; ev crossed 240k**. Raw
  verbatim: NOW=16:53:26, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=285,
  freshness ~2.5m. New row: [16:50:56]
  `hb=000 SERVE-WEDGE fb=699 flat kv=3062
  rmax=1 apc=893 bytes=+43,925` — single
  flicker after the [16:45:35] recovery;
  current grind = 1 row; kv +3 (3,059→
  3,062); fb flat at 699; apc 893 (new
  high territory); ev middle 242,991 (first
  >240k row). Tagged ≈ 86; clean rows
  total 53. Serving untouched in EVERY
  row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 231
  dispatched.
- 2026-09-29 17:02 — **RELAY CYCLE 231 (fix-84)
  reconciled: CYCLE_OK — 2-row grind broke;
  sustained heavy-load wave continues**.
  Raw verbatim: NOW=17:02:19,
  CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=287, freshness
  47s. Two new rows: [16:56:16] `hb=000
  SERVE-WEDGE fb=703 kv=3080 rmax=1
  apc=899 bytes=+61,614` (grind = 2 rows
  [16:50:56]+[16:56:16]) and [17:01:32]
  `hb=200 UNTAGGED fb=704 kv=3093 rmax=1
  apc=911 bytes=+61,441` — recovery;
  current streak = 1 row; fb ladder
  699→704; kv +31 (3,062→3,093); apc new
  highs (899→911); ev crossed 245k
  (247,536); bytes ~61k in both rows =
  sustained heavy wave since 16:40. Tagged ≈
  87; clean rows total 54. Serving
  untouched in EVERY row (h=200 p=9 500=0
  oom=0); zero true-degradation triggers —
  record-only (family rule stands). Cycle
  232 dispatched.
- 2026-09-29 17:10 — **RELAY CYCLE 232 (fix-85)
  reconciled: CYCLE_OK — streak to 2 at
  heaviest sustained loads; ev crossed
  250k**. Raw verbatim: NOW=17:10:31,
  CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=288, freshness
  ~3m48s. New row: [17:06:40] `hb=200
  UNTAGGED fb=711 kv=3118 rmax=1 apc=917
  bytes=+56,908` — current streak
  [17:01:32]→[17:06:40] = 2 rows being
  served CLEAN at the sustained heavy-wave
  scale; ev crossed 250k (250,864, new day
  high); fb ladder 704→711; kv +25 (3,093→
  3,118); apc 911→917. Note: lane's echo of
  the [16:35:13] line dropped "shm=" (echo
  slip; file row intact). Tagged ≈ 87;
  clean rows total 55. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 233
  dispatched.
- 2026-09-29 17:18 — **RELAY CYCLE 233 (fix-86)
  reconciled: CYCLE_OK — single flicker +
  recovery; escalation pace holds at high
  scale**. Raw verbatim: NOW=17:18:46,
  CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=290, freshness
  ~1m48s. Two new rows: [17:12:00] `hb=000
  SERVE-WEDGE fb=712 kv=3130 rmax=1 apc=937
  bytes=+64,472` (snapped the 2-row streak;
  ev crossed 255k) and [17:17:01] `hb=200
  UNTAGGED fb=716 kv=3145 rmax=0 apc=942
  bytes=+50,759` — recovery; current streak
  = 1 row; fb ladder 711→716; kv +27
  (3,118→3,145); apc new highs (937→942);
  ev middle 258,664 (day high); bytes
  sustained 50k-64k since 16:40. Tagged ≈
  88; clean rows total 56. Serving
  untouched in EVERY row (h=200 p=9 500=0
  oom=0); zero true-degradation triggers —
  record-only (family rule stands). Cycle
  234 dispatched.
- 2026-09-29 17:25 — **RELAY CYCLE 234 (fix-87)
  reconciled: CYCLE_OK — streak extended to
  2 rows at light load; first fb flat (716)
  of the new regime**. Raw verbatim: NOW=
  17:25:04, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=291, freshness ~3m.
  New row: [17:22:02] `hb=200 UNTAGGED
  fb=716 flat kv=3146 rmax=0 apc=947
  bytes=+27,735` — current streak
  [17:17:01]→[17:22:02] = 2 rows; fb flat ✓
  (716×2, first flat of the heavy regime);
  kv +1 (3,145→3,146); apc +5 (942→947);
  ev crossed 260k (260,432, day high);
  load light. Tagged ≈ 88; clean rows
  total 57. Serving untouched in EVERY
  row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 235
  dispatched.
- 2026-09-29 17:31 — **RELAY CYCLE 235 (fix-88)
  reconciled: CYCLE_OK — streak extended to
  3 rows; churn braking at the new scale**.
  Raw verbatim: NOW=17:31:06,
  CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=292, freshness
  ~4m. New row: [17:27:02] `hb=200 UNTAGGED
  fb=716 flat kv=3148 rmax=0 apc=951
  bytes=+19,555` (lightest since 16:35) —
  current streak [17:17:01]→[17:27:02] = 3
  rows (~10 min); fb flat ✓ (716×3); kv +2
  (3,146→3,148); apc +4 (947→951); ev
  middle 261,572. Stabilizing signature at
  the new scale mirrors the pre-reprieve
  break of 15:38→15:43. Tagged ≈ 88; clean
  rows total 58. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 236
  dispatched.
- 2026-09-29 17:38 — **RELAY CYCLE 236 (fix-89)
  reconciled: CYCLE_OK — single flicker +
  recovery at the new scale; ladder
  resumed (+12 in 2 rows)**. Raw verbatim:
  NOW=17:38:59, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=294,
  freshness ~1m36s. Two new rows: [17:32:23]
  `hb=000 SERVE-WEDGE fb=722 kv=3164 rmax=1
  apc=952 bytes=+43,478` (snapped the
  3-row streak; ev crossed 264k) and
  [17:37:23] `hb=200 UNTAGGED fb=728
  kv=3184 rmax=0 apc=957 bytes=+44,053` —
  recovery; current streak = 1 row; fb
  ladder resumed 716→728 (+12 in 2 rows);
  kv +36 (3,148→3,184); apc +5 (952→957);
  ev middle 268,120 (day high). Tagged ≈
  89; clean rows total 59. Serving
  untouched in EVERY row (h=200 p=9 500=0
  oom=0); zero true-degradation triggers —
  record-only (family rule stands). Cycle
  237 dispatched.
- 2026-09-29 17:46 — **RELAY CYCLE 237 (fix-90)
  reconciled: CYCLE_ALERT (newest row
  tagged, record-only) — flash flicker;
  ev crossed 270k**. Raw verbatim: NOW=
  17:46:19, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=295, freshness
  3m31s. New row: [17:42:44] `hb=000
  SERVE-WEDGE fb=729 kv=3190 rmax=1 apc=972
  bytes=+43,005` — single flicker after the
  [17:37:23] recovery; current grind = 1
  row; fb +1 (728→729); kv +6 (3,184→
  3,190); apc 972 (new highs); ev middle
  271,716 (first >270k row). Lane echoed
  "om=0" on the [16:45:35] line's "oom",
  then self-corrected it verbatim (echo
  slip; file row intact). Tagged ≈ 90;
  clean rows total 59. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0);
  zero true-degradation triggers —
  record-only (family rule stands). Cycle
  238 dispatched.
- 2026-09-29 17:57 — **RELAY CYCLE 238 (fix-91)
  reconciled: CYCLE_ALERT (newest row
  tagged, record-only) — grind extended to
  3 rows; ladder accelerating (+16/2 rows,
  kv +50)**. Raw verbatim: NOW=17:57:40,
  CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=297, freshness
  4m16s. Two new rows, both tagged:
  [17:48:04] `hb=000 SERVE-WEDGE fb=736
  kv=3214 rmax=1 apc=975 bytes=+50,239` and
  [17:53:24] `hb=000 SERVE-WEDGE fb=745
  kv=3240 apc=975 flat bytes=+46,630` —
  grind = 3 consecutive rows ([17:42:44]→
  [17:53:24], ~11 min); fastest fb climb
  since the 16:40 brick (736→745, +16 in 2
  rows); kv +50 (3,190→3,240); ev middle
  276,324. Tagged ≈ 92; clean rows total
  59. Serving untouched in EVERY row
  (h=200 p=9 500=0 oom=0); zero true-
  degradation triggers — record-only
  (family rule stands). Cycle 239
  dispatched.
- 2026-09-29 18:09 — **RELAY CYCLE 239 (fix-92)
  reconciled: CYCLE_OK — grind-3 broke;
  flicker + instant recovery; ladder still
  at fastest climb**. Raw verbatim: NOW=
  18:09:32, CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=300 (triplets:
  +3 rows), freshness ~30s. Three new
  rows: [17:58:39] `hb=200 UNTAGGED
  fb=753 kv=3267 apc=975 bytes=+47,756`
  (grind-3 BROKE, exactly like the 12:35/
  13:12 grind-3s that preceded grind-9);
  [18:04:00] `hb=000 SERVE-WEDGE fb=762
  kv=3295 apc=975 bytes=+42,126` (1-row
  flicker); [18:09:00] `hb=200 UNTAGGED
  fb=766 kv=3308 rmax=1 apc=977 bytes=
  +32,918` — recovery; current streak = 1
  row; fb climb sustained (+21 in 3 rows,
  engagement-fastest pace); kv +68 (3,240→
  3,308); ev middle 281,296. Lane verdict
  line had a stray "17*" typo before the
  row id (echo slip; raw row is [18:09:00]).
  Tagged ≈ 93; clean rows total 61.
  Serving untouched in EVERY row (h=200
  p=9 500=0 oom=0); zero true-degradation
  triggers — record-only (family rule
  stands). Cycle 240 dispatched.
- 2026-09-29 18:18 — **RELAY CYCLE 240 (fix-93)
  reconciled: CYCLE_OK — streak extended to
  2; ladder matching the grind-9 build
  profile (+7 fb, +5 apc per row)**. Raw
  verbatim: NOW=18:18:12, CANARY_ALIVE_
  39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=301, freshness ~4m. New row:
  [18:14:13] `hb=200 UNTAGGED fb=773
  kv=3331 rmax=1 apc=982 bytes=+46,167` —
  current streak [18:09:00]→[18:14:13] = 2
  rows served CLEAN; fb climb sustained
  (766→773), kv +23 (3,308→3,331), apc
  977→982; ev 283,936 (day high); the
  uninterrupted ladder profile now matches
  the 13:35-14:16 build that produced
  grind-9. Lane corrected its own mistyped
  [17:37:23] echo line (file row intact).
  Tagged ≈ 93; clean rows total 62.
  Serving untouched in EVERY row (h=200
  p=9 500=0 oom=0); zero true-degradation
  triggers — record-only (family rule
  stands). Cycle 241 dispatched.
- 2026-09-29 18:25 — **RELAY CYCLE 241 (fix-94)
  reconciled: CYCLE_OK — 3-row clean
  streak; ladder stalls flat at 773 (deep-
  plateau echo), apc refills to 996**. Raw
  verbatim: NOW=18:25:48, CANARY_ALIVE_
  39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=303 (triplet barrel +2), fresh-
  ness 1m22s. Three new rows, all clean:
  [18:19:18] `hb=200 UNTAGGED fb=773 flat
  kv=3332 apc=988 bytes=+30,222`; [18:24:
  26] `hb=200 UNTAGGED fb=773 flat kv=3339
  apc=996 bytes=+37,761` (+ interim [18:19:
  18] same-line dedupe — N math 301→303=+2
  net rows) — streak now 3 rows ([18:09:
  00]→[18:24:26], ~15 min); fb UNCHANGED
  back-to-back (773×2 — deepest plateau
  since 16:35-17:37 while push unaffected:
  mv 287,524→291,132, unlocked peak 72,356
  day high, grind-ready). Lane minor echo
  slip "text=NONE" noted. Tagged ≈ 93;
  clean rows total 64. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0);
  zero true-degradation triggers —
  record-only (family rule stands). Cycle
  242 dispatched.
- 2026-09-29 18:33 — **RELAY CYCLE 242 (fix-95)
  reconciled: CYCLE_ALERT (newest row
  tagged, record-only) — 3-row grind
  broke; flicker at grind-ready scale**.
  Raw verbatim: NOW=18:33:58, CANARY_
  ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=304, freshness 4m12s. New row:
  [18:29:46] `hb=000 SERVE-WEDGE fb=780
  kv=3363 rmax=0 apc=1000 bytes=+42,331` —
  streak reset to 0 (current grind = 1);
  ladder resumed after the 773-flat echo
  (773→780, +7); kv +24 (3,339→3,363);
  apc crossed the millennium mark (1000 —
  first-ever); ev middle 293,952 (first
  >290k row); load mid-heavy. Tagged ≈ 94;
  clean rows total 64. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0);
  zero true-degradation triggers —
  record-only (family rule stands). Cycle
  243 dispatched.
- 2026-09-29 18:41 — **RELAY CYCLE 243 (fix-96)
  reconciled: CYCLE_OK — grind-2 done;
  ev crossed 300k (first >300k row);
  apc refill at 1028 (record scale)**. Raw
  verbatim: NOW=18:41:56, CANARY_ALIVE_
  39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=306 (304→+2), freshness
  ~1m49s. Two new rows: [18:35:07] `hb=000
  SERVE-WEDGE fb=782 kv=3372 rmax=1 apc=
  1010 bytes=+39,463` (grind extended to
  2 rows); [18:40:07] `hb=200 UNTAGGED
  fb=783 kv=3382 rmax=0 apc=1028 bytes=
  +58,564` — grind BROKE; recovery; cur-
  rent streak = 1 row; fb climb slower
  (780→783), kv +19 (3,363→3,382), apc
  +28 (1000→1028, record refill contin-
  ues), ev 300,732 first >300k row (day
  high); grind pattern self-similar to
  the grind-9 precursors (grind-1/2/3
  repeat build). Tagged ≈ 96; clean rows
  total 65. Serving untouched in EVERY
  row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-only
  (family rule stands). Cycle 244
  dispatched.
- 2026-09-29 18:48 — **RELAY CYCLE 244 (fix-97)
  reconciled: CYCLE_OK — streak extended
  to 2; second fb-flat echo at 783×2;
  grind-ready cluster grows (5 rows in
  the [18:29+ window])**. Raw verbatim:
  NOW=18:48:24, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=
  307, freshness ~3m16s. New row: [18:45:
  08] `hb=200 UNTAGGED fb=783 flat kv=
  3389 rmax=1 apc=1044 bytes=+39,830` —
  streak now 2 rows ([18:40:07]→[18:45:08]
  served CLEAN); fb UNCHANGED (783×2 —
  second flat echo since the grind
  stretched to 2); kv +7 (3,382→3,389);
  apc +16 (1028→1044, record refill
  continues millennium+); ev 303,056;
  peak 75,355 (day high); grind-ready
  cluster now 5 rows total ([18:29:46]→
  [18:45:08]) — grind-1/grind-2/recovery/
  flat ladder pattern building toward
  grind-9-scale cluster. Tagged ≈ 96;
  clean rows total 66. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0);
  zero true-degradation triggers —
  record-only (family rule stands). Cycle
  245 dispatched.
- 2026-09-29 18:54 — **RELAY CYCLE 245 (fix-98)
  reconciled: CYCLE_OK — streak extended
  to 3; third consecutive fb-flat
  (783×3 — deepest flat run since grind-3
  broke)**. Raw verbatim: NOW=18:54:46,
  CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=308, freshness
  ~4m37s. New row: [18:50:09] `hb=200
  UNTAGGED fb=783 FLAT kv=3398 rmax=1
  apc=1062 bytes=+43,996` — streak now 3
  rows ([18:40:07]→[18:50:09], ~10 min);
  fb FLAT ×3 (783 — third consecutive,
  deepest flat run since the grind
  stretched to 2); kv +9 (3,389→3,398);
  apc +18 (1044→1062, record refill
  continues); ev middle 304,660; peak
  poly 75,808 (day high); grind-ready
  cluster now 6 rows in the [18:29+
  window] (grind-1, grind-2, recovery,
  2 flat, 3 flat — grind-9-scale build
  continues, mirrors the 13:35-14:16
  grind-9 build profile). Tagged ≈ 96;
  clean rows total 67. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0);
  zero true-degradation triggers —
  record-only (family rule stands). Cycle
  246 dispatched.
- 2026-09-29 19:03 — **RELAY CYCLE 246 (fix-99)
  reconciled: CYCLE_OK — streak extended
  to 5 rows (~20 min, ties post-14:21
  family best); flat run ended at 783×4
  with fb climb resumed; heaviest burst
  since the 16:40 brick (+74,729)**. Raw
  verbatim: NOW=19:03:09, CANARY_ALIVE_
  39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=310 (308+2), freshness ~2m38s.
  Two new rows: [18:55:21] `hb=200
  UNTAGGED fb=783 flat kv=3409 apc=1082
  bytes=+62,396` and [19:00:31] `hb=200
  UNTAGGED fb=785 kv=3457 apc=1086 bytes=
  +74,729` — streak [18:40:07]→[19:00:31]
  = 5 rows (ties the 15:13-15:38 and
  16:09-16:35 family bests); fb climb
  resumed after the 783×4 flat run (785);
  kv +48 (3,409→3,457, big jump); apc
  1082→1086 (record refill continues);
  ev 309,472 (crossed 305k, day high);
  peak poly 76,929 (day high); grind-
  ready cluster now 7 rows ([18:29:46]→
  [19:00:31]) with the grind-9-scale
  build continuing. Tagged ≈ 96; clean
  rows total 69. Serving untouched in
  EVERY row (h=200 p=9 500=0 oom=0);
  zero true-degradation triggers —
  record-only (family rule stands). Cycle
  247 dispatched.
- 2026-09-29 19:10 — **RELAY CYCLE 247 (fix-100)
  reconciled: CYCLE_OK — streak = 6 (ties
  post-14:21 family best); day-heaviest
  row of engagement (+82,825); kv
  biggest single-row jump (+66)**. Raw
  verbatim: NOW=19:10:34, CANARY_ALIVE_
  39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=311, freshness ~4m57s. New
  row: [19:05:37] `hb=200 UNTAGGED fb=786
  kv=3523 rmax=1 apc=1096 bytes=+82,825`
  — streak [18:40:07]→[19:05:37] = 6 rows
  (~25 min, ties the 15:13-15:38 and
  16:09-16:35 bests); fb +1 (785→786);
  kv +66 (3,457→3,523, largest single-row
  jump of engagement); apc +10 (1086→
  1096); ev 311,991; peak poly 77,528
  (day high); grind-ready cluster now 8
  rows ([18:29:46]→[19:05:37]) — grind-
  9-scale build continues. Tagged ≈ 96;
  clean rows total 70. Serving untouched
  in EVERY row (h=200 p=9 500=0 oom=0);
  zero true-degradation triggers —
  record-only (family rule stands). Cycle
  248 dispatched.
- 2026-09-29 19:51 — **CHAIN BREAK NOTE —
  cycle 249 (fix-102) lane died mid-wait
  (background session stopped before
  terminal result; no probe ran); wedge
  death + user restart #1 → epoch G**.
  Cycle 249 record NOT available (probe
  never executed between the cycle-248
  commit and the user report; board
  stopped-unreconciled). Direct diagnostics
  (read-only): epoch F (engine-20260929-
  010246.log, booted 01:02, 8.7 MB) last
  write 19:37 — tail shows a CLEAN SIGTERM
  shutdown at 19:37:27 (root-kit none;
  resource-tracker warnings: "8 leaked
  semaphore objects", "10 leaked shared_
  memory objects"). User restarted at
  19:40:36 → epoch G = engine-20260929-
  194036.log (APIServer pid 169166).
  Rowlog [19:42:27] NEWBOOTLOG (epoch F
  replaced); [19:47:27] h=000 shm=2
  (first non-zero shm of the engagement;
  epoch G still booting). Canary station
  pid 39394 SURVIVED the wedge death +
  both restarts. Cycle 250 dispatched.
- 2026-09-29 20:14 — **RELAY CYCLE 250 (fix-103
  lane omitted; direct orchestrator probe
  due to broken-cycle hand-off) reconciled:
  CYCLE_OK — epoch H SERVING; chain
  RESUMED**. Raw verbatim: NOW=20:14:26,
  CANARY_ALIVE_39394, HEALTH=200,
  ROWLOG_N=324, freshness ~1m. New rows:
  [19:52:48] `hb=000 SERVE-WEDGE h=200
  p=9 fb=3 kv=7 apc=1 bytes=+128,804
  ev=2672.0` (epoch G boot-ramp; fresh
  engine identity; canary hb not yet
  wrapped); [19:58:08] `hb=000 fb=12 kv=29
  apc=3 bytes=+34,188` (G ramp); [20:03:
  08] `h=000 p=4 fb=12 kv=29 apc=3` (user
  stop #2 in progress, census mid-drop);
  [20:08:09] `h=000 p=9 ALERT: NEWBOOTLOG`
  (epoch H boot detected); [20:13:26]
  `h=200 p=9 hb=200 shm=3 fb=0 kv=0
  apc=0 bytes=+71,728` (epoch H's FIRST
  live row: h=200 + hb=200 immediately —
  earliest canary hb wrap-in of the
  engagement; shm=3 boot artifact,
  expect 0 under serving).
  - latest.log tail confirms epoch H live:
  APIServer pid 176751; 8×TP rows at
  20:13:26 warn `jit_monitor.py:132 Triton
  kernel JIT compilation during inference
  ... latency spike` (_rejection_kernel/
  _resample_kernel, is_warmup=False —
  boot-order JIT, not a stall); then
  normal serving: POST 200 OK 20:13:32 +
  20:14:22, Engine 000 gen 110-125 tok/s,
  spec-decode mean acc 2.0-2.5, GPU KV
  66-69% climbing.
  - ROOT CAUSE (user smoked out; confirmed):
  the wedge is the fork's KV-LOOKUP
  stale-hit escape (scheduler.py:1393/1403)
  — `stale-hit fallback ... fallbacks=3
  wall=20.2` → `SERVERKILL ... escape
  (0, False) until boundary change`; the
  escape usually self-recovers (epoch G
  log had 4 SERVERKILL events; first two
  19:52:53/19:53:50 recovered, gen bounced
  96-130 tok/s; the 19:55:35 escape STUCK
  — only /metrics answered — until user
  stop #2 at 20:02, which per the rowlog
  p=4 window completed before H's bootlog
  at 20:03-20:08). Epoch F: 44 SERVERKILL
  events over 18.5 h — the all-day
  SERVE-WEDGE hb=000 ladder was this same
  escape cycle (fb climb to 841 / kv
  3,756 = KV-cache pressure build, not
  a separate failure mode). Both user
  stops were clean SIGTERM shutdowns;
  zero crash/OOM/oom signatures.
  - Watch continued on epoch H: shm=3
  should resettle to 0 within a cycle;
  escape events may re-fire (stochastic;
  44 in F) — next wedge-stick will show
  as a NEWBOOTLOG+user-restart or a
  multi-row hb=000 grind that doesn't
  recover. Cycle 251 dispatching.
- 2026-09-29 20:24 — **RELAY CYCLE 251 (fix-103)
  reconciled: CYCLE_OK — epoch H fully clean first
  watch; shm resettle confirmed; per-epoch counter
  reset confirmed**. Raw verbatim: NOW=20:23:48,
  CANARY_ALIVE_39394, HEALTH=200, PROBE_EXIT=0,
  ROWLOG_N=326, freshness ~17s. Two new rows, both
  untagged: [20:18:28] `h=200 p=9 shm=0 hb=200 fb=4
  kv=14 rmax=1 rep=0 apc=3 bytes=+90,050 ev=1456.0`
  and [20:23:31] `h=200 p=9 shm=0 gpu=0 hb=200 fb=4
  kv=15 apc=8 bytes=+45,769 ev=3048.0` — 3 clean
  cycles without an escape event on epoch H so far
  (20:13:26→20:23:31).
  - Epoch-H ramp: fb 0→4, kv 0→15, apc 0→8;
  bytes +135,819 in the first 10 serving min;
  **shm=3→0 settled exactly as forecast** (one
  cycle).
  - ev counter RESET at epoch boundary confirmed
  (H baseline 196 vs F's 12.4k plateau / G's
  2672 ramp) — serving stats do not survive
  restarts; epoch H starts a blank quantitative
  baseline, not a regression.
  - Canary pid 39394 survived ALL THREE epoch
  transitions (wedge deaths 19:37/19:55 +
  reboots 19:40:36/20:02).
  - Investigation closing facts: engine imports
  the venv's own site-packages vllm
  (0.1.dev20051+g487ecf187, patched ESC211 code
  present — 14 matches); ALL ESC211 knobs at
  launch defaults (max_lookups=8, pin_seconds=
  10, fallback_limit=3, ttl=150s, repeat_window=
  300s, escape_relax_full_fit=on, mirror=1,
  tombstones=8M, hugetlb tier on) — no user env
  pre-sets ever tuned. Tagged ≈ 99; epoch-H
  clean rows 3. Cycle 252 dispatched.
- 2026-09-29 20:32 — **RELAY CYCLE 252 (fix-104)
  reconciled: CYCLE_OK — epoch H clean streak
  reaches 4 rows; tier recovery pattern **. Raw
  verbatim: NOW=20:31:16, CANARY_ALIVE_39394,
  HEALTH=200, PROBE_EXIT=0, ROWLOG_N=327.
  New row: [20:28:31] `h=200 p=9 shm=0 gpu=0
  hb=200 fb=4 kv=16 rmax=1 rep=0 apc=16
  bytes=+37,900 ev=5568.0` — untagged; epoch-H
  streak [20:13:26]→[20:28:31] = 4 rows. Ramp:
  fb FLAT at 4 (3rd row — no fallback lines
  firing on the fresh engine), kv +1 (16), apc
  DOUBLED 8→16 (cache refill active), ev
  3048→5568; bytes +37,900 (light-load row).
  The lane's closing prose garbled mid-sentence
  (mixed-charter chatter) — RAW rows + verdict
  matched verbatim; echo noise only, file rows
  authoritative. Tagged ≈ 99; epoch-H clean
  rows 4. note: fix-105 (ESCWALL Tier-1+2
  staging lane by user decision) running in
  parallel — write scope = launcher + 2
  scheduler.py copies + backups + local mirrors
  only; disjoint from relay probes. Cycle 253
  dispatched.
- 2026-09-29 20:52 — **ESCWALL Tier-1+2 STAGED
  (user-approved "Both A + B staged"; fix-105
  lane died EMPTY before doing anything —
  executed directly, synchronously)**.
  - Remote ground truth: vllm-bin/dist-packages/
  vllm is a symlink-tree of venv site-packages
  vllm (`ls -i`: venv & dist scheduler.py share
  inode 504322 — ONE runtime file, not two).
  This resolves the fix-105-era "mystery":
  stage(V)'s own backup was visible through the
  symlink as stage(D)'s "pre-existing bak"; the
  D-skip also correctly prevented double-
  patching the same file.
  - Tier-1 LAUNCHER: md5 741f82c3f9d5da91054695
  b04a92e4a7 → **2cab139c3686b3f3411d6411b
  3204d3e** — three `${VAR:-default}` exports
  appended after the EVICTION_TOMBSTONES line
  (block marked by ESCWALL lines 476/479):
  STALE_HIT_TTL_SECS=60, STALE_HIT_FALLBACK_
  LIMIT=2, ESCAPE_WALL_SECS=240 (all user-
  override reachable); `bash -n` OK; backup
  `.bak-20260929-tier1` = 741f82c3 pre-state ✓.
  - Tier-2 RUNTIME scheduler.py (venv path =
  engine's actual import; dist = same inode):
  md5 4f7c9265a2b27e01013cbaa09b2f6653 →
  **86577993f20a74d0d2106e118047cc95** —
  (a) ESCWALL env parse inserted after the
  repeat-window line (:701 marker; _escape_
  wall_secs at :706/:717, default 240, 0=off);
  (b) escape-wall release branch in the escape
  fast-path (:1318/:1335): past the wall with
  frozen `num_computed_tokens`, logs
  `KV-LOOKUP ESCWALL release … wall= (no
  boundary change; cold re-arm)` and hard-
  releases escape/streak/repeat/fallback/
  wall-start bookkeeping so the request falls
  through to a fresh lookup — converting the
  permanent (0, False) hold into a bounded
  wedge cycle; py_compile OK on the engine's
  venv python; backup `.bak-20260929-escwall`
  = 4f7c9265 pre-state ✓.
  - Local mirrors updated identity-path and
  verified (Get-FileHash): tools/mirror-…-mtp-
  no-docker.sh = 2cab139c ✓; patched-files/
  distributed/kv_transfer/kv_connector/v1/
  offloading/scheduler.py = 86577993 ✓.
  - Gate story: run-1 dry-run crashed on a
  python compile() lint applied to the BASH
  launcher (lint made compile-flagged only);
  run-2 staged L+V cleanly (D abort-by-
  symlink); run-3 (--apply) aborted everywhere
  because stage() re-ran against run-2's own
  baks and staged an empty list; run-4
  install-only runner (md5-verified staged/
  bak/prestate per target) installed L+V GREEN:
  bash -n OK, PYCOMPILE_OK_V, launcher ESCWALL
  markers 2, scheduler markers 3.
  - NOT engaged: engine untouched — all fixes
  load at the user's next restart (§7 user-
  owned). Post-restart watch plan: NEWBOOTLOG
  rebase + §11 kv greps unchanged (kv_cache
  2115000000 / pool 237,865); NEW: APIServer
  environ grep should show the three exports;
  fb/kv cadence may RISE (TTL 60 + budget 2 =
  earlier escape trips — shorter staircases);
  `KV-LOOKUP ESCWALL release` logs appear only
  when a stuck escape actually hits the 240s
  wall. Verdict bars unchanged. Cycle 253 in
  flight (fix-106).
- 2026-09-29 19:21 — **RELAY CYCLE 248 (fix-101)
  reconciled: CYCLE_ALERT (newest row
  tagged, record-only) — grind-6 broke;
  grind-3 formed at grind-ready scale;
  fb ladder step +25 (engagement-larg-
  est)**. Raw verbatim: NOW=19:21:41,
  CANARY_ALIVE_39394, HEALTH=200,
  PROBE_EXIT=0, ROWLOG_N=314 (311+3,
  triplet barrel), freshness ~3s. Three
  new rows, all tagged: [19:10:58] `hb=
  000 SERVE-WEDGE fb=790 kv=3572 rmax=
  1 apc=1111 bytes=+73,584` (grind-6
  BROKE, streak reset to 0); [19:16:18]
  `hb=000 fb=798 kv=3617 apc=1112 bytes=
  +63,914`; [19:21:38] `hb=000 fb=811
  kv=3664 apc=1112 flat bytes=+65,812` —
  current grind = 3 consecutive rows
  ([19:10:58]→[19:21:38]); fb ladder
  786→811 (+25, LARGEST single ladder
  step of engagement); kv +141 in 3
  rows (3,523→3,664); apc 1112 (mil-
  lennium+ refills flat); ev middle
  316,688. Grind-9-scale build climbs:
  grind-6 streak → grind-3 forming (the
  13:33-14:16 grind-9 build had the
  same 6-row clean streak preceding the
  grind). Echo slip noted: [18:09:00]
  hb mis-echoed as 000 (file row is
  hb=200). Tagged ≈ 99; clean rows
  total 70. Serving untouched in EVERY
  row (h=200 p=9 500=0 oom=0); zero
  true-degradation triggers — record-
  only (family rule stands). Cycle 249
  dispatched.

