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
  when a   stuck escape actually hits the 240s
  wall. Verdict bars unchanged. Cycle 253 in
  flight (fix-106).
- 2026-09-29 20:57 — cycle 253 (fix-106)
  reconciled: CYCLE_OK.
  - [20:33:32] clean (fb=4 kv=19 apc=22).
  - [20:38:52] wedge-tagged: hb=000 single
  5-min dip, no wall loss; kv 19→27, fb
  4→7, apc flat 22 — the ESC211 fallback
  staircase in its self-releasing form
  (pre-restart ladder behavior as mapped;
  staged Tier-1+2 will bound it).
  - [20:43:53] clean: h=200 p=9 wall=200
  fb=9 kv=36 apc=28 bytes=+50332 ev 7184→
  10972. Newest row untagged → CYCLE_OK.
  - ROWLOG_N=330; epoch-H clean rows 6 of 7.
  Cycle 254 dispatching (fresh lane).
- 2026-09-29 21:10 — cycle 254 RECOVERY +
  **ESCWALL ENGAGED (epoch I)**.
  - fix-107 lane died on infra (Bad Gateway)
  with no probe → direct orchestrator probe
  (cycle-250 pattern; complex command shipped
  as /tmp/restart_verify.sh per user rule).
  - User restart #3 at ~20:54: old APIServer
  pid 176751 GONE (clean stop); NEWBOOTLOG
  [20:59:14]; epoch I APIServer pid 190081
  (EngineCore 190642, Worker 191003; cmdline
  identical to epoch H).
  - MODE: NEWBOOTLOG LANE:
    1. Tier-1 ENGAGED: /proc/190081/environ
    shows STALE_HIT_TTL_SECS=60,
    STALE_HIT_FALLBACK_LIMIT=2,
    ESCAPE_WALL_SECS=240 (plus preexisting
    tombstones 8M / tier_hugetlb 1 / dir).
    2. Tier-2 ENGAGED (behavioral, live log):
    21:08:46-57 — req chatcmpl-81e803… hit
    reproduction: identical-hit lookups →
    "stale-hit fallback fallbacks=1 wall=0.3"
    → "fallbacks=2 wall=10.4" (the 10 s
    KVC-PIN visibly persisted between the
    two fallbacks) → "KV-LOOKUP SERVERKILL
    … fallbacks=2 … -> escape (0, False)"
    — the ladder CLOSED at fallbacks=2 in
    10.4 s (limit knob verified behaviorally;
    default form was fallbacks=3), and the
    server kept POSTing 200s through the
    window (200s at 21:08:41/46/47/09:01).
  - Cycle 254 verdict: CYCLE_OK — newest
  rowlog row [21:09:15] clean: h=200 p=9
  wall=200 fb=4 kv=12 shm=1 bytes=+122834
  ev=1960 (fresh-epoch counters). ROWLOG_N=
  335; canary 39394 + HEALTH=200 live.
  - Watch note: "KV-LOOKUP ESCWALL release"
  line not yet observed — appears only if an
  escape sticks ≥240 s with a frozen
  boundary; today's escape drained itself,
  so the release path remains unobserved-by-
  design. fab/kv cadence may rise (earlier
  trips were the intent). Cycle 255
  dispatching (fresh lane).
- 2026-09-29 21:20 — cycle 255 (fix-108)
  reconciled: CYCLE_OK. [21:14:18] clean:
  h=200 p=9 wall=200 fb=6 kv=27 (small-
  scale early-epoch drift, no fence-climb
  pattern) apc=1 bytes=+54021 ev=3996.
  Epoch I: 2 clean rows since boot.
  ROWLOG_N=336. Cycle 256 dispatching
  (fresh lane).
- 2026-09-29 21:25 — cycle 256 (fix-109)
  reconciled: CYCLE_OK. [21:19:19] clean:
  h=200 p=9 wall=200 fb=6 (flat) kv=30
  (+3) apc=7 bytes=+50058 ev=5620. Epoch
  I: 3 clean rows since boot. ROWLOG_N=
  337. Cycle 257 dispatching (fresh lane).
- 2026-09-29 21:30 — cycle 257 (fix-110)
  reconciled: CYCLE_OK. [21:24:21] clean:
  h=200 p=9 wall=200 fb=6 (flat) kv=35
  (+5) apc=10 bytes=+20731 ev=6524. Epoch
  I: 4 clean rows since boot. ROWLOG_N=
  338. Cycle 258 dispatching (fresh lane).
- 2026-09-29 21:36 — cycle 258 (fix-111)
  reconciled: CYCLE_OK. [21:29:40] clean
  (fb=7 kv=44 apc=24 ev=8192), [21:34:41]
  clean (fb=9 kv=81 +37, largest epoch-I
  kv jump so far; apc=35 ev=11448) — wall
  intact both rows (hb=200 wall=200),
  fence-hit volume growing under serving
  load; escalation signature would be the
  same climb on hb=000 rows. Epoch I: 6
  clean rows since boot. ROWLOG_N=340.
  Cycle 259 dispatching (fresh lane).
- 2026-09-29 21:43 — cycle 259 (fix-112)
  reconciled: CYCLE_OK. [21:39:41] clean:
  h=200 p=9 wall=200 fb=9 (flat) kv=91
  (+10) apc=46 ev=12412.0,888.0,297.0 —
  first epoch-I activation of ev slots
  2/3 (small values; activity signature,
  no fence-flag). Epoch I: 7 clean rows
  since boot. ROWLOG_N=341. Cycle 260
  dispatching (fresh lane).
- 2026-09-29 21:48 — cycle 260 (fix-113)
  reconciled: CYCLE_OK. [21:44:43] clean:
  h=200 p=9 wall=200 fb=9 (flat) kv=96
  (+5) apc=52 ev=12412.0,1428.0,394.0
  (ev slots 2/3 still small). Epoch I: 8
  clean rows since boot. ROWLOG_N=342.
  Cycle 261 dispatching (fresh lane).
- 2026-09-29 21:55 — cycle 261 (fix-114)
  reconciled: CYCLE_OK. [21:49:46] clean:
  h=200 p=9 wall=200 fb=11 (+2) kv=116
  (+20, accelerating fence-hit volume)
  apc=62 ev=12404.0,3220.0,830.0 (slots
  2/3 rising). No hb dips; wedge shape
  still absent (wedge = hb=000 + climb +
  wall loss). Epoch I: 9 clean rows since
  boot. ROWLOG_N=343. Cycle 262
  dispatching (fresh lane).
- 2026-09-29 22:01 — cycle 262 (fix-115)
  reconciled: CYCLE_OK. [21:54:55] clean
  (fb=11 kv=119 +3 apc=65 ev slot2 3688/
  slot3 1018), [21:59:55] clean (fb=11
  flat, kv=122 +3, apc=69, ev
  12412.0,4880.0,1256.0) — kv drift
  decelerating, both walls intact. Epoch
  I: 11 clean rows since boot. ROWLOG_N=
  345. Cycle 263 dispatching (fresh lane).
- 2026-09-29 22:07 — cycle 263 (fix-116)
  reconciled: **CYCLE_ALERT: SERVE-WEDGE**
  — [22:05:16] wedge-tagged, LONE (first
  in epoch I): h=200 p=9, 500=0, shm=0,
  hb=000 (both fence-hit mid-row), fb
  11→15, kv 122→137, apc=73, ev slots 2/3
  rising (6532/1672), bytes +38982.
  Verdict-bearers healthy (h/500/oom) —
  one-cycle grace applies (serving
  responsive; post-fix bounded ladder,
  worst-case capped by the 240 s ESCWALL
  wall; epoch I's 21:08 Serverkill
  self-drained in seconds). Cycle 264
  dispatched immediately — its newest row
  decides recovery vs continuance (≥2
  consecutive wedge rows = escalation
  signal; if it recurs, verify latest.log
  for "KV-LOOKUP ESCWALL release" /
  "SERVERKILL" to confirm the bounded
  path engaged).
- 2026-09-29 22:14 — cycle 264 (fix-117)
  reconciled: CYCLE_OK — recovery
  confirmed: [22:10:16] clean (h=200 p=9
  wall=200), so the [22:05:16] wedge
  self-released within one row cycle and
  did NOT stick (no wall=000-patrol). The
  recovery happened under epoch I's
  heaviest load yet: fb=21, kv=167 (+30,
  epoch-I max), apc=76, ev slots 7788/
  2007 — load-drift signature with
  serving healthy, not the wedge shape.
  Epoch I: 11 clean rows of 12 since
  boot. ROWLOG_N=347. Cycle 265
  dispatching (fresh lane).
- 2026-09-29 22:21 — cycle 265 (fix-118)
  reconciled: CYCLE_OK. [22:15:17] clean:
  h=200 p=9 wall=200 fb=22 (+1) kv=182
  (+15) apc=79 ev=12412.0,10912.0,2815.0
  — load drift continues under walls
  intact; ev slots 2/3 at 10.9k/2.8k vs
  wedge-era historical peaks 317k/78k
  (~30x smaller accumulation — consistent
  with the bounded ladder, NOT proof).
  Epoch I: 12 of 13 rows clean (lone
  wedge [22:05:16]). ROWLOG_N=348. Cycle
  266 dispatching (fresh lane).
- 2026-09-29 22:27 — cycle 266 (fix-119)
  reconciled: CYCLE_OK — [22:25:38]
  clean (fb=23 kv=205 apc=91 ev slots
  15412/3911). [22:20:37] wedge-tagged
  AGAIN (hb=000 kv=192 apc=87 ev slots
  12220/3095) — epoch I's 2nd wedge
  event, again a SINGLE row with full
  recovery by the next row (bounded
  oscillation, no stick). Wedge rate
  ~half of epoch F and both events
  released within one row; escalation
  only if a wedge sticks ≥2 consecutive
  rows. Epoch I: 13 of 15 rows clean
  (2 lone wedge rows: 22:05:16, 22:20:37).
  ROWLOG_N=350. Cycle 267 dispatching
  (fresh lane).
- 2026-09-29 22:33 — cycle 267 (fix-120)
  reconciled: CYCLE_OK. [22:30:44] clean:
  h=200 p=9 wall=200 fb=24 (+1) kv=219
  (+14) apc=101 (crossed 100) ev slots
  17300/4361 — load drift continues, no
  consecutive wedge rows (both previous
  wedges stay lone). Epoch I: 14 of 16
  rows clean. ROWLOG_N=351. Cycle 268
  dispatching (fresh lane).
- 2026-09-29 22:45 — cycle 268 (fix-121)
  reconciled: CYCLE_OK. [22:35:44]
  clean: h=200 p=9 wall=200 fb=25 (+1)
  kv=235 (+16, mild acceleration) apc=
  108 ev slots 21448/5408 — load drift
  under intact walls; escalation signal
  still unmet (no consecutive wedge
  rows). Epoch I: 15 of 17 rows clean.
  ROWLOG_N=352.
  - LEDGER WORKFLOW NOTE: during this
  reconcile it surfaced that the c248
  record block had been stranded at the
  file end since its 19:21 write (every
  later terminus-append landed ABOVE it;
  caught via a wrong-anchor guess on the
  c268 record — this edit missed because
  the file tail was not where assumed).
  Fixed by byte-safe relocation (commit
  4affbff: 35/35 move, content intact,
  single copy, line count unchanged).
  Rule adopted from here on: READ the
  actual ledger tail before every record
  append; anchor on fresh-read text only,
  never on memory. Cycle 269 dispatching
  (fresh lane).
- 2026-09-29 22:52 — cycle 269 (fix-122)
  reconciled: CYCLE_OK. New rows:
  [22:45:46] clean (fb=32 kv=273 apc=114
  ev 26132/6617), [22:51:04] clean (fb=
  32 flat, kv=283 +10, apc=119, ev
  28680/7277) — walls intact, 0 wedge
  rows this window; the 2 known wedge
  rows remain lone (non-consecutive) —
  bounded oscillation continues, no
  family escalation. Probe output echoed
  the [22:10:16..22:45:46] section twice
  (self-consistent echo slip; footer
  ROWLOG_N=355 authoritative). Epoch I:
  17 of 19 rows clean. kv climb continues
  (slot2 28.7k > slot1 12.4k; still ~11x
  below wedge-era peaks). Epoch I: 15 of
  19 rows clean. ROWLOG_N=355. Cycle 270
  dispatching (fresh lane).
- 2026-09-29 22:59 — cycle 270 (fix-123)
  reconciled: CYCLE_OK. [22:56:05]
  clean: h=200 p=9 wall=200 fb=33 (+1)
  kv=295 (+12) apc=126 ev slots 32012/
  8068 — climb continues under intact
  walls; 0 wedge rows since 22:20:37
  (the 2 known wedge rows remain lone
  and non-consecutive). ERRATA: c269's
  tail line typoed a stale clean-row
  total ("15 of 19") after correctly
  stating 17 of 19 earlier in the same
  block — actual c269 = 17/19; dup line
  logged here, not rewritten. Epoch I:
  18 of 20 rows clean. ROWLOG_N=356.
  Cycle 271 dispatching (fresh lane).
- 2026-09-29 23:07 — cycle 271 (fix-124)
  reconciled: CYCLE_OK. [23:01:06] clean
  (fb=35 kv=310 apc=131 ev 35064/8867),
  [23:06:07] clean (fb=35 flat, kv=320
  +10, apc=137, ev 36356/9145) — 8
  consecutive clean rows since the
  22:20:37 wedge (~40 min); 0 wedge
  rows this window; bounded oscillation
  persists, ESCWALL release still
  unobserved (every escape self-releases
  well under the 240 s wall — consistent
  by design). Epoch I: 20 of 22 rows
  clean. ROWLOG_N=358. Cycle 272
  dispatching (fresh lane).
- 2026-09-29 23:14 — cycle 272 (fix-125)
  reconciled: CYCLE_OK. [23:11:08]
  clean: h=200 p=9 wall=200 fb=37 (+2)
  kv=332 (+12) apc=148 (+11) ev slots
  38612/9769 — 9 consecutive clean rows
  since the 22:20:37 wedge; 0 wedge rows
  this window; escalation signal still
  unmet. Epoch I: 21 of 23 rows clean.
  ROWLOG_N=359. Cycle 273 dispatching
  (fresh lane).
- 2026-09-29 23:21 — cycle 273 (fix-126)
  reconciled: CYCLE_OK. [23:16:08]
  clean: h=200 p=9 wall=200 fb=37 (flat)
  kv=339 (+7) apc=153 ev slots 40740/
  10339 — 10 consecutive clean rows
  since the 22:20:37 wedge; escalation
  signal unmet. Epoch I: 22 of 24 rows
  clean. ROWLOG_N=360. Cycle 274
  dispatching (fresh lane).
- 2026-09-29 23:27 — cycle 274 (fix-127)
  reconciled: CYCLE_OK. [23:21:09]
  clean: h=200 p=9 wall=200 fb=37
  (flat) kv=340 (+1, nearly flat this
  row) apc=158 ev slots 41502/10508 —
  11 consecutive clean rows since the
  22:20:37 wedge; escalation signal
  unmet. Epoch I: 23 of 25 rows clean.
  ROWLOG_N=361. Cycle 275 dispatching
  (fresh lane).
- 2026-09-29 23:32 — cycle 275 (fix-128)
  reconciled: CYCLE_OK. [23:26:29]
  clean (fb=37 flat, kv=340 +0, apc=
  162, bytes +10862, ev 41786/10511),
  [23:31:30] clean (fb=37 flat, kv=342
  +2, apc=162 flat, bytes +13836, ev
  43002/10867) — load settling (smallest
  byte deltas since 22:25), 0 wedge
  rows since 22:20:37 (13 consecutive
  clean rows). Echo-drift note: lane's
  echoed historical rows differ slightly
  from earlier echoes ([22:25:38]/
  [22:30:44] values); footer + new-row
  content authoritative. Epoch I: 25 of
  27 rows clean. ROWLOG_N=363. Cycle
  276 dispatching (fresh lane).
- 2026-09-29 23:38 — cycle 276 (fix-129)
  reconciled: CYCLE_OK. [23:36:30]
  clean: h=200 p=9 wall=200 fb=37 (flat)
  kv=344 (+2) apc=167 bytes=+15319 ev
  43054/10872 — 14 consecutive clean
  rows since the 22:20:37 wedge; load
  settled (byte deltas 10-15k range).
  Epoch I: 26 of 28 rows clean.
  ROWLOG_N=364. Cycle 277 dispatching
  (fresh lane).
- 2026-09-29 23:44 — cycle 277 (fix-130)
  reconciled: CYCLE_OK — PROVISIONAL.
  Lane delivered verdict-only prose (raw
  rowlog rows MISSING from the result —
  echo-slip variant): claims newest row
  [23:41:31] untagged + rows 22:25:38 →
  23:41:31 window, CANARY_ALIVE_39394 +
  HEALTH=200, single wait + single probe
  exit 0. Family rule = judge from file
  rows, so provisional-ok pending file-
  row verification by cycle 278 (same
  window: rows 23:41/23:46 will show).
  Epoch I stay 26 of 28 rows clean
  (c276 counts; no verified delta yet).
  ROWLOG_N=unverified (provisional ~365).
  Cycle 278 dispatching (fresh lane).
- 2026-09-29 23:50 — cycle 278 (fix-131)
  reconciled: CYCLE_OK — file rows
  verified. [23:41:31] CLEAN (fb=37
  kv=345 apc=173) — confirms c277's
  provisional verdict (echo-slip gap
  closed: no wedge row was missed).
  [23:46:32] clean newest: fb=37 (flat)
  kv=346 (+1) apc=180 bytes=+19737 ev
  43330/10877 — 15 consecutive clean
  rows since the 22:20:37 wedge; ESCWALL
  release still unobserved (all escapes
  self-release under 240 s by design).
  Epoch I: 28 of 30 rows clean.
  ROWLOG_N=366. Cycle 279 dispatching
  (fresh lane).
- 2026-09-29 23:56 — cycle 279 (fix-132)
  reconciled: CYCLE_OK. [23:51:32]
  clean: h=200 p=9 wall=200 fb=38 (+1)
  kv=353 (+7) apc=191 bytes=+36521 ev
  45110/11388 — 16 consecutive clean
  rows since the 22:20:37 wedge; load
  mildly re-accelerating (bytes 36.5k).
  Epoch I: 29 of 31 rows clean.
  ROWLOG_N=367. Cycle 280 dispatching
  (fresh lane).
- 2026-09-30 00:05 — cycle 280 (fix-133)
  reconciled: CYCLE_OK with a new LONE
  wedge note. [23:56:53] = 3rd lone
  SERVE-WEDGE in epoch I (hb=000 kv=355
  fb=38 apc=192 bytes +23358, no WALL
  LOST marker) — bounded single-row
  form held: full recovery by [00:01:53]
  (h=200 wall=200 fb=43 +5 kv=369 +14
  apc=194 bytes +35278 ev 46610/11833).
  17-row clean run broken by the wedge;
  wedge cadence so far ~hourly-ish
  (22:05, 22:20, 23:56); escalation
  signal STILL unmet (no consecutive
  wedge rows). Epoch I: 30 of 33 rows
  clean. ROWLOG_N=369. Cycle 281
  dispatching (fresh lane).
- 2026-09-30 00:11 — cycle 281 (fix-134)
  reconciled: CYCLE_OK. [00:06:54]
  clean: h=200 p=9 wall=200 fb=43 (flat)
  kv=373 (+4) apc=198 bytes=+19263 ev
  47446/11844 — no new wedge this cycle;
  the 23:56:53 wedge remains lone/bounded
  (no consecutive-row stick; escalation
  signal unmet). Epoch I: 31 of 34 rows
  clean. ROWLOG_N=370. Cycle 282
  dispatching (fresh lane).
- 2026-09-30 00:19 — cycle 282 (fix-135)
  reconciled: CYCLE_ALERT: SERVE-WEDGE
  — family escalation signal ARMED (first
  ≥2-row stick in epoch I). [00:12:14]
  tagged (hb=000 kv=390 fb=49 apc=202
  bytes +31062) AND [00:17:35] tagged
  newest (hb=000 kv=404 fb=53 apc=206
  bytes +50584) — CONSECUTIVE wedge rows;
  pv wedge [23:56:53] had fully recovered
  by [00:01:53]; this stick began ~15 min
  later and had not released by newest
  row. Still bounded: no WALL LOST
  markers, all counters advancing, ~5-min
  cadence intact, CANARY_ALIVE_39394 +
  HEALTH=200. Next marker: a 3rd
  consecutive wedge row (or recovery) in
  cycle 283; ESCWALL release line not yet
  observed (wall 240 s would crown the
  00:12:14 stick by ~00:16:54 — release
  line not verified in this snapshot).
  Epoch I: 31 of 36 rows clean.
  ROWLOG_N=372. Cycle 283 dispatching
  (fresh lane).
- 2026-09-30 00:29 — cycle 283 (fix-136,
  escalation watch) reconciled: CYCLE_
  ALERT: SERVE-WEDGE with stick RELEASED.
  [00:22:35] clean (hb=200 fb=61 kv=429)
  — the epoch-I 2-row stick (00:12:14 +
  00:17:35) did NOT reach a 3rd
  consecutive row; recovery confirmed.
  BUT newest row [00:27:56] = fresh
  ISOLATED wedge row (hb=000 fb=71 kv=
  458 apc=212 bytes +53015 ev 51634/
  12887) — ladder rungs deepening (53→61
  →71, deepest fb spend in epoch I);
  6 wedge rows in epoch I total (22:05,
  22:20, 23:56, 00:12, 00:17, 00:27).
  Still bounded: no WALL LOST, counters
  advancing, CANARY_ALIVE_39394 +
  HEALTH=200. Next marker (c284):
  3rd consecutive row OR continued rung
  deepening → direct-orchestrator deep-
  dive. Epoch I: 34 of 40 rows clean.
  ROWLOG_N=374. Cycle 284 dispatching
  (fresh lane).
- 2026-09-30 00:36 — cycle 284 (fix-137,
  escalation watch) reconciled: CYCLE_OK.
  [00:32:56] clean (hb=200 fb=71 FLAT
  kv=464 apc=225 bytes +60503 ev 53482/
  13426) — the 00:27:56 wedge stayed
  ISOLATED (released, no 3rd consecutive
  row); no WALL LOST anywhere; all
  counters advancing. Open watch item:
  per-wedge rung lineage deepening
  (38→49→53→71); trigger for deep-dive =
  next wedge eating beyond fb=71 or any
  WALL LOST marker. Epoch I: 35 of 41
  rows clean (6 wedge rows: 22:05, 22:20,
  23:56, 00:12, 00:17, 00:27). ROWLOG_N=
  375. Cycle 285 dispatching (fresh
  lane).
- 2026-09-30 00:52 — cycle 285 (fix-138,
  escalation watch) reconciled: CYCLE_
  ALERT: SERVE-WEDGE on newest row
  [00:38:17] (hb=000 fb=73 kv=477; rowlog
  fb aggregate breached the fb=71 watch
  threshold). Prior row [00:32:56] clean
  → isolated single-row wedge (no
  consecutive stick; 7 wedge rows in
  epoch I). DEEP-DIVE (direct-orchestrator
  read-only engine-log scan, script
  /tmp/escwall_deepdive.sh) DISMISSES the
  deepening alarm: latest.log =
  engine-20260929-205722.log (epoch-I
  boot only, BOOTCOUNT=1); KVL_COUNT=501,
  FALLBACK_COUNT=78, ESCAPE_COUNT=18 with
  UNIFORM ladder form — every SERVERKILL
  closed at fallbacks=2 wall≈10.4 s (e.g.
  21:08:11 … 00:43:35 tail); RELEASE_
  COUNT=0 (no escape touched the 240 s
  cap — bounded by design). REINTERPRETA-
  TION: rowlog fb is a CUMULATIVE count
  of fallback EVENTS (load-dependent)
  across all requests in a window, NOT
  per-request ladder depth; engine ladder
  flat at 2 → bounded form INTACT.
  Escalation posture DE-ARMED for depth;
  new watch triggers: ≥3 consecutive
  hb=000 rows, any WALL LOST marker,
  fallbacks=3 in engine log, or first
  ESCWALL release line. Epoch I: 35 of
  42 rows clean. ROWLOG_N=376. Cycle 286
  dispatching (fresh lane).
- 2026-09-30 00:53 — cycle 286 (fix-139)
  reconciled: CYCLE_OK. [00:43:22]
  clean (hb=200 fb=74 kv=483 apc=246
  ev 59582/14911) and [00:48:40] clean
  newest (fb=80 kv=515 apc=248 ev
  60858/15215) — the [00:38:17] wedge
  released cleanly (2 consecutive clean
  rows after it); no new hb=000 rows, no
  sticks, no WALL LOST anywhere; fb
  aggregate 73→80 = normal drift (de-armed
  posture). Epoch I: 37 of 44 rows
  clean (7 wedge rows). ROWLOG_N=378.
  Cycle 287 dispatching (fresh lane).
- 2026-09-30 01:03 — cycle 287 (fix-140)
  reconciled: CYCLE_OK. Wedge #8 of
  epoch I at [00:54:00] (hb=000 fb=81
  kv=522 apc=255 ev 63322/15890) —
  ISOLATED single-row wedge, released by
  [00:59:01] clean (hb=200 fb=86 kv=539
  apc=259); no consecutive stick, no
  WALL LOST anywhere; fb 74→80→86 =
  normal aggregate drift (de-armed
  posture). Epoch I: 38 of 46 rows
  clean (8 wedge rows). ROWLOG_N=380.
  Cycle 288 dispatching (fresh lane).
- 2026-09-30 01:12 — cycle 288 (fix-141)
  reconciled: CYCLE_OK. Wedge #9 of
  epoch I at [01:04:21] (hb=000 fb=89
  kv=550 apc=263 ev 69356/17373) —
  ISOLATED single-row wedge, released by
  [01:09:22] clean (hb=200 fb=92 kv=561
  apc=266 ev 72327/18073); no consecutive
  stick, no WALL LOST anywhere; fb drift
  normal. Cadence tightening: wedge gaps
  15.7 min → 10:21 (00:38→00:54→01:04).
  Epoch I: 39 of 48 rows clean (9 wedge
  rows). ROWLOG_N=382. Cycle 289
  dispatching (fresh lane).
- 2026-09-30 01:21 — cycle 289 (fix-142)
  reconciled: CYCLE_ALERT: SERVE-WEDGE.
  Wedge #10 of epoch I at [01:19:55]
  (hb=000 fb=98 FLAT on wedge row kv=586
  apc=292 ev 78009/19451) — ISOLATED:
  [01:14:35] clean (fb=98 kv=581 apc=
  271), release expected next row; no
  consecutive stick, no WALL LOST
  anywhere. Cadence now steady ~10-15
  min (gaps 15:34 / 10:21 / 15:34) —
  regular recurring bounded wedge
  pattern, not degradation (engine
  ladder flat fb=2). Epoch I: 40 of 50
  rows clean (10 wedge rows). ROWLOG_N=
  384. Cycle 290 dispatching (fresh
  lane).
- 2026-09-30 01:30 — cycle 290 (fix-143)
  reconciled: CYCLE_ALERT: SERVE-WEDGE —
  second 2-row stick of epoch I: the
  [01:19:55] wedge did NOT release at
  the next row; newest [01:25:16] also
  hb=000 (fb=104 kv=606 apc=297 ev
  80605/20164). Mirrors the 00:12:14/
  00:17:35 stick which released at
  [00:22:35] — release row expected next.
  No WALL LOST anywhere; fb aggregate
  drift normal. Count: 11 wedge rows;
  epoch I: 40 of 51 rows clean.
  ROWLOG_N=385. Cycle 291 dispatching
  (fresh lane).
- 2026-09-30 01:41 — cycle 291 (fix-144,
  escalation watch) reconciled: CYCLE_OK
  (newest [01:35:45] clean: hb=200 fb=
  105 FLAT kv=619 apc=318 ev 86101/
  21493) — BUT family escalation marker
  RECORDED: the stick ran THREE
  consecutive wedge rows — [01:19:55]+
  [01:25:16]+[01:30:36] (fb 98→104→105,
  kv 586→606→611) — the FIRST 3-row
  stick in epoch I, released at
  [01:35:45]. Bounded-slide precedent
  extended: no WALL LOST, counters
  advanced throughout, server healthy.
  New family state = 3-row stick
  observed once; watch repeats or longer
  sticks (≥4 = user-visible alarm).
  Count: 12 wedge rows; epoch I: 41 of
  53 rows clean. ROWLOG_N=387. Cycle 292
  dispatching (fresh lane).
- 2026-09-30 01:48 — cycle 292 (fix-145,
  escalation watch) reconciled: CYCLE_OK.
  Wedge #13 at [01:41:06] (hb=000 fb=107
  kv=626 apc=325 ev 89433/22332) —
  ISOLATED single-row; released
  immediately by [01:46:06] newest clean
  (hb=200 fb=110 kv=640 apc=339 ev
  91741/22950). Stick run-lengths RESET
  from the 3-row peak back to single-row
  (no lengthening); cadence steady
  (hb=000 every ~2-4 rows); no WALL LOST
  anywhere; fb aggregate drift normal.
  Epoch I: 42 of 55 rows clean (13
  wedge rows). ROWLOG_N=389. Cycle 293
  dispatching (fresh lane).
- 2026-09-30 01:58 — cycle 293 (fix-146,
  escalation watch) reconciled: CYCLE_OK.
  No new wedge rows; [01:51:24] (fb=118
  kv=662 apc=348) and [01:56:44] (fb=119
  kv=671 apc=362 ev 95177/23738) clean —
  3 consecutive clean rows since wedge
  #13; run-length story unchanged
  (single-row wedges, no new sticks, ≥4
  nowhere near); no WALL LOST anywhere;
  fb aggregate drift normal; ev slots
  keep climbing (~95k/23.7k). Epoch I:
  44 of 57 rows clean (13 wedge rows).
  ROWLOG_N=391. Cycle 294 dispatching
  (fresh lane).
- 2026-09-30 02:07 — cycle 294 (fix-147,
  escalation watch) reconciled: CYCLE_OK.
  No new wedge rows; [02:01:45] newest
  clean (fb=126 kv=695 apc=369 ev
  98393/24510) — 4-row clean streak since
  wedge #13 ([01:46:06]…[02:01:45]); no
  stick start, no WALL LOST anywhere;
  fb aggregate drift normal. Epoch I:
  45 of 58 rows clean (13 wedge rows).
  ROWLOG_N=392. Cycle 295 dispatching
  (fresh lane).
- 2026-09-30 02:15 — cycle 295 (fix-148,
  escalation watch) reconciled: CYCLE_OK.
  No new wedge rows; [02:06:48] and
  [02:11:49] (fb=130 kv=715 apc=373 ev
  101293/25343) clean — 6-row clean
  streak ([01:46:06]→[02:11:49]); ev
  slot2 crossed 100k (101293, ~3.2x below
  wedge-era peak 317k); no stick start,
  no WALL LOST anywhere; fb aggregate
  drift normal. Epoch I: 47 of 60 rows
  clean (13 wedge rows). ROWLOG_N=394.
  Cycle 296 dispatching (fresh lane).
- 2026-09-30 02:24 — cycle 296 (fix-149,
  escalation watch) reconciled: CYCLE_OK.
  Wedge #14 at [02:17:09] (hb=000 fb=134
  kv=727 apc=374 ev 102661/25561 — note
  console-echo typo '25561:0' in lane
  paste; row values otherwise standard)
  — ISOLATED single-row, released by
  [02:22:10] newest clean (hb=200 fb=136
  kv=736 apc=380 ev 103473/25792); the
  6-row clean streak ended by this
  bounded stick; no WALL LOST anywhere;
  fb aggregate drift normal. Epoch I:
  48 of 62 rows clean (14 wedge rows).
  ROWLOG_N=396. Cycle 297 dispatching
  (fresh lane).
- 2026-09-30 02:31 — cycle 297 (fix-150,
  escalation watch) reconciled: CYCLE_OK.
  [02:27:10] newest clean (h=200 p=9
  wall=200 fb=136 FLAT (+0) kv=740 apc=
  383 ev 105537/26382); [02:22:10] also
  clean — 2 consecutive clean rows after
  wedge #14 [02:17:09]; no new hb=000
  rows, no WALL LOST anywhere; fb flat
  (no new fallback events this cycle);
  bounded cadence holds. Epoch I: 49 of
  63 rows clean (14 wedge rows).
  ROWLOG_N=397. Cycle 298 dispatching
  (fresh lane).
- 2026-09-30 02:36 — cycle 298 (fix-151,
  escalation watch) reconciled: CYCLE_OK.
  [02:32:11] newest clean (h=200 p=9
  wall=200 fb=136 FLAT 3rd row running
  kv=743 apc=386 ev 106837/26687); no
  new hb=000 rows; 3-row clean streak
  after wedge #14; no WALL LOST
  anywhere; no stick start (epoch max
  stick stays 3, never ≥4). Epoch I:
  50 of 64 rows clean (14 wedge rows).
  ROWLOG_N=398. Cycle 299 dispatching
  (fresh lane).
- 2026-09-30 02:42 — cycle 299 (fix-152,
  escalation watch) reconciled: CYCLE_OK.
  [02:37:12] newest clean (h=200 p=9
  wall=200 fb=136 FLAT 4th row running
  kv=744 apc=391 ev 107541/26887); no
  new hb=000 rows; 4-row clean streak
  ([02:22:10]…[02:37:12]); no WALL LOST
  anywhere; no stick start (epoch max
  stick stays 3). Quiet load row.
  Epoch I: 52 of 66 rows clean (14
  wedge rows). ROWLOG_N=399. Cycle 300
  dispatching (fresh lane).
- 2026-09-30 02:48 — cycle 300 (fix-153,
  escalation watch) reconciled: CYCLE_OK
  — ROWLOG milestone: 400th row (N=400).
  [02:42:12] newest clean (h=200 p=9
  wall=200 fb=136 FLAT 5th row running
  kv=745 apc=396 ev 108249/27057); no
  new hb=000 rows (still 14 wedges); 5-row
  clean streak ([02:22:10]…[02:42:12]);
  no WALL LOST; no stick start. Quiet
  load (bytes +14.7k, smallest since
  23:21). Epoch I: 54 of 68 rows clean.
  Cycle 301 dispatching (fresh lane).
- 2026-09-30 02:54 — cycle 301 (fix-154)
  reconciled: CYCLE_OK. [02:47:28] and
  [02:52:29] (fb=136 kv=749 apc=399 ev
  109661/27409) clean — 7-row clean
  streak ([02:22:10]…[02:52:29]), longest
  since the 16-row run ending at wedge
  #3; no new hb=000 rows (still 14
  wedges); no WALL LOST; fb flat since
  [02:22:10] (no fallback events). Epoch
  I: 56 of 70 rows clean. ROWLOG_N=402.
  Cycle 302 dispatching (fresh lane).
- 2026-09-30 03:00 — cycle 302 (fix-155)
  reconciled: CYCLE_OK. [02:57:29]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 8th row running kv=750 apc=
  404 ev 110409/27560); 8-row clean
  streak ([02:22:10]…[02:57:29]); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 57 of 71 rows clean.
  ROWLOG_N=403. Cycle 303 dispatching
  (fresh lane).
- 2026-09-30 03:05 — cycle 303 (fix-156)
  reconciled: CYCLE_OK. [03:02:30]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 10th row running kv=751 apc=
  410 ev 110441/27560); 10-row clean
  streak ([02:22:10]…[03:02:30]) —
  longest of the 02-h quiet stretch; no
  new hb=000 rows (still 14 wedges); no
  WALL LOST; fb flat (no fallback
  events); load still quieting (ev
  slots nearly flat). Epoch I: 58 of
  72 rows clean. ROWLOG_N=404. Cycle
  304 dispatching (fresh lane).
- 2026-09-30 03:12 — cycle 304 (fix-157)
  reconciled: CYCLE_OK. [03:07:30]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 11th row running kv=754 apc=
  413 ev 111957/27937); 11-row clean
  streak ([02:22:10]…[03:07:30]); no
  new hb=000 rows (still 14 wedges); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 59 of 73 rows
  clean. ROWLOG_N=405. Cycle 305
  dispatching (fresh lane).
- 2026-09-30 03:19 — cycle 305 (fix-158)
  reconciled: CYCLE_OK. [03:12:31]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 12th row running kv=755 apc=
  417 ev 113497/28285); 12-row clean
  streak ([02:22:10]…[03:12:31]; lane
  prose echoed '15' — file rows are
  authoritative); no new hb=000 rows
  (still 14 wedges); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  61 of 75 rows clean. ROWLOG_N=406.
  Cycle 306 dispatching (fresh lane).
- 2026-09-30 03:25 — cycle 306 (fix-159)
  reconciled: CYCLE_OK. [03:17:32] and
  [03:22:32] (fb=136 kv=756 apc=422 ev
  114317/28551) clean — 14-row clean
  streak ([02:22:10]…[03:22:32]); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 63 of 77 rows clean.
  ROWLOG_N=408. Cycle 307 dispatching
  (fresh lane).
- 2026-09-30 03:31 — cycle 307 (fix-160)
  reconciled: CYCLE_OK. [03:27:33]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 14th row running kv=758 apc=
  427 ev 115149/28709); 14-row clean
  streak ([02:22:10]…[03:27:33];
  recounted from file rows — lane prose
  counts drift on streak length); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 64 of 78 rows clean.
  ROWLOG_N=409. Cycle 308 dispatching
  (fresh lane).
- 2026-09-30 03:38 — cycle 308 (fix-161)
  reconciled: CYCLE_OK. [03:32:34]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 15th row running kv=760 apc=
  431 ev 116061/28989); 15-row clean
  streak ([02:22:10]…[03:32:34];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  65 of 79 rows clean. ROWLOG_N=410.
  Cycle 309 dispatching (fresh lane).
- 2026-09-30 03:45 — cycle 309 (fix-162)
  reconciled: CYCLE_OK. [03:37:34] and
  [03:42:35] (fb=136 kv=761 apc=435 ev
  116749/29141) clean — 17-row clean
  streak ([02:22:10]…[03:42:35];
  recounted from file rows; lane prose
  echoed '28 consecutive' — echo slip,
  file rows authoritative); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  67 of 81 rows clean. ROWLOG_N=412.
  Cycle 310 dispatching (fresh lane).
- 2026-09-30 03:50 — cycle 310 (fix-163)
  reconciled: CYCLE_OK. [03:47:36]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 18th row running kv=763 apc=
  440 ev 117125/29146); 18-row clean
  streak ([02:22:10]…[03:47:36];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  68 of 82 rows clean. ROWLOG_N=413.
  Cycle 311 dispatching (fresh lane).
- 2026-09-30 03:56 — cycle 311 (fix-164)
  reconciled: CYCLE_OK. [03:52:36]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 19th row running kv=765 apc=
  445 ev 117525/29337); 19-row clean
  streak ([02:22:10]…[03:52:36];
  recounted from file rows — lane prose
  slipped the streak start to [02:32:
  11], file rows authoritative); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  69 of 83 rows clean. ROWLOG_N=414.
  Cycle 312 dispatching (fresh lane).
- 2026-09-30 04:02 — cycle 312 (fix-165)
  reconciled: CYCLE_OK. [03:57:37]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 20th row running kv=767 apc=
  448 ev 118325/29498); 20-row clean
  streak ([02:22:10]…[03:57:37];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  70 of 84 rows clean. ROWLOG_N=415.
  Cycle 313 dispatching (fresh lane).
- 2026-09-30 04:07 — cycle 313 (fix-166)
  reconciled: CYCLE_OK. [04:02:37]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 21st row running kv=767 apc=
  455 ev 118361/29498); 21-row clean
  streak ([02:22:10]…[04:02:37];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  71 of 85 rows clean. ROWLOG_N=416.
  Cycle 314 dispatching (fresh lane).
- 2026-09-30 04:12 — cycle 314 (fix-167)
  reconciled: CYCLE_OK. [04:07:38]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 22nd row running kv=767 apc=
  462 ev 118397/29498); 22-row clean
  streak ([02:22:10]…[04:07:38];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  72 of 86 rows clean. ROWLOG_N=417.
  Cycle 315 dispatching (fresh lane).
- 2026-09-30 04:18 — cycle 315 (fix-168)
  reconciled: CYCLE_OK. [04:12:39]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 23rd row running kv=767 apc=
  469 ev 118433/29501); 23-row clean
  streak ([02:22:10]…[04:12:39];
  recounted from file rows; lane prose
  echoed '24' — echo slip, file rows
  authoritative); no new hb=000 rows
  (still 14 wedges, nothing since
  [02:17:09]); no WALL LOST; fb flat
  (no fallback events). Epoch I: 73 of
  87 rows clean. ROWLOG_N=418. Cycle
  316 dispatching (fresh lane).
- 2026-09-30 04:25 — cycle 316 (fix-169)
  reconciled: CYCLE_OK. 2 new rows
  [04:17:54] and [04:22:54] (fb=136
  kv=770 apc=473 ev 119341/29771)
  clean — 25-row clean streak
  ([02:22:10]…[04:22:54]; recounted
  from file rows); no new hb=000 rows
  (still 14 wedges, nothing since
  [02:17:09]); no WALL LOST; fb flat
  (no fallback events). Epoch I: 75 of
  89 rows clean. ROWLOG_N=420. Cycle
  317 dispatching (fresh lane).
- 2026-09-30 04:31 — cycle 317 (fix-170)
  reconciled: CYCLE_OK. [04:27:55]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 26th row running kv=773 apc=
  477 ev 120228/30012 — ev slot1
  crossed 120k); 26-row clean streak
  ([02:22:10]…[04:27:55]; recounted
  from file rows); no new hb=000 rows
  (still 14 wedges, nothing since
  [02:17:09]); no WALL LOST; fb flat
  (no fallback events). Epoch I: 76 of
  90 rows clean. ROWLOG_N=421. Cycle
  318 dispatching (fresh lane).
- 2026-09-30 04:36 — cycle 318 (fix-171)
  reconciled: CYCLE_OK. [04:32:56]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 27th row running kv=773 apc=
  484 ev 120256/30012); 27-row clean
  streak ([02:22:10]…[04:32:56];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  77 of 91 rows clean. ROWLOG_N=422.
  Cycle 319 dispatching (fresh lane).
- 2026-09-30 04:42 — cycle 319 (fix-172)
  reconciled: CYCLE_OK. [04:37:56]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 28th row running kv=774 apc=
  488 ev 121267/30190); 28-row clean
  streak ([02:22:10]…[04:37:56];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  78 of 92 rows clean. ROWLOG_N=423.
  Cycle 320 dispatching (fresh lane).
- 2026-09-30 04:48 — cycle 320 (fix-173)
  reconciled: CYCLE_OK. [04:42:57]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 29th row running kv=774 apc=
  494 ev 121811/30359); 29-row clean
  streak ([02:22:10]…[04:42:57];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  79 of 93 rows clean. ROWLOG_N=424.
  Cycle 321 dispatching (fresh lane).
- 2026-09-30 04:54 — cycle 321 (fix-174)
  reconciled: CYCLE_OK. [04:47:57]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 30th row running kv=776 apc=
  498 ev 122395/30545); 30-row clean
  streak ([02:22:10]…[04:47:57];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  80 of 94 rows clean. ROWLOG_N=425.
  Cycle 322 dispatching (fresh lane).
- 2026-09-30 05:00 — cycle 322 (fix-175)
  reconciled: CYCLE_OK. 2 new rows
  [04:52:58] and [04:57:59] (fb=136
  kv=777 apc=503 ev 122983/30701)
  clean — 32-row clean streak
  ([02:22:10]…[04:57:59]; recounted
  from file rows); no new hb=000 rows
  (still 14 wedges, nothing since
  [02:17:09]); no WALL LOST; fb flat
  (no fallback events). Epoch I: 82 of
  96 rows clean. ROWLOG_N=427. Cycle
  323 dispatching (fresh lane).
- 2026-09-30 05:06 — cycle 323 (fix-176)
  reconciled: CYCLE_OK. [05:02:59]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 33rd row running kv=777 apc=
  510 ev 123015/30733); 33-row clean
  streak ([02:22:10]…[05:02:59];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  83 of 97 rows clean. ROWLOG_N=428.
  Cycle 324 dispatching (fresh lane).
- 2026-09-30 05:11 — cycle 324 (fix-177)
  reconciled: CYCLE_OK. [05:08:00]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 34th row running kv=780 apc=
  513 ev 124767/31126); 34-row clean
  streak ([02:22:10]…[05:08:00];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  84 of 98 rows clean. ROWLOG_N=429.
  Cycle 325 dispatching (fresh lane).
- 2026-09-30 05:18 — cycle 325 (fix-178)
  reconciled: CYCLE_OK. [05:13:01]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 35th row running kv=783 apc=
  516 ev 125995/31435); 35-row clean
  streak ([02:22:10]…[05:13:01];
  recounted from file rows; lane prose
  echoed '36' — echo slip, file rows
  authoritative); no new hb=000 rows
  (still 14 wedges, nothing since
  [02:17:09]); no WALL LOST; fb flat
  (no fallback events). Epoch I: 85 of
  99 rows clean. ROWLOG_N=430. Cycle
  326 dispatching (fresh lane).
- 2026-09-30 05:23 — cycle 326 (fix-179)
  reconciled: CYCLE_OK. [05:18:01]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 36th row running kv=784 apc=
  522 ev 126031/31471); 36-row clean
  streak ([02:22:10]…[05:18:01];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  86 of 100 rows clean. ROWLOG_N=431.
  Cycle 327 dispatching (fresh lane).
- 2026-09-30 05:29 — cycle 327 (fix-180)
  reconciled: CYCLE_OK. [05:23:02]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 37th row running kv=784 apc=
  529 ev 126063/31503); 37-row clean
  streak ([02:22:10]…[05:23:02];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  87 of 101 rows clean. ROWLOG_N=432.
  Cycle 328 dispatching (fresh lane).
- 2026-09-30 05:35 — cycle 328 (fix-181)
  reconciled: CYCLE_OK. [05:28:02]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 38th row running kv=787 apc=
  532 ev 126803/31623); 38-row clean
  streak ([02:22:10]…[05:28:02];
  recounted from file rows; longest
  clean run of epoch I); no new hb=000
  rows (still 14 wedges, nothing since
  [02:17:09]); no WALL LOST; fb flat
  (no fallback events). Epoch I: 88 of
  102 rows clean. ROWLOG_N=433. Cycle
  329 dispatching (fresh lane).
- 2026-09-30 05:41 — cycle 329 (fix-182)
  reconciled: CYCLE_OK. 2 new rows
  [05:33:03] and [05:38:04] (fb=136
  kv=789 apc=537 ev 127491/31724)
  clean — 40-row clean streak
  ([02:22:10]…[05:38:04], epoch I's
  longest; recounted from file rows); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 90 of 104 rows clean.
  ROWLOG_N=435. Cycle 330 dispatching
  (fresh lane).
- 2026-09-30 05:46 — cycle 330 (fix-183)
  reconciled: CYCLE_OK. [05:43:04]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 41st row running kv=790 apc=
  543 ev 127519/31726); 41-row clean
  streak ([02:22:10]…[05:43:04], new
  epoch I record; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 91 of 105 rows
  clean. ROWLOG_N=436. Cycle 331
  dispatching (fresh lane).
- 2026-09-30 05:52 — cycle 331 (fix-184)
  reconciled: CYCLE_OK. [05:48:05]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 42nd row running kv=791 apc=
  549 ev 127547/31739); 42-row clean
  streak ([02:22:10]…[05:48:05], new
  epoch I record; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 92 of 106 rows
  clean. ROWLOG_N=437. Cycle 332
  dispatching (fresh lane).
- 2026-09-30 05:58 — cycle 332 (fix-185)
  reconciled: CYCLE_OK. [05:53:06]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 43rd row running kv=793 apc=
  552 ev 128995/32103); 43-row clean
  streak ([02:22:10]…[05:53:06],
  record still growing; recounted from
  file rows); no new hb=000 rows (still
  14 wedges, nothing since [02:17:09]);
  no WALL LOST; fb flat (no fallback
  events). Epoch I: 93 of 107 rows
  clean. ROWLOG_N=438. Cycle 333
  dispatching (fresh lane).
- 2026-09-30 06:04 — cycle 333 (fix-186)
  reconciled: CYCLE_OK. [05:58:06]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 44th row running kv=794 apc=
  557 ev 129763/32297); 44-row clean
  streak ([02:22:10]…[05:58:06];
  recounted from file rows); no new
  hb=000 rows (still 14 wedges, nothing
  since [02:17:09]); no WALL LOST; fb
  flat (no fallback events). Epoch I:
  94 of 108 rows clean. ROWLOG_N=439.
  Cycle 334 dispatching (fresh lane).
- 2026-09-30 06:10 — cycle 334 (fix-187)
  reconciled: CYCLE_OK. 2 new rows
  [06:03:14] and [06:08:15] (fb=136
  kv=795 apc=562 ev 130539/32489)
  clean — 46-row clean streak
  ([02:22:10]…[06:08:15], record
  growing; recounted from file rows); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 96 of 110 rows clean.
  ROWLOG_N=441. Cycle 335 dispatching
  (fresh lane).
- 2026-09-30 06:16 — cycle 335 (fix-188)
  reconciled: CYCLE_OK. [06:13:15]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 47th row running kv=797 apc=
  567 ev 131327/32729); 47-row clean
  streak ([02:22:10]…[06:13:15],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 97 of 111 rows
  clean. ROWLOG_N=442. Cycle 336
  dispatching (fresh lane).
- 2026-09-30 06:23 — cycle 336 (fix-189)
  reconciled: CYCLE_OK. [06:18:16]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 48th row running kv=799 apc=
  570 ev 132919/33098); 48-row clean
  streak ([02:22:10]…[06:18:16],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 98 of 112 rows
  clean. ROWLOG_N=443. Cycle 337
  dispatching (fresh lane).
- 2026-09-30 06:28 — cycle 337 (fix-190)
  reconciled: CYCLE_OK. [06:23:16]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 49th row running kv=800 apc=
  576 ev 132951/33098); 49-row clean
  streak ([02:22:10]…[06:23:16],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 99 of 113 rows
  clean. ROWLOG_N=444. Cycle 338
  dispatching (fresh lane).
- 2026-09-30 06:35 — cycle 338 (fix-191)
  reconciled: CYCLE_OK. 2 new rows
  [06:28:17] and [06:33:18] (fb=136
  kv=801 apc=581 ev 133899/33358)
  clean — 51-row clean streak
  ([02:22:10]…[06:33:18], record
  growing; recounted from file rows); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 101 of 115 rows clean.
  ROWLOG_N=446. Cycle 339 dispatching
  (fresh lane).
- 2026-09-30 06:41 — cycle 339 (fix-192)
  reconciled: CYCLE_OK. [06:38:18]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 52nd row running kv=802 apc=
  586 ev 134287/33452); 52-row clean
  streak ([02:22:10]…[06:38:18],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 102 of 116 rows
  clean. ROWLOG_N=447. Cycle 340
  dispatching (fresh lane).
- 2026-09-30 06:47 — cycle 340 (fix-193)
  reconciled: CYCLE_OK. [06:43:19]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 53rd row running kv=804 apc=
  590 ev 134683/33514); 53-row clean
  streak ([02:22:10]…[06:43:19],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 103 of 117 rows
  clean. ROWLOG_N=448. Cycle 341
  dispatching (fresh lane).
- 2026-09-30 06:53 — cycle 341 (fix-194)
  reconciled: CYCLE_OK. [06:48:20]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 54th row running kv=805 apc=
  595 ev 135099/33646); 54-row clean
  streak ([02:22:10]…[06:48:20],
  record growing; recounted from file
  rows); lane echo of historic [06:28:17]
  read kv=880 vs file rows kv=800;
  corrected (echo slip, file rows
  authoritative; older-than-newest, no
  verdict impact); no new hb=000 rows
  (still 14 wedges, nothing since
  [02:17:09]); no WALL LOST; fb flat
  (no fallback events). Epoch I: 104 of
  118 rows clean. ROWLOG_N=449. Cycle
  342 dispatching (fresh lane).
- 2026-09-30 06:59 — cycle 342 (fix-195)
  reconciled: CYCLE_OK. [06:53:20]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 55th row running kv=806 apc=
  600 ev 135539/33807); 55-row clean
  streak ([02:22:10]…[06:53:20],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 105 of 119 rows
  clean. ROWLOG_N=450. Cycle 343
  dispatching (fresh lane).
- 2026-09-30 07:05 — cycle 343 (fix-196)
  reconciled: CYCLE_OK. [06:58:21]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 56th row running kv=806 apc=
  607 ev 135575/33807); 56-row clean
  streak ([02:22:10]…[06:58:21],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 106 of 120 rows
  clean. ROWLOG_N=451. Cycle 344
  dispatching (fresh lane).
- 2026-09-30 07:10 — cycle 344 (fix-197)
  reconciled: CYCLE_OK. 2 new rows
  [07:03:35] and [07:08:35] (fb=136
  kv=809 apc=611 ev 136419/33977)
  clean — 58-row clean streak
  ([02:22:10]…[07:08:35], record
  growing; recounted from file rows); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 108 of 122 rows clean.
  ROWLOG_N=453. Cycle 345 dispatching
  (fresh lane).
- 2026-09-30 07:17 — cycle 345 (fix-198)
  reconciled: CYCLE_OK. [07:13:36]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 59th row running kv=812 apc=
  614 ev 137327/34246); 59-row clean
  streak ([02:22:10]…[07:13:36],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 109 of 123 rows
  clean. ROWLOG_N=454. Cycle 346
  dispatching (fresh lane).
- 2026-09-30 07:23 — cycle 346 (fix-199)
  reconciled: CYCLE_OK. [07:18:36]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 60th row running kv=813 apc=
  619 ev 137823/34393); 60-row clean
  streak ([02:22:10]…[07:18:36],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 110 of 124 rows
  clean. ROWLOG_N=455. Cycle 347
  dispatching (fresh lane).
- 2026-09-30 07:29 — cycle 347 (fix-200)
  reconciled: CYCLE_OK. [07:23:37]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 61st row running kv=813 apc=
  626 ev 137859/34393); 61-row clean
  streak ([02:22:10]…[07:23:37],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 111 of 125 rows
  clean. ROWLOG_N=456. Cycle 348
  dispatching (fresh lane).
- 2026-09-30 07:34 — cycle 348 (fix-201)
  reconciled: CYCLE_OK. [07:28:38]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 62nd row running kv=815 apc=
  628 ev 139358/34773); 62-row clean
  streak ([02:22:10]…[07:28:38],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 112 of 126 rows
  clean. ROWLOG_N=457. Cycle 349
  dispatching (fresh lane).
- 2026-09-30 07:41 — cycle 349 (fix-202)
  reconciled: CYCLE_OK. 2 new rows
  [07:33:42] and [07:38:42] (fb=136
  kv=816 apc=633 ev 139914/34794)
  clean — 64-row clean streak
  ([02:22:10]…[07:38:42], record
  growing; recounted from file rows); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 114 of 128 rows clean.
  ROWLOG_N=459. Cycle 350 dispatching
  (fresh lane).
- 2026-09-30 07:46 — cycle 350 (fix-203)
  reconciled: CYCLE_OK. [07:43:43]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 65th row running kv=818 apc=
  638 ev 140462/34976); 65-row clean
  streak ([02:22:10]…[07:43:43],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 115 of 129 rows
  clean. ROWLOG_N=460. Cycle 351
  dispatching (fresh lane).
- 2026-09-30 07:52 — cycle 351 (fix-204)
  reconciled: CYCLE_OK. [07:48:44]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 66th row running kv=818 apc=
  645 ev 140490/34979); 66-row clean
  streak ([02:22:10]…[07:48:44],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 116 of 130 rows
  clean. ROWLOG_N=461. Cycle 352
  dispatching (fresh lane).
- 2026-09-30 07:57 — cycle 352 (fix-205)
  reconciled: CYCLE_OK. [07:53:44]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 67th row running kv=819 apc=
  651 ev 141066/35150); 67-row clean
  streak ([02:22:10]…[07:53:44],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 117 of 131 rows
  clean. ROWLOG_N=462. Cycle 353
  dispatching (fresh lane).
- 2026-09-30 08:03 — cycle 353 (fix-206)
  reconciled: CYCLE_OK. [07:58:45]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 68th row running kv=821 apc=
  656 ev 141674/35338); 68-row clean
  streak ([02:22:10]…[07:58:45],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 118 of 132 rows
  clean. ROWLOG_N=463. Cycle 354
  dispatching (fresh lane).
- 2026-09-30 08:09 — cycle 354 (fix-207)
  reconciled: CYCLE_OK. [08:03:46]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 69th row running kv=821 apc=
  663 ev 141710/35338); 69-row clean
  streak ([02:22:10]…[08:03:46],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 119 of 133 rows
  clean. ROWLOG_N=464. Cycle 355
  dispatching (fresh lane).
- 2026-09-30 08:14 — cycle 355 (fix-208)
  reconciled: CYCLE_OK. [08:08:46]
  newest clean (h=200 p=9 wall=200 fb=
  136 FLAT 70th row running kv=822 apc=
  670 ev 142234/35437); 70-row clean
  streak ([02:22:10]…[08:08:46],
  record growing; recounted from file
  rows); no new hb=000 rows (still 14
  wedges, nothing since [02:17:09]); no
  WALL LOST; fb flat (no fallback
  events). Epoch I: 120 of 134 rows
  clean. ROWLOG_N=465. Cycle 356
  dispatching (fresh lane).
- 2026-09-30 08:20 — cycle 356 (fix-209)
  reconciled: CYCLE_OK. 2 new rows
  [08:13:47] and [08:18:47] (fb=136
  kv=824 apc=679 ev 146314/36421)
  clean — 72-row clean streak
  ([02:22:10]…[08:18:47], record
  growing; recounted from file rows); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST; fb flat (no fallback events).
  Epoch I: 122 of 136 rows clean.
  ROWLOG_N=467. Cycle 357 dispatching
  (fresh lane).
- 2026-09-30 08:29 — cycle 357 (fix-210)
  reconciled: CYCLE_OK. [08:23:48]
  newest clean (h=200 p=9 wall=200; no
  ALERT tag; kv=841 apc=692 ev
  148370/36898); fb ROSE 136→141 on
  [08:23:48] — first fallback-EVENT
  delta since [02:22:10] (69 flat rows
  [02:22:10]…[08:18:47]); +5 events
  under heavy load (bytes +54257):
  aggregate load drift per de-armed
  semantics, NOT an escalation (no
  wedge, no hb=000, no wall loss);
  73-row clean streak
  ([02:22:10]…[08:23:48], record
  growing; recounted from file rows); no
  new hb=000 rows (still 14 wedges,
  nothing since [02:17:09]); no WALL
  LOST. Epoch I: 123 of 137 rows clean.
  ROWLOG_N=468. Cycle 358 dispatching
  (fresh lane).
- 2026-09-30 08:39 — cycle 358 (fix-211)
  reconciled: CYCLE_OK. [08:34:09]
  newest clean (h=200 p=9 wall=200 fb=
  148 kv=863 apc=703 ev 153021/38105);
  NEW WEDGE row [08:29:08] (hb=000,
  fb=142, ALERT SERVE-WEDGE) — epoch I
  wedge #15, first since [02:17:09]
  (6h12m quiet, epoch's longest gap);
  1-row stick released next row:
  bounded ladder verdict upheld. 73-row
  clean streak ([02:22:10]…[08:23:48])
  ended by the wedge; new tail streak
  1 row ([08:34:09]); no ≥4 stick, no
  user-visible alarm; no WALL LOST; fb
  141→148 across the window (wedge row
  +1, release row +6 under heavy load
  bytes +42081): aggregate drift, NOT
  an escalation; no WALL LOST.
  Epoch I: 124 of 139 rows clean
  (15 wedges). ROWLOG_N=470. Cycle 359
  dispatching (fresh lane).
- 2026-09-30 08:48 — cycle 359 (fix-212)
  reconciled: CYCLE_ALERT: SERVE-WEDGE
  (2-row stick IN PROGRESS — newest row
  tagged). NEW wedge rows [08:39:29]
  (h=000 fb=153) and [08:44:50]
  (h=000 fb=162) — consecutive
  hb=000, run-length 2, OPEN at probe
  time (no release row yet); the
  1-row stick [08:29:08] had released
  clean [08:34:09] (fb 148, kv=863)
  ending the prior 73-row streak; wedge
  cadence tightened (~5m20s after
  release vs 6h12m prior gap);
  heavy-load correlate (bytes ~+53K on
  stick rows, ev 156705/38949); NO
  WALL LOST, no wall=000; run=2 < ≥4
  user-visible alarm (ties epoch-I
  2-row stick record form; next row is
  the 3-tie / ≥4 watch point); fb
  148→162 (+14 window, row-proved
  aggregate drift). Epoch I: 124 of
  141 clean (17 wedge rows). ROWLOG_N=
  472. Cycle 360 dispatching (fresh
  lane).
- 2026-09-30 09:08 — cycle 360
  (fix-213) reconciled: CYCLE_ALERT:
  SERVE-WEDGE 🔴 USER-VISIBLE ALARM.
  Stick EXTENDED to run-length 5
  consecutive hb=000 rows ([08:39:29],
  [08:44:50], [08:50:10], [08:55:31],
  [09:00:51]; ≥4 threshold crossed at
  [08:55:31]) — past epoch-I worst
  (3-row stick); newest [09:00:51]
  (h=200 p=9 wall=200 fb=179 kv=991
  apc=721 ev 161321/40091); NO WALL
  LOST, no wall=000, BOOTCOUNT=1.
  Direct deep-dive 09:06 (read-only):
  ESCAPE_COUNT 18→36 (18 new in ~1 h
  load wave), ALL UNIFORM closes
  (walls 10.4–136.8 s, fallbacks≤2),
  RELEASE_COUNT=0 (240 s cap never
  reached) — engine ladder verified
  BOUNDED throughout the stick;
  KVL_COUNT=1019, FALLBACK_COUNT=183,
  KV-LOOKUP heartbeat healthy 09:02–09:04.
  Heavy-load correlate: bytes +50–62K
  per row since ~08:05; user notified
  per policy; cadence continues, watch
  release next cycle. Epoch I: 124 of
  144 clean (20 wedge rows). ROWLOG_N=
  475. Cycle 361 dispatching (fresh
  lane).
- 2026-09-30 09:25 — cycle 361
  (fix-214) reconciled: CYCLE_ALERT:
  SERVE-WEDGE (stick EXTENDED — newest
  row tagged). Run-length now 8
  consecutive hb=000 rows ([08:39:29],
  [08:44:50], [08:50:10], [08:55:31],
  [09:00:51], [09:06:11], [09:11:32],
  [09:16:52]); 3 new rows; newest
  [09:16:52] (h=200 p=9 wall=200 fb=
  200 kv=1071 apc=721(F) ev
  165004/41083); NO WALL LOST, no
  wall=000, BOOTCOUNT=1. Second direct
  deep-dive 09:25: ESCAPE_COUNT 36→43
  (+7 more UNIFORM closes, walls
  10.4–21.1 s, every episode still
  fallbacks=2), RELEASE_COUNT=0 (cap
  never reached), FALLBACK_COUNT=
  183→208, KV-LOOKUP hits healthy
  09:19–09:23, no fallbacks=3 lines —
  ladder remains fully bounded through
  the stick; heater load persists
  (+47–60K bytes/row). Alarm posture:
  user-visible alarm continues; cadence
  continues; no action taken. Epoch I:
  124 of 147 clean (23 wedge rows).
  ROWLOG_N=478. Cycle 362 dispatching
  (fresh lane).
- 2026-09-30 09:40 — cycle 362
  (fix-215) reconciled: CYCLE_ALERT:
  SERVE-WEDGE (stick EXTENDED — newest
  row tagged). Run-length now 11
  consecutive hb=000 rows ([08:39:29]
  →[09:32:53]); 3 new rows; newest
  [09:32:53] (h=200 p=9 wall=200 fb=
  213 kv=1142 apc=721(F) ev
  170384/42361); NO WALL LOST, no
  wall=000, BOOTCOUNT=1. Third direct
  deep-dive 09:40: ESCAPE_COUNT 43→45
  (+2, both fallbacks=2 wall=10.4 s
  UNIFORM), FALLBACK_COUNT 208→214,
  RELEASE_COUNT=0 (cap never reached),
  no fallbacks=3 lines, KV-LOOKUP hits
  healthy through 09:37 — ladder holds
  bounded across the whole stick.
  apc flat 721 for 7 rows; heavy load
  persists (+48–64K bytes/row); fb
  drift 200→213. User-visible alarm
  continues; vigilance only. Epoch I:
  124 of 150 clean (26 wedge rows).
  ROWLOG_N=481. Cycle 363 dispatching
  (fresh lane).
- 2026-09-30 09:47 — cycle 363
  (fix-216) reconciled: CYCLE_OK —
  ALARM CLEARED. The 12-row stick
  (12 consecutive hb=000 rows,
  [08:39:29]→[09:38:14], the
  epoch-I record: ~64 min) released
  with TWO final rows in-window:
  [09:38:14] (h=000 fb=214 apc=734)
  then [09:43:21] (h=200 p=9 wall=200
  fb=215 kv=1162 apc=840(F), ev
  175804/43687; +87208 bytes = window
  max). Ladder held bounded through the
  whole stick (3 dives: 45 escapes,
  0 releases, cap never reached, no
  fallbacks=3, BOOTCOUNT=1); finished
  aggregate fb drift 153→215 during
  wedge cluster. Alarm does not
  re-fire; no WALL LOST / wall=000
  anywhere. Heavy load past the stick
  (+62–87K bytes). NEW clean tail row
  [09:43:21] = newest; watch for
  relapse under sustained load. Epoch
  I: 125 of 152 clean (27 wedge rows).
  ROWLOG_N=483. Cycle 364 dispatching
  (fresh lane).
- 2026-09-30 09:58 — cycle 364
  (fix-217) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — RE-ENTRY. A NEW stick
  opened 5:20 after the record-stick
  release (matches epoch cadence);
  clean tail lasted exactly 1 row
  ([09:43:21]). Two new wedge rows:
  [09:48:41] (h=000 fb=224 kv=1193
  apc=841) + [09:54:01] (h=000 fb=230
  kv=1217 apc=843, ev 181444/45095) =
  tail hb=000 run-length 2, OPEN at
  probe time (below the ≥4 alarm
  threshold, not re-escalated). No
  WALL LOST / wall=000 anywhere. No
  new deep-dive (run<3 gate; 09:40
  dive verified ladder bounded, 0
  releases). fb 215→230 on the wedge
  rows; apc slow-drifting 840→843;
  heavy load persists (+57–59K
  bytes/row). Vigilance: run≥4
  re-escalates the user-visible alarm.
  Epoch I: 125 of 154 clean (29
  wedge rows). ROWLOG_N=485. Cycle
  365 dispatching (fresh lane).
- 2026-09-30 10:12 — cycle 365
  (fix-218) reconciled: CYCLE_OK —
  dense-cadence wedges all bounded.
  3 new rows: 2-row re-entry stick
  [09:48:41]+[09:54:01] RELEASED at
  [09:59:09] (h=200 p=9 wall=200
  fb=239 apc=843); an immediate
  follow-up 1-row wedge [10:04:29]
  (h=000 fb=240 apc=844) RELEASED
  at [10:09:40] (h=200 p=9 wall=200
  fb=241 kv=1281 apc=858, ev
  189668/47040) = new clean tail,
  1-row streak, no stick open. Wedge
  slots alternating with 1-row
  release streaks every ~5:20 under
  the heavy wave — cadence dense but
  no stick extended toward ≥4. No
  WALL LOST / wall=000 anywhere; no
  new deep-dive (bounded ≤2-row
  sticks; trio of dives verified
  ladder bounded). fb 230→241;
  apc s843→858. Epoch I: 127 of 157
  clean (30 wedge rows). ROWLOG_N=
  488. Cycle 366 dispatching (fresh
  lane).
- 2026-09-30 10:21 — cycle 366
  (fix-219) reconciled: CYCLE_OK —
  wedge slot SKIPPED CLEAN. 1 new
  clean row: [10:14:57] (h=200 p=9
  wall=200 fb=247 kv=1310 apc=863,
  ev 193278/48018); clean tail now
  2 consecutive rows ([10:09:40],
  [10:14:57]) — first extended post-
  record-release streak; the ~5:20
  cadence slot passed with NO wedge
  (dense regime absorbing the load).
  No WALL LOST / wall=000 anywhere;
  fb 241→247 (aggregate); apc
  858→863; heavy band persists
  (+57–64K rows/row window). Epoch
  I: 128 of 158 clean (30 wedge
  rows). ROWLOG_N=489. Cycle 367
  dispatching (fresh lane).
- 2026-09-30 10:33 — cycle 367
  (fix-220) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — NEW 2-row stick
  OPEN (clean streak ended after
  [10:14:57]; no release row yet):
  [10:20:17] (h=000 fb=255 apc=870)
  + [10:25:38] (h=000 fb=267 kv=
  1378 apc=871, ev 197126/48920).
  No WALL LOST / wall=000 anywhere;
  fb 247→267 (+20 across the stick
  rows, drift); apc 863→871; heavy
  band persists (+54–57K/brow). Stick
  run=2: below the 3-row old-worst
  tie and the ≥4 alarm threshold —
  next rows decide; no new deep-dive
  (run<3 gate; trio of dives verified
  bounded). Epoch I: 128 of 160
  clean (32 wedge rows). ROWLOG_N=
  491. Cycle 368 dispatching (fresh
  lane).
- 2026-09-30 10:44 — cycle 368
  (fix-221) reconciled: CYCLE_OK —
  OLD-WORST TIE THEN RELEASE. The
  open stick extended to run 3 at
  [10:30:58] (h=000 fb=275 apc=871)
  = epoch-I pre-record 3-row worst
  re-tell (loud but not the alarm),
  then RELEASED at [10:36:03]
  (h=200 p=9 wall=200 fb=282 kv=
  1432 apc=871, ev 202022/50073)
  BEFORE run 4 — alarm NOT re-raised;
  bounded again. Latest [10:41:10]
  (h=200 p=9 wall=200 fb=285 kv=
  1447 apc=885, ev 204658/50714)
  = clean tail 2 rows ([10:36:03],
  [10:41:10]). No WALL LOST /
  wall=000 anywhere; fb 267→285
  aggregate; apc 871→885; heavy
  band mostly (+50–56K, one +46K
  dip unconfirmed as cool-down).
  Epoch I: 130 of 163 clean (33
  wedge rows). ROWLOG_N=494. Cycle
  369 dispatching (fresh lane).
- 2026-09-30 10:53 — cycle 369
  (fix-222) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — 1-row wedge opened
  after the 2-row clean tail
  ([10:36:03]+[10:41:10]) ended:
  [10:46:30] (h=000 fb=292 kv=1474
  apc=889, ev 206714/51207) tagged
  = newest row; tail hb=000 run-
  length 1, below both the 3-row
  old-worst tie and the ≥4 alarm
  gate. No WALL LOST / wall=000
  anywhere; fb 285→292 (aggregate);
  apc 885→889; heavy band persists
  (+55–56K/row). Dense-but-bounded
  regime continues; next rows decide
  release (bounded 1–2-row pattern)
  vs stick extension toward tie or
  alarm. Epoch I: 130 of 164 clean
  (34 wedge rows). ROWLOG_N=495.
  Cycle 370 dispatching (fresh
  lane).
- 2026-09-30 11:02 — cycle 370
  (fix-223) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — OLD-WORST TIE #2.
  The 1-row wedge extended to a
  3-row stick: [10:46:30] + [10:51:51]
  (h=000 fb=293 apc=904) + [10:57:11]
  (h=000 fb=298 kv=1506 apc=908, ev
  213246/52854) — tail run 3 re-tells
  the pre-record 3-row worst (loud,
  NOT the alarm); last hb=200 row
  was [10:41:10] (no release since);
  bounded regime still intact. No
  WALL LOST / wall=000 anywhere; fb
  292→298 (aggregate); apc 889→908;
  heavy band persists (+57–64K/row).
  Next slot decides: 4th consecutive
  hb=000 row = ALARM vs 1 untagged
  hb=200 row = release (bounded).
  Epoch I: 130 of 166 clean (36
  wedge rows). ROWLOG_N=497. Cycle
  371 dispatching (fresh lane).
- 2026-09-30 11:14 — cycle 371
  (fix-224) reconciled: CYCLE_OK —
  ALARM THRESHOLD TOUCHED, STICK
  RELEASED BOUNDEDLY. The 3-row tie
  stick extended to run 4 at
  [11:02:31] (h=000 fb=307 apc=909)
  — the ≥4 user-visible alarm
  threshold was TOUCHED (warning
  delivered) — then the stick
  RELEASED at [11:07:32] (h=200
  p=9 wall=200 fb=310 kv=1555
  apc=915, ev 223080/55195) BEFORE
  run 5; newest row clean. NEW
  epoch-I bounded-stick worst = 4
  rows (well below the 12-row
  record). Fourth direct deep-dive
  11:12: ESCAPE_COUNT 45→57 (+12
  UNIFORM closes, walls 10.4 s in
  the stick window), RELEASE_COUNT=
  0 — 240 s cap verified NEVER
  reached THROUGH the run-4 stick;
  FALLBACK_COUNT 214→310, no
  fallbacks=3 lines, BOOTCOUNT=1,
  KV-LOOKUP hits healthy. No WALL
  LOST / wall=000 anywhere. fb
  298→310 aggregate; apc 908→915;
  load heavy with marginal cooling
  at release (+43018). Epoch I:
  131 of 168 clean (37 wedge
  rows). ROWLOG_N=499. Cycle 372
  dispatching (fresh lane).
- 2026-09-30 11:19 — cycle 372
  (fix-225) reconciled: CYCLE_OK —
  clean streak extended to 2
  ([11:07:32]+[11:12:39]); NEW
  clean row [11:12:39] (h=200 p=9
  wall=200 fb=310 kv=1564 apc=935,
  ev 225884/55920). NO NEW WEDGE —
  FIRST fb-hold since the wedge
  cluster: fb HELD FLAT at 310
  across the whole window (zero
  aggregate drift, row-proved)
  while apc kept climbing
  915→935 and the heavy band
  persisted (+60439). Ladder fully
  bounded (through run-4 stick per
  dive #4); no WALL LOST /
  wall=000 anywhere; no alarm
  touch this cycle. Epoch I: 132
  of 169 clean (37 wedge rows).
  ROWLOG_N=500. Cycle 373
  dispatching (fresh lane).
- 2026-09-30 11:26 — cycle 373
  (fix-226) reconciled: CYCLE_OK —
  clean streak extended to 4
  ([11:07:32]→[11:23:04]); 2 new
  clean rows [11:17:51] (h=200 p=9
  wall=200 fb=310 kv=1573 apc=945,
  ev 231630/57378) + [11:23:04]
  (h=200 p=9 wall=200 fb=310 kv=
  1586 apc=960, ev 237028/58681;
  rmax 1→0 tick, benign). FB HOLD
  continues: flat 310 across 4
  consecutive windows since the
  release — zero aggregate drift
  while the heavy band persists
  (+64–66K/row, near band top). No
  WALL LOST / wall=000 anywhere; no
  wedge, no alarm touch. Ladder
  bounded (dive #4 verified incl.
  the run-4 stick). Epoch I: 134 of
  171 clean (37 wedge rows).
  ROWLOG_N=502. Cycle 374
  dispatching (fresh lane).
- 2026-09-30 11:34 — cycle 374
  (fix-227) reconciled: CYCLE_OK —
  clean streak extended to 5
  ([11:07:32]→[11:28:04]): NEW
  clean row [11:28:04] (h=200 p=9
  wall=200 fb=316 kv=1608 apc=969,
  ev 240142/59463). FB HOLD at 310
  BROKE: drift resumed (+6 → 316,
  aggregate, after the 4-window
  flat hold). apc 960→969; bytes
  marginal off the top band
  (+58045). No WALL LOST /
  wall=000 anywhere; no new wedge;
  no alarm touch. Epoch I: 135 of
  172 clean (37 wedge rows).
  ROWLOG_N=503. Cycle 375
  dispatching (fresh lane).
- 2026-09-30 11:42 — cycle 375
  (fix-228) reconciled: CYCLE_OK —
  the ~11:33 wedge due-slot fired
  on cadence: 1-row bounded stick
  [11:33:25] (h=000 fb=320 kv=
  1622 apc=982, ev 241781/59860)
  RELEASED within the window at
  [11:38:44] (h=200 p=9 wall=200
  fb=321 kv=1630 apc=992, ev
  247139/61169); 5-row streak
  [11:07:32]→[11:28:04] ended; new
  clean tail 1-row. No WALL LOST /
  wall=000 anywhere; no alarm
  touch; fb 316→320→321 (pre-heat-
  scale +4/+1); apc 969→992; bytes
  cooling modestly (+52K/+48K, off
  the 64–66K top band). Epoch I:
  136 of 174 clean (38 wedge
  rows). ROWLOG_N=505. Cycle 376
  dispatching (fresh lane).
- 2026-09-30 11:51 — cycle 376
  (fix-229) reconciled: CYCLE_OK —
  the ~11:44 due-slot fired CLEAN
  (no wedge): clean streak
  extended to 2 ([11:38:44]+[11:43:58]);
  NEW clean row [11:43:58] (h=200
  p=9 wall=200 fb=326 kv=1650
  apc=996, ev 251918/62438). fb
  321→326 (+5 steady small-scale,
  no pre-heat spike); apc 992→996;
  bytes ~49K plateau (no re-heat,
  still elevated vs +15–25K norm).
  No WALL LOST / wall=000 anywhere;
  no alarm touch. Epoch I: 137 of
  175 clean (38 wedge rows).
  ROWLOG_N=506. Cycle 377
  dispatching (fresh lane).
- 2026-09-30 12:00 — cycle 377
  (fix-230) reconciled: CYCLE_OK —
  the ~11:49 due-slot opened
  another 1-row bounded wedge
  [11:49:19] (h=000 fb=328 kv=1662
  apc=1007, ev 257838/63943) after
  the 2-row streak ended; RELEASED
  in-window at [11:54:33] (h=200
  p=9 wall=200 fb=333 kv=1679
  apc=1023, ev 259202/64277) =
  newest row clean (1-row clean
  tail, no stick open). fb 326→
  328→333 (+2/+5 small-scale); apc
  996→1023; bytes plateau (+48–52K);
  cadence of alternating 1-row
  wedge/release slots intact. No
  WALL LOST / wall=000 anywhere; no
  alarm touch. Epoch I: 138 of 177
  clean (39 wedge rows). ROWLOG_N=
  508. Cycle 378 dispatching (fresh
  lane).
- 2026-09-30 12:09 — cycle 378
  (fix-231) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — NEW 2-row stick
  OPEN (alternating 1-row pattern
  broke into a 2-row stick): [11:59:54]
  (h=000 fb=338 apc=1031) + [12:05:14]
  (h=000 fb=352 kv=1739 apc=1031(F),
  ev 263702/65350) — no release row
  yet at probe time. No WALL LOST /
  wall=000 anywhere; no alarm touch
  (run 2 < 3-lineage, <4 threshold).
  fb 333→338→352 (the +14 step at
  [12:05:14] was the largest recent
  gain, during the stick); apc flat
  1031 across the wedge rows; bytes
  re-heating mildly (+55K/+59K off
  the 48–52K plateau). Epoch I: 138
  of 179 clean (41 wedge rows).
  ROWLOG_N=510. Cycle 379
  dispatching (fresh lane).
- 2026-09-30 12:20 — cycle 379
  (fix-232) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — ALARM THRESHOLD
  RE-TOUCHED. The 2-row stick
  extended to run 4: tail [11:59:54],
  [12:05:14], [12:10:35] (h=000 fb=362
  apc=1033), [12:15:55] (h=000 fb=370
  kv=1796 apc=1036, ev 267259/66106)
  — NO release row yet at probe time
  (matches the epoch bounded-stick
  worst from [10:46:30]→[11:02:31]).
  Fifth direct deep-dive 12:18:
  ESCAPE_COUNT 57→62 (+5 incl.
  in-stick closes 12:02–12:04 and
  12:15:47, ALL walls 10.4 s at
  fallbacks=2), RELEASE_COUNT=0 —
  240 s cap remains NEVER reached
  even through this OPEN run-4 stick;
  FALLBACK_COUNT 310→375; KV-LOOKUP
  hits healthy 12:13–12:18 incl.
  fallbacks=1 walls 0.2–0.3 recovers;
  BOOTCOUNT=1. fb steps large in-
  stick: +14/+10/+8; apc 1031→1036;
  bytes cooled back to the 48–52K
  band. No WALL LOST / wall=000
  anywhere. Epoch I: 138 of 181
  clean (43 wedge rows). ROWLOG_N=
  512. Cycle 380 dispatching (fresh
  lane).
- 2026-09-30 12:32 — cycle 380
  (fix-233) reconciled: CYCLE_OK —
  threshold-touched-and-released:
  the OPEN run-4 stick extended to
  run 5 at [12:21:15] (h=000 fb=377
  apc=1037) = NEW epoch-I bounded-
  stick worst (past the 4-row worst,
  into the 12-row record lineage),
  then RELEASED at [12:26:20] (h=200
  p=9 wall=200 fb=380 kv=1846 apc=
  1047, ev 273305/67453) BEFORE run
  6. Sixth direct deep-dive 12:28
  across the run-5 portion + release:
  ESCAPE_COUNT HELD at 62 (ZERO new
  escape closes), FALLBACK_COUNT
  375→381 (+6, only fallbacks=1
  walls 0.2–5.2 self-recoveries),
  RELEASE_COUNT=0 — the 240 s cap
  remains NEVER reached even through
  the run-5 stick; KVL_COUNT→1855;
  hits healthy with near-perfect
  g0FAE convergence; BOOTCOUNT=1.
  fb in-stick deltas SMALLER this
  stick (+7/+3 vs prior +14/+10/+8,
  decelerating); apc 1036→1047;
  bytes 48–52K band w/ +59K re-heat
  at release. No WALL LOST /
  wall=000 anywhere. Epoch I: 139
  of 183 clean (44 wedge rows).
  ROWLOG_N=514. Cycle 381
  dispatching (fresh lane).
- 2026-09-30 12:38 — cycle 381
  (fix-234) reconciled: CYCLE_OK —
  ~12:31 due-slot fired CLEAN
  (skipped): clean streak extended
  to 2 ([12:26:20]+[12:31:26]); NEW
  clean row [12:31:26] (h=200 p=9
  wall=200 fb=384 kv=1875 apc=1049,
  ev 276713/68347). fb 380→384 (+4
  mild drift, deceleration regime
  continues); apc 1047→1049; bytes
  re-heat HELD at ~+60K for a second
  row (above the 48–52K band —
  cooling trend paused, no wedge
  formed anyway). No WALL LOST /
  wall=000 anywhere; no alarm
  touch. Ladder bounded (dive #6
  verified through run-5 + release:
  0 new escapes, cap 0). Epoch I:
  140 of 184 clean (44 wedge rows).
  ROWLOG_N=515. Cycle 382
  dispatching (fresh lane).
- 2026-09-30 12:47 — cycle 382
  (fix-235) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — NEW 2-row stick
  OPEN: [12:36:46] (h=000 fb=386
  apc=1061) + [12:42:06] (h=000
  fb=396 kv=1931 apc=1065, ev
  280857/69358) — the ~12:36 slot
  broke the 2-row streak; no release
  row yet. fb pace RE-ACCELERATED:
  +10 at [12:42:06] (largest per-row
  jump since the run-5 stick);
  apc 1049→1065; bytes re-heat
  persists (~+59–60K, third row
  above the 48–52K band; cooling
  paused — wedge-slot intensification
  correlate holding). No WALL LOST /
  wall=000 anywhere; no alarm touch
  (run 2 < 3-lineage, <4 threshold).
  Epoch I: 140 of 186 clean (46
  wedge rows). ROWLOG_N=517. Cycle
  383 dispatching (fresh lane).
- 2026-09-30 12:56 — cycle 383
  (fix-236) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — the open 2-row stick
  EXTENDED to run 3: NEW tagged row
  [12:47:27] (h=000 fb=404 kv=1956
  apc=1071, ev 282713/69727) after
  [12:36:46]+[12:42:06]; NO release
  row yet. Worst-lineage watch:
  run 3 entered; ONE more tagged row
  = run 4 = alarm re-touch (fourth
  touch lineage: 12-row record,
  4-row, 5-row). fb pace re-
  accelerated: deltas +2, +10, +8
  (396→404, run-5-stick-scale steps
  w/ slight decay vs peak);
  apc 1065→1071; bytes re-heat
  persists ~5 rows above the 48–52K
  band (ebbing ~57K); kv→1956; ev
  →282713/69727. No WALL LOST /
  wall=000 (h=200 all rows); ladder
  bounded (rep=0); canary alive,
  HEALTH=200. Epoch I: 140 of 187
  clean (47 wedge rows). ROWLOG_N=
  518. Cycle 384 dispatching (fresh
  lane).
- 2026-09-30 13:10 — cycle 384
  (fix-237) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — tail stick EXTENDED
  to run 6: 3 new tagged rows
  [12:52:47] (h=000 fb=410 — THE
  4TH ALARM RE-TOUCH at run 4),
  [12:58:08] (h=000 fb=420 kv=1980
  apc=1071 F), [13:03:28] (h=000
  fb=430 kv=2046 apc=1071 F, ev
  286017/70570). Depth now BEYOND
  every earlier bounded worst except
  the 12-row record (2nd-deepest
  stick of epoch; record lineage =
  12; no release row since
  [12:31:26]). fb +26 across the
  three rows (+6/+10/+10 from 404,
  run-5-stick scale, in-stick
  accel continues 386→430);
  apc PLATEAU 1071×3; bytes 8
  consecutive rows above the 48–52K
  band, ebbing 60.5→53.7K; Seventh
  direct deep-dive (13:05 through
  run-6 depth): ESCAPE_COUNT 62→77
  (+15, ALL UNIFORM ≤2-fallback
  escapes at walls 10.3–18.3 s incl
  18.3 s close 12:56:14);
  FALLBACK_COUNT 381→434 (+53);
  KVL→2059 (+204, largest yet);
  RELEASE_COUNT=0 — cap STILL never
  reached even through the run-6
  stick; no fallbacks=3; BOOTCOUNT
  =1; hits healthy w/ near-perfect
  g0FAE convergence. No WALL LOST /
  wall=000 anywhere. Epoch I: 140
  of 190 clean (50 wedge rows).
  ROWLOG_N=521. Cycle 385
  dispatching (fresh lane).
- 2026-09-30 13:19 — cycle 385
  (fix-238) reconciled: CYCLE_OK —
  threshold-touched-and-released:
  the stick extended to run 7 at
  [13:08:48] (h=000 fb=441 kv=2077,
  ev 286489/70734) then RELEASED at
  [13:13:49] (h=200 p=9 wall=200
  fb=444 kv=2089 apc=1078, ev
  287305/70930) = a 7-row released
  stick, the NEW post-record
  released-depth worst (beats the
  5-row prior worst; still below
  the 12-row epoch record; 4th
  touch lineage now: record-12,
  4-row, 5-row, 7-row — all
  released). fb +11 in-stick then
  +3 at release (rapid decel at
  release); apc plateau broke
  1071→1074→1078 (creep resumed);
  bytes 9th row above the band,
  then cooled exactly INTO it at
  release (+50714); kv→2089; ev
  →287305/70930; rmax=0 on the
  release row = first rmax≠1 since
  [11:23:04] = bounded-release
  drain correlate (benign). No
  WALL LOST / wall=000 anywhere;
  ladder bounded (cap 0 per dive
  #7). Epoch I: 141 of 192 clean
  (51 wedge rows). ROWLOG_N=523.
  Cycle 386 dispatching (fresh
  lane).
- 2026-09-30 13:29 — cycle 386
  (fix-239) reconciled: CYCLE_OK —
  2 NEW rows, wedge released
  in-window, threshold NOT touched
  (run 1): [13:19:09] h=000 (fb=449
  kv=2104 apc=1081, ev 291085/
  71812, bytes +38199 — FIRST
  below-band row since the 48–52K
  band formed ~11:33) 1-row wedge,
  THEN [13:24:15] CLEAN release
  (h=200 p=9 wall=200 fb=452 kv=
  2115 apc=1113, ev 293116/72251).
  The ~13:19 slot fired a 1-row
  wedge slot (bounded watched,
  released); rmax back to 1
  throughout (the [13:13:49] rmax=0
  stays isolated-benign). fb
  small-deltas hold (+5/+3, top
  edge); apc creep +3/+32 — LARGE
  +32 jump at release (+1113),
  post-release cooling did NOT
  persist; bytes dipped BELOW band
  then re-heated to ~+60K (the
  pre-7-row-stick mix — wedge-slot
  intensification correlate,
  aggregate drift only). No WALL
  LOST / wall=000 anywhere; no
  probe failures. Epoch I: 142 of
  194 clean (52 wedge rows).
  ROWLOG_N=525. Cycle 387
  dispatching (fresh lane).
- 2026-09-30 13:35 — cycle 387
  (fix-240) reconciled: CYCLE_OK —
  ~13:29 slot fired CLEAN (no
  wedge): streak extended to 2
  ([13:24:15]+[13:29:21]); NEW
  clean row [13:29:21] (h=200 p=9
  wall=200 fb=452 kv=2124 apc=1131,
  ev 297048/73276). Threshold NOT
  touched. fb FLAT: 452→452 — the
  first fb-hold row since the
  c371–c374 window (deceleration
  phase); apc creep continued
  1113→1131 (+18, not settled);
  bytes +65654 = NEW epoch-HIGH
  per-row bytes delta, breaking
  above the 57–60K pre-stick mix
  WITHOUT a wedge (untagged heat at
  epoch-high — the intensify
  correlate steepens). No WALL
  LOST / wall=000 anywhere;
  PROBE_EXIT=0, canary alive,
  HEALTH=200. Epoch I: 143 of 195
  clean (52 wedge rows). ROWLOG_N=
  526. Cycle 388 dispatching (fresh
  lane).
- 2026-09-30 13:45 — cycle 388
  (fix-241) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — NEW 2-row stick
  OPEN: [13:34:41] (h=000 fb=458
  apc=1148, ev 300368/74028) +
  [13:40:02] (h=000 fb=468 kv=2188
  apc=1152, ev 302520/74681) — the
  ~13:34 slot broke the 2-row
  streak ([13:24:15]+[13:29:21]);
  no release row yet; run 3 =
  worst-lineage watch, run ≥4 =
  alarm touch (5th). WEDGE-CORRELATE
  CONFIRMED: the untagged epoch-high
  heat row [13:29:21] (bytes +65654
  ) PRECEDED both tagged rows —
  heat preceded wedge fire (same
  preconditioning correlate as the
  7-row stick). fb FLAT BROKE:
  452→458→468 (+6/+10, run-5-stick-
  scale steps); apc 1131→1148→1152
  (creep decelerating); bytes back
  into the 57–60K mix (no new
  epoch-high). No WALL LOST /
  wall=000 anywhere (h=200 all
  rows); ladder bounded (cap 0).
  Epoch I: 143 of 197 clean (54
  wedge rows). ROWLOG_N=528. Cycle
  389 dispatching (fresh lane).
- 2026-09-30 13:52 — cycle 389
  (fix-242) reconciled: CYCLE_OK —
  ~13:45 slot RELEASED the 2-row
  stick at run 2 (yardstick NOT
  breached: no run-3 lineage, no
  5th alarm touch): NEW row
  [13:45:04] (h=200 p=9 wall=200
  fb=474 kv=2219 apc=1159, ev
  304764/75118). Post-release
  streak = 1 (the release row).
  fb back to smalls: +6 (468→474);
  apc creep +7 (1152→1159); bytes
  +76493 on the RELEASE row itself
  = NEW epoch-high (prior high
  +65654 at [13:29:21]) — heat
  ON an untagged clean row with no
  wedge fire behind it; the
  heat-precedes-wedge correlate did
  NOT repeat, but this record-heat
  row is now the freshest correlate
  reference for the next slot. kv
  →2219, ev→304764/75118. No WALL
  LOST / wall=000 anywhere; no
  probe failures. Epoch I: 144 of
  198 clean (54 wedge rows).
  ROWLOG_N=529. Cycle 390
  dispatching (fresh lane).
- 2026-09-30 13:58 — cycle 390
  (fix-243) reconciled: CYCLE_OK —
  streak extended to 3 ([13:45:04] +
  [13:50:05] + [13:55:06]): NEW
  clean rows [13:50:05] (fb=474 F
  kv=2221 apc=1172, ev 306478/
  75542, bytes +41781) + [13:55:06]
  (fb=474 F kv=2224 apc=1174, ev
  308253/75984, bytes +21653 =
  lowest per-row bytes in the
  visible window — HARD cool-down
  below the 48–52K band, on clean
  rows). The epoch-high-heat
  release row [13:45:04] (bytes
  +76493) did NOT trigger a wedge:
  correlate did not repeat (heat →
  2 clean slots). fb FLAT at 474
  ×2 — second consecutive hold,
  deceleration firming; apc creep
  decelerating (+13→+2); rmax=0
  transient on [13:55:06] =
  second isolated occurrence
  (benign). No WALL LOST /
  wall=000 anywhere; no probe
  failures. Epoch I: 146 of 200
  clean (54 wedge rows). ROWLOG_N=
  531. Cycle 391 dispatching (fresh
  lane).
- 2026-09-30 14:05 — cycle 391
  (fix-244) reconciled: CYCLE_OK —
  ~14:00 slot fired CLEAN: streak
  extended to 4 ([13:45:04] +
  [13:50:05] + [13:55:06] +
  [14:00:06]); NEW clean row
  [14:00:06] (h=200 p=9 wall=200
  fb=474 F kv=2226 apc=1177, ev
  310289/76481, bytes +25389).
  Threshold NOT touched. fb FLAT
  extends to ×4 at 474 (drift fully
  stalled — longest flat-hold since
  the c371–c374 window); apc creep
  mild restart +3 (1174→1177);
  bytes 3rd consecutive below-band
  row (cool-down deepening toward
  the +15–25K norm; floor record
  still +21653 at [13:55:06]);
  rmax=0 now on TWO consecutive
  rows ([13:55:06]+[14:00:06]) —
  no longer an isolate, still
  benign (no tag); kv→2226;
  ev→310289/76481. No WALL LOST /
  wall=000 anywhere; no probe
  failures. Epoch I: 147 of 201
  clean (54 wedge rows). ROWLOG_N=
  532. Cycle 392 dispatching (fresh
  lane).
- 2026-09-30 14:11 — cycle 392
  (fix-245) reconciled: CYCLE_OK —
  ~14:05 slot fired CLEAN: streak
  extended to 5 ([13:45:04]…
  [14:05:07]); NEW clean row
  [14:05:07] (h=200 p=9 wall=200
  fb=474 F kv=2228 apc=1181, ev
  310897/76708, bytes +18362 = NEW
  floor record, below-band cool-
  down now INSIDE the +15–25K
  norm). Threshold NOT touched.
  fb FLAT extends to ×5 at 474
  (drift fully stalled, longest
  hold); apc creep continues mild
  +4 (1177→1181); rmax=0 extends
  to a THIRD consecutive row
  ([13:55:06]+[14:00:06]+[14:05:07])
  — no longer an isolate, still
  benign (no tag); no tagged row
  appeared while rmax=0 persists.
  kv→2228; ev→310897/76708. No
  WALL LOST / wall=000 anywhere;
  no probe failures. Epoch I: 148
  of 202 clean (54 wedge rows).
  ROWLOG_N=533. Cycle 393
  dispatching (fresh lane).
- 2026-09-30 14:17 — cycle 393
  (fix-246) reconciled: CYCLE_OK —
  ~14:10 slot fired CLEAN: streak
  extended to 6 ([13:45:04]…
  [14:10:08]) = the LONGEST streak
  since the 08:05 arrival cluster
  (beats the post-arrival 5-row
  best; epoch 73-row record still
  stands). NEW clean row
  [14:10:08] (h=200 p=9 wall=200
  fb=474 F kv=2229 apc=1186, ev
  311509/76833, bytes +19557).
  Threshold NOT touched. fb FLAT
  extends to ×6 at 474 (drift
  fully stalled); apc creep +5
  (1181→1186); bytes 2nd in-norm
  row (+19557, floor record 18362
  stands); rmax=0 extends to a
  FOURTH consecutive row — still
  benign/uncombined (no tagged row
  under any rmax=0 row). No WALL
  LOST / wall=000 anywhere; no
  probe failures. Epoch I: 149 of
  203 clean (54 wedge rows).
  ROWLOG_N=534. Cycle 394
  dispatching (fresh lane).
- 2026-09-30 14:27 — cycle 394
  (fix-247) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — NEW 2-row stick
  OPEN: [14:15:28] (h=000 fb=474 F
  kv=2232 apc=1187, ev 313533/
  77323, bytes +22062 in-norm) +
  [14:20:49] (h=000 fb=478 kv=2256
  apc=1190, ev 317077/78195, bytes
  +56055 re-heat) — the ~14:15 slot
  broke the 6-row streak (the
  post-arrival record, 7 not
  reached); no release row yet.
  fb: FLAT held ×7 THROUGH the
  wedge row, then resumed +4; apc
  +1 then +3 (pace shifted from
  the +5 steady); rmax=0 run ENDED
  at exactly 4 (no 5th row, and no
  rmax=0-under-tag combo ever
  formed — both new tagged rows
  carry rmax=1); kv wedge-row spike
  +24. No WALL LOST / wall=000
  anywhere; no probe failures.
  Epoch I: 149 of 205 clean (56
  wedge rows). ROWLOG_N=536. Cycle
  395 dispatching (fresh lane).
- 2026-09-30 14:37 — cycle 395
  (fix-248) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — ALARM TOUCHED,
  5th: the tail stick extended to
  run 4 ([14:15:28]→[14:31:29]):
  [14:26:09] (h=000 fb=484 apc=
  1192, ev 320217/79036, bytes
  ~57K) = run 3 (worst-lineage
  watch entered) + [14:31:29]
  (h=000 fb=490 kv=2312 apc=1196,
  ev 322917/79669) = RUN 4, NO
  5ROW TOUCH REACHEDS STILL (alarm
  touched at run 4, no release).
  Record lineage: 12-row epoch
  record still stands, 7-row
  released post-record worst, 5-row
  bounded worst. Eighth DIRECT
  deep-dive (14:33, through run-4
  depth): ESCAPE_COUNT 77→90 (+13,
  ALL UNIFORM ≤2-fallback
  SERVERKILL escapes at walls 10.4
  s only — six closes: 13:37:15/
  13:43:48/14:15:40/14:19:28/
  14:21:16/14:26:55);
  FALLBACK_COUNT 434→493 (+59);
  KVL→2322 (+263, largest yet);
  RELEASE_COUNT=0 — cap STILL never
  reached through run-4 depth; no
  fallbacks=3; BOOTCOUNT=1; hits
  healthy with PERFECT 1:1 g0FAE
  trace convergence INSIDE the
  wedge window (6:6, 108:108, 111:
  111, 131:131) — no degradation
  signal. kv increments elevated
  (+29/+27 after +24 wedge-row
  spike); rmax=1 all 4 stick rows
  (rmax=0-under-tag combo NEVER
  formed); bytes sustained ~57K
  re-heat; fb +6/+6 moderate steady.
  No WALL LOST / wall=000 anywhere;
  no probe failures. Epoch I: 149
  of 207 clean (58 wedge rows).
  ROWLOG_N=538. Cycle 396
  dispatching (fresh lane).
- 2026-09-30 14:46 — cycle 396
  (fix-249) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — the alarm-touch
  stick EXTENDED to run 5: NEW
  tagged row [14:36:50] (h=000
  fb=499 kv=2346 apc=1199, ev
  325145/80140, bytes +54583)
  after [14:15:28]→[14:31:29] run
  4; NO release row yet. Depth now
  BEYOND the old 5-row bounded-
  worst; still below the released
  7-row post-record worst and the
  12-row epoch record. Ninth
  DIRECT deep-dive (14:43, through
  run-5 depth + in-stick):
  ESCAPE_COUNT 90→92 (+2, BOTH
  UNIFORM ≤2-fallback escapes at
  walls EXACTLY 10.4 s incl an
  in-stick close 14:42:46);
  FALLBACK_COUNT 493→505 (+12);
  KVL→2370 (+48); RELEASE_COUNT=0
  — cap STILL never reached
  through run-5 depth; no
  fallbacks=3; BOOTCOUNT=1; hits
  healthy with near-perfect ≤3-
  token differentiation (162:164,
  184:186, 191:193 — no
  degradation). Trend: fb +9 (mild
  accel past the +6/row pace);
  apc 1199 (+3); bytes ~53K
  (slight cooling); kv 2346
  (largest single-step +34); rmax=
  1 all 5 stick rows (no rmax=0-
  under-tag combo ever). No WALL
  LOST / wall=000 anywhere; no
  probe failures. Epoch I: 149 of
  208 clean (59 wedge rows).
  ROWLOG_N=539. Cycle 397
  dispatching (fresh lane).
- 2026-09-30 14:54 — cycle 397
  (fix-250) reconciled: CYCLE_OK —
  threshold-touched-and-released
  (deeper-first-release): NEW
  tagged row [14:42:10] (h=000
  fb=503 apc=1205, ev 327193/
  80771) extended the tail run to
  6 (PAST the 5-row worst, toward
  the 7-row released post-record
  worst; 6th touch of this stick's
  alarm state... no — touch count
  lineage = 5 events), THEN
  RELEASED at [14:47:11] (h=200
  p=9 wall=200 fb=509 kv=2386
  apc=1205, ev 330116/81435, bytes
  +44787, rmax=0 on release row
  only). Released run-length 6 =
  below the 7-row released
  post-record worst; epoch record
  still 12. fb pace settled: +4/
  +6 (the +9 accel declined, back
  on the +4–6/row pace); apc 1205
  flat at release; kv +16/+24;
  rmax=0-under-tag NEVER formed
  (all 6 stick rows rmax=1). No
  WALL LOST / wall=000 anywhere;
  no probe failures. Epoch I:
  150 of 210 clean (60 wedge
  rows). ROWLOG_N=541. Cycle 398
  dispatching (fresh lane).
- 2026-09-30 14:59 — cycle 398
  (fix-251) reconciled: CYCLE_OK —
  ~14:52 slot fired CLEAN: streak
  extended to 2 ([14:47:11] +
  [14:52:12]); NEW clean row
  [14:52:12] (h=200 p=9 wall=200
  fb=509 F kv=2390 apc=1209, ev
  330172/81435, bytes +20646).
  Threshold NOT touched. apc creep
  mild +4 (1205→1209); bytes DEEP
  COOL: +20646 (below the ~45K
  band into the low-20K zone —
  wedge outflow fully subsided,
  near the old +15–25K norm);
  ev-slot deltas FLAT; rmax=0 ×2
  consecutive post-release rows
  (benign; rmax=0-under-tag
  NEVER formed). No WALL LOST /
  wall=000 anywhere; no probe
  failures. Epoch I: 151 of 211
  clean (60 wedge rows).
  ROWLOG_N=542. Cycle 399
  dispatching (fresh lane).
- 2026-09-30 15:08 — cycle 399
  (fix-252) reconciled: CYCLE_OK —
  streak extended to 4
  ([14:47:11]…[15:02:32]): NEW
  clean rows [14:57:12] (fb=509 F
  apc=1212, ev 330932/81651, bytes
  +15283 = NEW floor record in the
  +15–25K norm) + [15:02:32]
  (fb=509 F apc=1214, ev 332556/
  82006, bytes +20531). Threshold
  NOT touched. fb FLAT extends to
  ×4 at 509 (longest hold since
  the c371–c374 window); apc
  creep gentle +2/+2 → 1214;
  ev slots grew slightly;
  rmax RETURNED to 1 ×2 — the
  rmax=0 post-release stretch
  closed at EXACTLY 2 rows, never
  under tag; kv→2393. No WALL
  LOST / wall=000 anywhere; no
  probe failures. Epoch I: 153 of
  213 clean (60 wedge rows).
  ROWLOG_N=544. Cycle 400
  dispatching (fresh lane).
- 2026-09-30 15:15 — cycle 400
  (fix-253) reconciled: CYCLE_OK —
  ~15:07 slot fired CLEAN: streak
  extended to 5 ([14:47:11]…
  [15:07:32]); NEW clean row
  [15:07:32] (h=200 p=9 wall=200
  fb=514 kv=2419 apc=1220, ev
  334256/82480, bytes +50727).
  Threshold NOT touched. fb drift
  RESUMED at +5 (the ×4 flat hold
  ended); apc creep pace JUMPED
  +6 (1214→1220, largest step
  since the wedge in-slot +6s);
  kv +26 (sharper than +3/+4
  prior); bytes RE-HEATED +50727
  (out of the norm back into the
  elevated ~50K zone, just below
  the 54–58K wedge-signature
  range — the heat-precedes-wedge
  correlate is armed again; the
  last two such heat rows preceded
  stick fires within 1–2 slots).
  rmax=0 3rd isolate (on this
  clean row; the rmax=1 stretch
  closed at exactly 2 again;
  rmax=0-under-tag NEVER formed).
  No WALL LOST / wall=000
  anywhere; no probe failures.
  Epoch I: 154 of 214 clean (60
  wedge rows). ROWLOG_N=545.
  Cycle 401 dispatching (fresh
  lane).
- 2026-09-30 15:23 — cycle 401
  (fix-254) reconciled: CYCLE_OK —
  wedge fired in-window and
  RELEASED in-window (like the
  13:19→13:24 slot): NEW rows
  [15:12:53] (h=000 p=9 fb=515
  apc=1228, ev 335448/82876, bytes
  +30993, rmax=1 = the only
  on-tag pattern) ALERT:
  SERVE-WEDGE + [15:17:54] (hb=200
  untagged: fb=518 kv=2443
  apc=1237, ev 338156/83406, bytes
  +44068). Run only 1 (NOT
  alarm-touch); the 5-row clean
  streak ([14:47:11]→[15:07:32])
  closed at EXACTLY 5 (not the
  record-tying 6); post-release
  streak restarts at 1. HEAT
  correlate NOW FIRMLY NAMED:
  [15:07:32] heat row (~50K
  re-heat, fb jump +5, kv +26)
  → wedge fired exactly 1 slot
  later — 3rd confirmed instance
  (the [13:45:04]-lineage gap was
  wider — heat rows AWAIT the
  next stick, at short or long
  lag). On-tag pattern PERFECT:
  the wedge row carries EXACTLY
  ONE rmax=1; rmax=0-under-tag
  NEVER formed. fb DECELERATED
  (+1 wedge, +3 release → 518);
  apc ACCELERATED further (+8/+9
  → 1237); kv warming again +7/+
  17 → 2443; ev grew; bytes
  cooled to 30993 at the wedge
  row then re-heated to +44068
  on release (elevated zone,
  above norm, below the 54–58K
  signature). rmax=0 again on the
  newest clean row. No WALL LOST /
  wall=000 anywhere; no probe
  failures. Epoch I: 155 of 216
  clean (61 wedge rows).
  ROWLOG_N=547. Cycle 402
  dispatching (fresh lane).
- 2026-09-30 15:32 — cycle 402
  (fix-255) reconciled: CYCLE_OK —
  ~15:22 slot fired CLEAN: clean
  streak now 2 ([15:17:54] +
  [15:22:54]); NEW clean row
  [15:22:54] (h=200 p=9 wall=200
  fb=519 kv=2453 apc=1241, ev
  340068/83923, bytes +43657).
  Threshold NOT touched. fb
  deceleration DEEPENED beyond the
  +3–5 alert-window pace at +1;
  apc acceleration STALLED (the
  +8/+9 chain settled to +4 →
  1241); kv warming continues +10;
  ev grew slightly. bytes still
  ELEVATED +43657 with slight
  COOLING (−411) — did NOT
  re-heat into the ~50K+ zone;
  correlate not triggered. rmax=0
  again (on the newest clean
  row); on-tag pattern PERFECT;
  rmax=0-under-tag NEVER formed.
  No WALL LOST / wall=000
  anywhere; no probe failures.
  Epoch I: 156 of 217 clean (61
  wedge rows). ROWLOG_N=548.
  Cycle 403 dispatching (fresh
  lane).
- 2026-09-30 15:39 — cycle 403
  (fix-256) reconciled: CYCLE_OK —
  streak extended to 4
  ([15:17:54]…[15:33:01]): NEW
  clean rows [15:28:00] (fb=519 F
  apc=1255 +14 SPIKE, ev 340960/
  84091, bytes +49073 near-50K
  flare, rmax=1 untagged-normal) +
  [15:33:01] (fb=519 F apc=1260
  +5, ev 341676/84380, bytes
  +25211 cooled into norm, rmax=0).
  NO alarm touch (fixer prose
  over-read: conflated the clean
  streak with the ALERT
  run-length — the file rows are
  authoritative: ZERO tagged rows
  in-window; the yardstick counts
  consecutive hb=000 rows only).
  NOTE: the [15:28:00] near-50K
  flare (+49073, below the 50727
  checkpoint) COMPLETED without a
  wedge firing — a non-firing
  flare; the heat correlate stays
  probabilistic (lag varies,
  flares can hold). fb ×3 flat
  hold at 519; apc +14 spike then
  decel +5 (net +19 over 2
  slots); kv warming gentle +4/+3
  → 2460; ev grew. On-tag pattern
  PERFECT; rmax=0-under-tag NEVER
  formed. No WALL LOST / wall=000
  anywhere; no probe failures.
  Epoch I: 158 of 219 clean (61
  wedge rows). ROWLOG_N=550.
  Cycle 404 dispatching (fresh
  lane).
- 2026-09-30 15:45 — cycle 404
  (fix-257) reconciled: CYCLE_OK —
  ~15:38 slot fired CLEAN: streak
  extended to 5 ([15:17:54]…
  [15:38:01]); NEW clean row
  [15:38:01] (h=200 p=9 wall=200
  fb=519 F kv=2465 apc=1267, ev
  342932/84622, bytes +38442).
  Threshold NOT touched (touch
  lineage stays 5). ONE more
  clean row ties the 6-row
  post-arrival zone record
  ([13:45:04]→[14:10:08]). fb FLAT
  extends ×4 at 519; apc settle
  HOLDS (bracket +14/+5/+7 — no
  second consecutive large step);
  bytes mid-band re-tick +38442 —
  NO new flare (probabilistic
  correlate window not opened);
  kv warming +5; ev mild growth.
  Render variance note: this dump
  printed [15:02:32] fb=514 where
  prior probes rendered 509
  (past-row re-rendering inside
  the live rowlog emit window;
  the new-row rule is unaffected
  — newest-row values consistent).
  On-tag pattern PERFECT;
  rmax=0-under-tag NEVER formed;
  no WALL LOST / wall=000; no
  probe failures. Epoch I: 159 of
  220 clean (61 wedge rows).
  ROWLOG_N=551. Cycle 405
  dispatching (fresh lane).
- 2026-09-30 15:52 — cycle 405
  (fix-258) reconciled: CYCLE_OK —
  ~15:43 slot fired CLEAN: streak
  extended to 6, TIES the 6-row
  post-arrival zone record
  ([13:45:04]→[14:10:08]); NEW
  clean row [15:43:09] (h=200 p=9
  wall=200 fb=519 F kv=2477
  apc=1273, ev 345004/85135, bytes
  +42894, rmax=1 untagged-normal).
  Threshold NOT touched (touch
  lineage stays 5). ONE more clean
  row sets a NEW 7-row zone
  record. fb FLAT extends ×5 at
  519; apc settle CONTINUES
  (bracket +14/+5/+7/+6 — no
  first back-to-back ≥+10 step);
  bytes mid-band hold +42894 — NO
  new flare (correlate window
  stayed closed); kv warming
  gentle +12; ev mild growth.
  Historical re-read: past rows
  identical this window; verdict
  anchored on the newest row.
  On-tag pattern PERFECT;
  rmax=0-under-tag NEVER formed;
  no WALL LOST / wall=000; no
  probe failures. Epoch I: 160 of
  221 clean (61 wedge rows).
  ROWLOG_N=552. Cycle 406
  dispatching (fresh lane).
- 2026-09-30 15:58 — cycle 406
  (fix-259) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — streak record
  FIRST: [15:48:09] clean (fb=519 F
  apc=1278 +5, ev 346690/85480,
  bytes +28543 cool, kv=2478,
  rmax=1 untagged-normal) → clean
  streak 7 SET the NEW post-arrival
  zone record ([15:17:54]→
  [15:48:09]). THEN the ~15:53
  slot WEDGE-FIRED: [15:53:30]
  (h=000 p=9 fb=522 kv=2488
  apc=1286, ev 349562/86155, bytes
  +36350, rmax=1 = the only on-tag
  pattern) ALERT: SERVE-WEDGE —
  NEW live stick at ALERT run 1
  (watched; NOT a touch: touch
  lineage STAYS 5 — fixer prose
  overread to 6; only fires
  reaching run ≥4 count). NO heat
  flare preceded (mid-band fields;
  fires WITHOUT flares are
  long-established — the correlate
  stays probabilistic: fires can
  come 1–5 slots after a flare OR
  with no flare at all). fb
  re-accelerated +3 on the wedge
  row (519→522); apc settle
  continues (bracket +14/+5/+7/+6/
  +5/+8 — no back-to-back large
  step); kv warming +12 → 2488; ev
  grew; bytes mid-band (no flare).
  On-tag pattern PERFECT;
  rmax=0-under-tag NEVER formed;
  no WALL LOST / wall=000; no
  probe failures. Epoch I: 161 of
  223 clean (62 wedge rows).
  ROWLOG_N=554. Cycle 407
  dispatching (fresh lane).
- 2026-09-30 16:09 — cycle 407
  (fix-260) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — new rows:
  [15:58:33] (h=200 p=9 fb=530
  kv=2515 apc=1289, ev 351986/
  86762, bytes +57038 — re-heated
  INTO the 54–58K signature band)
  RELEASED the run-1 stick
  in-window; [16:03:53] (h=000 p=9
  fb=538 kv=2546 apc=1320, ev
  354086/87213, bytes +63400 = NEW
  epoch-high ABOVE the 54–58K
  band, rmax=1 = the only on-tag
  pattern) ALERT: SERVE-WEDGE —
  brand-new live stick at run 1
  after just ONE clean row
  (fastest re-fire since the
  cluster opened; the 7-row
  record streak closed at
  [15:48:09] in c406). NOT a
  touch (lineage stays 5). Pace
  shifts: fb stepped UP to +8/+8
  (522→530→538); apc +31 single
  step (first ≥+10; no
  back-to-back pair yet); kv
  accelerating +27/+31; bytes
  epoch-high. On-tag PERFECT;
  rmax=0-under-tag NEVER formed;
  no WALL LOST / wall=000; no
  probe failures. Epoch I: 163 of
  226 clean (63 wedge rows).
  ROWLOG_N=556. Cycle 408
  dispatching (fresh lane).
- 2026-09-30 16:19 — cycle 408
  (fix-261) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — live stick
  EXTENDED to run 2: NEW tagged row
  [16:09:14] (h=000 p=9 fb=548
  kv=2586 apc=1321, ev 355434/
  87606, bytes +59472 above-band
  hold, rmax=1 = the only on-tag
  pattern) ALERT: SERVE-WEDGE
  ([16:03:53] run 1 → run 2,
  consecutive). NOT a touch
  (lineage stays 5); run 3 next =
  worst-lineage watch; run 4 =
  ALARM TOUCH (6th). Pace shift
  deepens: fb +8/+8/+10 (522→530→
  538→548); kv ACCELERATING +27/
  +31/+40 (2546→2586); apc settle
  resumed +1 after the +31
  break-out; bytes cooled OFF the
  +63400 epoch-high but HOLDS
  above the 54–58K band; ev grew.
  On-tag PERFECT (3 tagged rows in
  window, each exactly one
  rmax=1); rmax=0-under-tag NEVER
  formed; no WALL LOST / wall=000;
  no probe failures. Epoch I: 163
  of 227 clean (64 wedge rows).
  ROWLOG_N=557. Cycle 409
  dispatching (fresh lane).
- 2026-09-30 16:31 — cycle 409
  (fix-262) reconciled: CYCLE_OK —
  DOUBLE touch-and-release: the
  run-2 stick ([16:03:53]→
  [16:09:14]) released at
  [16:14:30] (h=200 fb=561 kv=2628
  apc=1321 flat, ev 357786/88148,
  bytes +53076 cooled below band),
  then a fresh 1-row stick touched
  at [16:19:50] (h=000 fb=573,
  rmax=1 untagged-fire) and
  released again at [16:24:54]
  (h=200 fb=586, ev 359714/88606,
  bytes +48447). NO alarm touch
  (max run 2; lineage stays 5);
  post-release streak 1. FB
  STEP-UP REALIZED: +13/+12/+13 =
  first three consecutive ≥+12
  steps (548→561→573→586; the
  +8/+8/+10 did NOT hold); kv
  ACCELERATING +42/+41/+35
  (2628→2669→2704); apc FLAT ×3
  at 1321 after the +31 break-out
  (no back-to-back ≥+10 pair);
  bytes cooled BELOW the 54–58K
  band on all three rows
  (epoch-high 63400 stands); ev
  grew. On-tag pattern PERFECT;
  rmax=0-under-tag NEVER formed;
  minor past-row re-render
  variation in ev values (rows
  only, verdict anchored on the
  newest-row rule). No WALL LOST /
  wall=000; no probe failures.
  Epoch I: 165 of 230 clean (65
  wedge rows). ROWLOG_N=560.
  Cycle 410 dispatching (fresh
  lane).
- 2026-09-30 16:44 — cycle 410
  (fix-263) reconciled: CYCLE_OK —
  rapid-fire #4 touch-and-release:
  fresh 1-row stick fired [16:30:14]
  (h=000 p=9 fb=595 +9 FIRST
  sub-+12 brake of the regime,
  kv=2734 +30 first brake below
  +35, apc=1321 flat ×4, ev
  361386/88992, bytes +52119
  mid-zone, rmax=1 only on-tag)
  ALERT: SERVE-WEDGE — released one
  row later at [16:35:25] (h=200
  untagged: fb=607 +12 re-advance,
  kv=2777 +43 LARGEST family step,
  apc=1322 +1 nudge, ev 362126/
  89245, bytes +56804 = NEW ≥54K
  heat row → correlate RE-ARMED).
  Post-release streak reset then
  restarted at 1. No touch (max
  run 2 all-day; lineage stays 5).
  The rebound story: every row
  advances; fb pace braked once
  then re-armed; kv braked once
  then drew the largest family
  step. On-tag PERFECT;
  rmax=0-under-tag NEVER formed; no
  WALL LOST / wall=000; no probe
  failures. Epoch I: 166 of 232
  clean (66 wedge rows).
  ROWLOG_N=562. Cycle 411
  dispatching (fresh lane).
- 2026-09-30 16:55 — cycle 411
  (fix-264) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — the re-armed
  correlate HIT: fresh stick fired
  [16:40:45] (h=000 p=9 fb=615 +8,
  kv=2819 +42, apc=1324 +2 nudge,
  ev 362634/89329, bytes +62127
  ≥54K, rmax=1 only on-tag) ALERT:
  SERVE-WEDGE, EXACTLY 1 slot after
  the ≥54K heat row [16:35:25]
  (4th named flare→fire at lag 1);
  then the live stick EXTENDED to
  run 2 at [16:46:05] (h=000 p=9
  fb=622 +7, kv=2868 +49 = NEW
  largest family step, apc=1324
  FLAT, ev 364426/89778, bytes
  +63723 = NEW epoch-high ABOVE
  the +63400 record). NOT a touch
  (lineage stays 5); run 3 next =
  worst-lineage watch; run 4 =
  ALARM TOUCH (6th). kv ≥+30
  family EXTENDS: +42/+49 (no
  second brake). fb braked
  (+8/+7; every row still
  ADVANCES — no flat hold
  anywhere). apc re-formed
  flat-hold at 1324 after the
  +2 nudge. On-tag PERFECT;
  rmax=0-under-tag NEVER formed;
  no WALL LOST / wall=000; no
  probe failures. Epoch I: 167 of
  235 clean (68 wedge rows).
  ROWLOG_N=564. Cycle 412
  dispatching (fresh lane).
- 2026-09-30 17:09 — cycle 412
  (fix-265) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — the live stick
  EXTENDED run 2 → 5 = ALARM TOUCH
  (6th): NEW tagged rows [16:51:26]
  (h=000 p=9 fb=631 +9, kv=2907
  +39 extension, apc=1325 +1,
  ev 365850/90150, bytes +54901
  in-band), [16:56:46] (h=000 p=9
  fb=640 +9, kv=2935 +28 brake,
  apc=1326 +1, ev 368402/90827,
  bytes +47191 first sub-54K
  in-stick), [17:02:07] (h=000 p=9
  fb=648 +8, kv=2959 +24 brake,
  apc=1327 +1, ev 371334/91486,
  bytes +45999 sub-54K ×2). Touch
  lineage 5 → 6; depth 5 = ONE
  slot short of the deepest
  released lineage (6-row released
  [14:15:28]→[14:42:10]); run 6
  next would EXCEED that deepest
  released lineage. Trending
  details: fb brake continues
  cleanly +9/+9/+8 (every row
  advances, no flat hold, no
  re-arm); apc +1 drips ×3 (no
  ≥+10 pair); kv: family extended
  once (+39) then the BRAKE
  returned (+28/+24 ×2 sub-+30);
  bytes: DOME COOLING divergence —
  sub-54K ×2 while hb=000 sticks
  are live (novel context worth
  watching; the band hold ended).
  On-tag PERFECT (all window
  tagged rows exactly one
  rmax=1); rmax=0-under-tag NEVER
  formed; no WALL LOST / wall=000;
  no probe failures. Epoch I: 167
  of 238 clean (71 wedge rows).
  ROWLOG_N=567. Cycle 413
  dispatching (fresh lane).
- 2026-09-30 17:37 — cycle 413
  (fix-266) reconciled: CYCLE_ALERT:
  SERVE-WEDGE — live stick
  EXTENDED to run 10 (DOUBLE
  DIGITS): NEW tagged rows
  [17:07:27] (fb=656 +8, kv=2990
  +31 ext, ev 371874/91646, bytes
  +46794), [17:12:47] (fb=667 +11,
  kv=3021 +31 double-ext, ev
  373406/92031, bytes +45885),
  [17:18:08] (fb=677 +10, kv=3048
  +27 brake, ev 373490/92084,
  bytes +45384 stick-low),
  [17:23:28] (fb=686 +9, kv=3078
  +30 ext, ev 373990/92112, bytes
  +50219), [17:28:49] (fb=693 +7,
  kv=3100 +22 brake, ev 375962/
  92734, bytes +52775). Every named
  depth CROSSED this cycle: run 6
  EXCEEDS the deepest released
  lineage (6-row); run 8 beyond
  the 7-row released post-record
  worst; RECORD BREAK at run 13 =
  3 rows away (epoch 12-row
  stick). Longest-held stick of
  the epoch by a wide margin; NOT
  released. apc DRIPS DEAD: flat
  ×6 at 1327 (no ≥+10 movement
  since the +31 break-out). bytes:
  COLD-STICK DIVERGENCE ×7 sub-54K
  sustained (45384 stick-low →
  re-advance 52775, still
  sub-band); no re-heat ≥54K on
  any live row; the novel
  deep-cold-stick context now
  spans the ENTIRE 10-row stick.
  kv family: extend/extend/brake/
  extend/brake (2959→3100, +141
  net; not perishing). fb:
  clean-brake family intact
  (+8/+11/+10/+9/+7; every row
  advances; 629→693 over the
  stick; the +11/+10 pair is a
  modest mid-stick pace step).
  On-tag PERFECT 10/10 (each
  exactly one rmax=1);
  rmax=0-under-tag NEVER formed;
  no WALL LOST / wall=000; no
  probe failures. Epoch I: 167 of
  243 clean (76 wedge rows).
  ROWLOG_N=572. Cycle 414
  dispatching (fresh lane).
- 2026-09-30 17:48 — cycle 414
  (fix-267) reconciled: CYCLE_OK —
  the run-10 double-digit stick
  RELEASED at depth 10: first
  untagged [17:34:05] (h=200 p=9
  fb=697 +4, kv=3117 +17,
  apc=1327 FLAT ×7 (the release
  row), ev 377998/93261, bytes
  +53608 sub-54K) then [17:39:06]
  (fb=700 +3, kv=3137 +20,
  apc=1335 +8 = first movement
  after the ×7 freeze, ev 378694/
  93461, bytes +58631 = re-heated
  ABOVE the 54K band on a clean
  row — the cold-stick divergence
  resolved UPWARD, not down to
  the norm). NO RECORD BREAK
  (run 11 never formed — the
  epoch 12-row record stick
  [08:39:29]→[09:38:14] STANDS);
  closed depth 10 = deepest
  release of the lineage family
  (4 rows deeper than the 6-row
  deepest); post-release streak
  ×2. kv brake chain ×2 sub-+30
  (+17/+20 after the extension;
  third consecutive = perish
  watch). On-tag PERFECT through
  release (10/10); no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 169 of 245 clean (76
  wedge rows). ROWLOG_N=574.
  Cycle 415 dispatching (fresh
  lane).
- 2026-09-30 17:58 — cycle 415
  (fix-268) reconciled: CYCLE_OK —
  correlate fire lag-1 again:
  fresh 1-row stick fired
  [17:44:26] (h=000 p=9 fb=700
  FLAT on the wedge row, kv=3139
  +2, apc=1343 +8 second
  movement, ev 380534/93852,
  bytes +52398 cooled BELOW band
  on the fire row itself, rmax=1
  only on-tag) ALERT: SERVE-WEDGE,
  1 slot after the ≥54K re-heat
  row [17:39:06] — 6TH NAMED
  flare→fire AT LAG 1; released
  in-window at [17:49:31] (h=200
  untagged: fb=708 +8 step-up,
  kv=3167 +28, apc=1344 +1, ev
  383074/94498, bytes +50198
  below band ×2). Run only 1 (no
  touch; lineage stays 6);
  post-release streak 1. kv:
  FOURTH consecutive sub-+30
  (+17/+20/+2/+28) — PERISH WATCH
  confirmed-and-EXTENDED to ×4;
  no family re-extend since the
  stick-era +49/+39. apc: MOTION
  regime ×2 (+8/+1 minimal —
  watch a 3rd movement or
  re-freeze). fb: wedge row held
  FLAT (+0) then +8 step-up
  (largest post-stick delta
  after the +4/+3 brake). bytes:
  cool-back-BELOW-band ×2 — the
  post-release re-heat faded
  WITHOUT re-arming a fire
  (correlate fine-grained). On-tag
  PERFECT (rmax=0-under-tag
  NEVER); no WALL LOST / wall=000;
  no probe failures. Epoch I: 170
  of 247 clean (77 wedge rows).
  ROWLOG_N=576. Cycle 416
  dispatching (fresh lane).
- 2026-09-30 18:09 — cycle 416
  (fix-269) reconciled: CYCLE_OK —
  touch-and-release (7th rapid
  pulse): 1-row stick fired
  [17:54:52] (h=000 p=9 fb=713
  +5, kv=3192 +25 = FIFTH
  consecutive sub-+30, apc=1344
  FLAT ×3 (freeze resumes — no
  third movement), ev 384486/
  94763, bytes +55590 ≥54K heat
  row, rmax=1 only on-tag) ALERT:
  SERVE-WEDGE — NO ≥54K heat row
  preceded (lag-1 row [17:49:31]
  was +50198 below band; NOT the
  7th named correlate — a
  NO-FLARE fire); released
  in-window at [18:00:07] (h=200
  untagged: fb=723 +10, kv=3223
  +31 = family RE-ARMS one row
  after the melt, apc=1344 flat,
  ev 386026/95165, bytes +47629).
  Run 1 only (no touch; lineage
  stays 6); post-release streak 1.
  The ≥54K heat row [17:54:52]
  did NOT fire at lag 1 (cooled)
  — fire-lag watch armed and
  un-fired; fires continue both
  with and without flares
  (fine-grained correlate). kv
  family thrash: melt-deepened
  ×5 then re-arm +31 (watch
  extension/melt alternation
  windows). apc re-frozen ×3 at
  1344 (motion regime did NOT
  confirm). fb +5/+10 (neither
  re-arm ×2, nor brake chain, nor
  a flat hold formed — regime
  TBD). On-tag PERFECT
  (rmax=0-under-tag NEVER formed);
  no WALL LOST / wall=000; no
  probe failures. Epoch I: 171 of
  249 clean (78 wedge rows).
  ROWLOG_N=578. Cycle 417
  dispatching (fresh lane).
- 2026-09-30 18:22 — cycle 417
  (fix-270) reconciled: CYCLE_OK —
  touch-and-release (8th rapid
  pulse): heat row [18:05:23]
  (h=200 untagged, fb=723 FLAT,
  ev 387114/95441, bytes +59591
  ≥54K, apc=1353 +9 after the ×3
  freeze, kv=3230 +7) → the
  wedge FIRED at lag 1 ([18:10:44]:
  h=000, fb=728 +5, kv=3252 +22,
  apc=1353 flat, ev 388974/95909,
  bytes +55934) ALERT: SERVE-WEDGE
  — 1-row stick bounded on both
  sides, released in-window at
  [18:15:53] (h=200 untagged,
  fb=731 +3, kv=3263 +11,
  apc=1362 +9, ev 391201/96397,
  bytes +50296 cooled). Run 1 only
  (no touch; lineage stays 6);
  post-release streak 1.
  Fire-lag family: the lag-1 case
  CONFIRMED LIVE again here
  (alongside the lag-0 case at
  [17:54:52]); the fire-row heat
  row was NOT fired at lag 1
  (armed-and-un-fired again). apc:
  freeze broke → drip ×2 into a
  drip-chain (+9/+9 with a flat
  row between; a 3rd consecutive
  movement names the drip-chain;
  a ≥+10 move names the big-step
  pair watch). kv: melt-chain ×3
  (+7/+22/+11 — a fourth sub-+30
  melts ×4; a ≥+30 re-arms). fb:
  flat ×1 → +5/+3 (no regime
  named — leans brake-family).
  rmax=0 on the newest clean row
  (on-tag PERFECT; rmax=0-under-
  tag NEVER formed). No WALL LOST /
  wall=000; no probe failures.
  Epoch I: 173 of 252 clean (79
  wedge rows). ROWLOG_N=581.
  Cycle 418 dispatching (fresh
  lane).
- 2026-09-30 18:29 — cycle 418
  catch-up (fix-272) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  catch-up probe covered the
  overdue slot (the prior lane
  died on its status line; the
  ping wait was skipped by
  catch-up override): the ~18:16
  slot EMITTED NO_ROW (first
  file-emit gap witnessed in
  epoch I — rows otherwise appear
  every slot; flagged; watch
  whether gaps persist) and the
  first witnessed post-release
  row WEDGE-FIRED at lag-0 form:
  [18:21:13] (h=000 p=9 fb=737
  +6, kv=3286 +23 = FOURTH
  consecutive sub-+30 → MELT ×4
  NAMED, apc=1366 +4 = THIRD
  consecutive movement → DRIP-CHAIN
  NAMED ×3 (1353→1362→1366), ev
  394225/97134, bytes +51289
  sub-band, rmax=1 only on-tag)
  ALERT: SERVE-WEDGE. NO ≥54K heat
  row preceded (the
  [18:05:23]-armed lag-1 fire-lag
  watch was DISARMED by the
  [18:15:53] low-tide release;
  this fire is a sub-54K lag-0
  form — the fire-lag watch stays
  DISARMED until a new ≥54K row).
  Live run 1 (below ≥3
  worst-lineage / ≥4 ALARM TOUCH
  — no touch; lineage stays 6).
  fb: +3/+6 — no regime named
  (brake-family lean). On-tag
  PERFECT (rmax=0-under-tag
  NEVER; the lone rmax=0 rows
  remain untagged-clean-only). No
  WALL LOST / wall=000; no probe
  failures. Epoch I: 173 of 253
  clean (80 wedge rows).
  ROWLOG_N=582. Cycle 419
  dispatching (fresh lane).
- 2026-09-30 18:41 — cycle 419
  (fix-273) reconciled: CYCLE_OK —
  touch-and-release (9th rapid
  pulse): live stick EXTENDED run
  1→2 at [18:26:33] (h=000 p=9,
  fb=746 +9 = STEP-UP RE-ARM
  NAMED FIRST (≥+8 landed before
  any brake pair or flat hold),
  kv=3319 +33 ≥+30 = MELT FAMILY
  RE-ARMS (the melt ×5 record did
  NOT land — +33 broke the sub-+30
  streak at step 5), apc=1367 +1
  = drip-chain ×4, ev 395729/
  97520, bytes +54772 ≥54K heat
  row (fire-lag RE-ARMS; first
  ≥54K since the 18:15:53
  low-tide disarm), rmax=1 only
  on-tag) ALERT: SERVE-WEDGE —
  then RELEASED in-window at
  [18:31:34] (h=200 untagged:
  fb=752 +6 sub-8, kv=3337 +18
  (new sub-+30 count 1), apc=1368
  +1 = drip-chain ×5 OPEN, ev
  397244/97924, bytes +50454
  cooled (armed-and-un-fired
  again), rmax=0 on clean row).
  Closed run 2 (no touch; lineage
  stays 6); post-release streak 1.
  File-emit gap RESOLVED: rows
  resumed every slot (18:21:13 →
  18:26:33 → 18:31:34); no second
  gap. On-tag PERFECT
  (rmax=0-under-tag NEVER formed).
  No WALL LOST / wall=000; no
  probe failures. Epoch I: 174 of
  255 clean (81 wedge rows).
  ROWLOG_N=584. Cycle 420
  dispatching (fresh lane).
- 2026-09-30 18:51 — cycle 420
  (fix-274) reconciled: CYCLE_OK —
  touch-and-release (10th rapid
  pulse): 1-row stick fired
  [18:36:54] (h=000 p=9, fb=752
  Δ+0 = FLAT HOLD NAMED FIRST,
  kv=3344 +7 (2nd consecutive
  sub-+30 → MELT-CHAIN WATCH),
  apc=1374 +6 = drip-chain ×6
  (SIXTH consecutive movement),
  ev 399848/98573, bytes +48896
  sub-cool, rmax=1 only on-tag)
  ALERT: SERVE-WEDGE — NO fresh
  ≥54K heat row in lag-1 (the
  +54772 armed heat was two rows
  back with its lag-1 successor
  already cooled; this fire =
  lag-2/delayed form outside the
  named lag-0/lag-1 frames);
  released in-window at
  [18:42:14] (h=200 untagged:
  fb=754 +2 sub-brake/sub-step,
  kv=3361 +17 = melt-chain ×3,
  apc=1381 +7 = drip-chain ×7
  OPEN, ev 401184/98844,
  bytes +62549 = NEW ≥54K heat
  row (fire-lag re-armed;
  armed-and-un-fired — no
  successor yet), rmax=1 on clean
  row). Run 1 only (no touch;
  lineage stays 6); post-release
  streak 1. apc: chain ×7 open
  (1353→1362→1366→1367→1368→1374
  →1381); no ≥+10. kv: melt-chain
  ×3 landed post-re-arm (the
  melt-record clock restarted at
  the +33 re-arm). fb: flat named
  (+0) then +2. On-tag PERFECT; no
  WALL LOST / wall=000; no probe
  failures. Epoch I: 175 of 257
  clean (82 wedge rows).
  ROWLOG_N=586. Cycle 421
  dispatching (fresh lane).
- 2026-09-30 19:01 — cycle 421
  (fix-275) reconciled: CYCLE_OK —
  touch-and-release (11th rapid
  pulse): 1-row stick fired
  [18:47:34] (h=000 p=9, fb=759
  +5, kv=3390 +29 = melt-chain
  ×4 (one off the ≥+30 re-arm
  line), apc=1384 +3 =
  drip-chain ×8, ev 402532/
  99135, bytes +62089 ≥54K
  fire-row heat, rmax=1 only
  on-tag) ALERT: SERVE-WEDGE —
  fire form LANDED = LAG-1 (fired
  one slot after the [18:42:14]
  armed +62549 heat); released
  in-window at [18:52:35]
  (h=200 untagged: fb=765 +6,
  kv=3416 +26 = melt-chain ×5
  OPEN, apc=1386 +2 =
  drip-chain ×9 OPEN, ev
  403751/99405, bytes +56577 =
  THIRD consecutive ≥54K heat
  row (fire-lag watch ARMED AND
  UN-FIRED, no successor yet),
  rmax=1 on clean row).
  Fire-lag family: fires now
  include the lag-2 delayed form
  (at [18:36:54]) alongside
  lag-0 and lag-1; the +62089
  charge did NOT fire at lag-1
  (next slot clean — lag-2 slot
  pending). bytes heat-band:
  three straight ≥54K rows
  (+62549 → +62089 → +56577).
  apc: chain ×9 open (1353→…→
  1386); no ≥+10. kv: melt-chain
  ×5 open (+29/+26 both sub-+30;
  re-arm ×2 not landed). fb:
  sub-step drift (+2/+5/+6 —
  nothing named). Run 1 only (no
  touch; lineage stays 6);
  post-release streak 1.
  On-tag PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 176 of 259 clean (83
  wedge rows). ROWLOG_N=588.
  Cycle 422 dispatching (fresh
  lane).
- 2026-09-30 19:09 — cycle 422
  (fix-276) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  live stick open at run 1:
  [19:02:56] fired as the 12th
  rapid pulse — fire form =
  LAG-2 DELAYED (2nd lag-2
  precedent: the armed +56577
  heat fired 2 slots later;
  [18:57:35] was the clean
  lag-1 pass); the heat-band run
  CLOSED at exactly 3 (+62549 →
  +62089 → +56577; successors
  sub-band: +15170 cool,
  +45143). First a clean row
  [18:57:35] (h=200 untagged:
  fb=765 FLAT (+0) = FLAT HOLD
  #1 named first, kv=3417 +1,
  apc=1387 +1 = drip-chain ×10,
  ev 403779/99420, rmax=1 on
  clean row) extended the streak
  1→2; then the fire row
  [19:02:56] (h=000 p=9, fb=769
  +4 = single brake-candidate
  (brake NOT confirmed — needs a
  consecutive +3/+4), kv=3432
  +15 = melt-chain ×7, apc=1391
  +4 = DRIP-CHAIN ×11 OPEN, ev
  407079/100299 (flower crossed
  400K), bytes +45143 sub-band,
  rmax=1 only on-tag) ALERT:
  SERVE-WEDGE. Live ALERT run 1
  (below ≥3 worst-lineage / ≥4
  ALARM TOUCH — no touch;
  lineage stays 6); no release
  pending at tail. apc: chain ×11
  open; no ≥+10. kv: melt-chain
  ×7 open (+29/+26/+1/+15;
  re-arm ×2 not landed). fb: flat
  #1 (+0) then +4. On-tag
  PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 177 of 261 clean (84
  wedge rows). ROWLOG_N=590.
  Cycle 423 dispatching (fresh
  lane).
- 2026-09-30 19:18 — cycle 423
  (fix-277) reconciled: CYCLE_OK —
  touch-and-release (13th rapid
  pulse): live stick EXTENDED run
  1→2 at [19:08:16] (h=000 p=9,
  fb=774 +5, kv=3467 +35 ≥+30 =
  RE-ARM ×2 NAMED (the melt ×8
  chain did NOT form), apc=1394
  +3 = drip-chain ×12, ev
  408871/100775, bytes +58649
  ≥54K NEW heat row (fire-lag
  re-armed), rmax=1 only on-tag)
  ALERT: SERVE-WEDGE — then
  RELEASED in-window at
  [19:13:21] (h=200 untagged:
  fb=779 +5, kv=3491 +24 (new
  sub-+30 count 1 post re-arm ×2),
  apc=1398 +4 = DRIP-CHAIN ×13
  OPEN, ev 410897/101261 (flower
  pace ~+1.8–2.0K/slot steady),
  bytes +60288 ≥54K, rmax=1 on
  clean row). Closed run 2 (no
  touch; lineage stays 6);
  post-release streak 1.
  Fire-lag watch: the +58649
  heat's lag-1 successor
  [19:13:21] did NOT fire and
  did NOT cool — heat-on-heat:
  the successor was itself a
  fresh ≥54K row (+60288),
  extending the heat-run to 2
  consecutive ≥54K rows (a 3rd
  would match the old 3-row band
  +62549/62089/56577); the watch
  re-arms on the +60288 heat
  (armed-and-un-fired, no
  successor yet). fb: the +4
  brake candidate DIED without
  confirmation (successor +5 —
  the uncalled gap between
  brake +3/+4 and re-arm ≥+8;
  nothing named). On-tag
  PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 178 of 263 clean (85
  wedge rows). ROWLOG_N=592.
  Cycle 424 dispatching (fresh
  lane).
- 2026-09-30 19:28 — cycle 424
  (fix-278) reconciled: CYCLE_OK —
  clean streak EXTENDED to 2:
  [19:18:35] (h=200 p=9 untagged:
  fb=783 +4 — tail now +4/+5/+5/
  +4: EXPLICIT FLAG, the
  uncalled-mid-gap drift pattern
  is live and growing, no
  standing name; kv=3538 +47
  ≥+30 = RE-ARM ×3 NAMED (the
  melt-chain ×2 watch DIED with
  only one sub-+30 (+24)); apc=
  1399 +1 = DRIP-CHAIN ×14
  LANDED; ev 413193/101779
  (flower pace ~+1.8–2.0K/slot,
  modest uptick), bytes +69052
  ≥54K = 3RD consecutive ≥54K
  row → the 3-band match is
  COMPLETE on CLEAN rows
  (heat-on-heat non-resolution;
  no fire, no cool; lag-1/lag-2
  delayed fire remain ARMED AND
  UN-FIRED for the ~19:23 slot),
  rmax=1 on clean row). No wedge
  fired this cycle — no release
  row; touch lineage stays 6;
  live ALERT run 0. Fire forms
  now: lag-0/lag-1/lag-2
  delayed/heat-on-heat
  non-resolution; heat-run 3
  completed (a 4th straight ≥54K
  = NEW heat-run record).
  On-tag PERFECT (rmax=0-under-
  tag NEVER formed). No WALL OST /
  wall=000; no probe failures.
  Epoch I: 179 of 264 clean (85
  wedge rows). ROWLOG_N=593.
  Cycle 425 dispatching (fresh
  lane).
- 2026-09-30 19:38 — cycle 425
  (fix-279) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  live stick open at run 1:
  [19:28:59] fired as the 14th
  rapid pulse — fire form =
  LAG-2 DELAYED + HEAT-ON-HEAT
  (the +69052 armed heat fired 2
  slots later; the lag-1 slot
  [19:23:38] cleaned first; the
  firing row is itself a ≥54K
  heat row); the heat-run
  EXTENDED to a NEW 5-row RECORD
  (+58649 → +60288 → +69052 →
  +66418 → +67588; ≥54K rows
  both before the fire — two
  rows back and one row back).
  First a clean row [19:23:38]
  (h=200 untagged: fb=791 +8 =
  RE-ARM NAMED (≥+8 — closed the
  ×4 mid-gap-flag family),
  kv=3581 +43 = RE-ARM ×4,
  apc=1401 +2 = drip-chain ×15,
  ev 415177/102354, bytes +66418
  = 4TH consecutive ≥54K = NEW
  HEAT-RUN RECORD, rmax=1 on
  clean row) extending the
  streak 2→3; then the fire row
  [19:28:59] (h=000 p=9, fb=800
  +9 = 2ND consecutive ≥+8,
  kv=3635 +54 = re-arm ×5,
  apc=1402 +1 = DRIP-CHAIN ×16
  OPEN, ev 416697/102679 (flower
  pace eased ~+1.5K/slot),
  bytes +67588 = 5TH straight
  ≥54K, rmax=1 only on-tag)
  ALERT: SERVE-WEDGE. Live ALERT
  run 1 (below ≥3 worst-lineage /
  ≥4 ALARM TOUCH — no touch;
  lineage stays 6); no release
  pending at tail. apc: chain ×16
  open; no ≥+10. kv: re-arm ×4
  ×5 (+43/+54) — no sub-+30
  since the ×3 re-arm (melt
  clock keeps resetting). fb:
  re-arm ×2 (+8/+9 → 800).
  On-tag PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 180 of 266 clean (86
  wedge rows). ROWLOG_N=595.
  Cycle 426 dispatching (fresh
  lane).
- 2026-09-30 19:50 — cycle 426
  (fix-280) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  LIVE STICK OPEN AT RUN 4 =
  ALARM TOUCH (7th): the stick
  EXTENDED — [19:34:19] (h=000
  p=9, fb=815 +15 = 3RD
  consecutive ≥+8 → RE-ARM ×3
  NAMED (then +13/+12 = 5
  straight ≥+8 → 840), kv=3685
  +50 = re-arm ×6, apc=1402
  FLAT, ev 417345/102782,
  bytes +62797 ≥54K = 6TH
  straight → heat-run RECORD
  extended to 6, rmax=1) run 2
  → [19:39:39] (h=000 p=9,
  fb=828 +13, kv=3730 +45,
  apc=1402 flat ×2, ev
  417997/102979, bytes +61295
  ≥54K = 7th straight, rmax=1)
  run 3 = WORST-LINEAGE WATCH →
  [19:45:00] (h=000 p=9, fb=840
  +12, kv=3771 +41, apc=1402
  flat ×3 = DRIP-CHAIN CLOSED AT
  ×16 (17th movement did NOT
  form), ev 419737/103402,
  bytes +56732 ≥54K = 8TH
  STRAIGHT (heat-run RECORD now
  8: +58649→+60288→+69052→
  +66418→+67588→+62797→+61295→
  +56732; the newest row is
  itself an armed heat: a 9th
  straight = name it), rmax=1)
  run 4 = ALARM TOUCH (7th) —
  thresholds escalated loudly;
  NOT RELEASED (no untagged row
  since the stick opened at
  19:28:59). TOUCH LINEAGE
  6→7. apc: drip-chain CLOSED
  at ×16 (flat ×3). kv: re-arm
  ×6 named (+50/+45/+41; no
  sub-+30 since the ×3 re-arm —
  melt clock keeps resetting; a
  steady ≥+30 stream so far).
  fb: re-arm ×3 named (+15); 5
  straight ≥+8 (with the prior
  +8/+9 → 800). On-tag PERFECT;
  no WALL LOST / wall=000; no
  probe failures. Epoch I: 180
  of 269 clean (89 wedge
  rows). ROWLOG_N=598. Cycle
  427 dispatching (fresh lane).
- 2026-09-30 20:03 — cycle 427
  (fix-281) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  LIVE STICK OPEN AT RUN 6
  (still NO release since the
  19:28:59 open): [19:50:20]
  (h=000 p=9, fb=852 +12 (tail
  now 7 straight ≥+8: +8/+9/
  +15/+13/+12/+12/+12 → 864),
  kv=3812 +41 = RE-ARM ×7
  NAMED, apc=1402 flat ×4, ev
  421409/103807, bytes +57665
  ≥54K = 9TH straight → heat-run
  RECORD extended, rmax=1) run 5
  → [19:55:41] (h=000 p=9,
  fb=864 +12, kv=3851 +39,
  apc=1402 flat ×5 HOLD, ev
  421929/103915 (flower pace
  eased again), bytes +55692
  ≥54K = 10TH STRAIGHT (heat-run
  RECORD now 10: +58649→…→
  +55692; every fire in this
  stick = heat-on-heat form; no
  cooling row inside the run),
  rmax=1) run 6. NOT RELEASED —
  7 consecutive rows shy of run
  13 = RECORD BREAK (epoch
  12-row record); 14+ = DEEP
  RECORD. apc: flat ×5 hold
  (chain closed at ×16). kv:
  re-arm ×7 named (+41/+39 —
  seven straight ≥+30 deltas
  since the ×3 re-arm; no
  sub-+30 yet). fb: 7 straight
  ≥+8 (tail continuation). On-tag
  PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 180 of 271 clean (91
  wedge rows). ROWLOG_N=600.
  Cycle 428 dispatching (fresh
  lane).
- 2026-09-30 20:12 — cycle 428
  (fix-282) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  LIVE STICK OPEN AT RUN 8
  (still NO release since the
  19:28:59 open): [20:01:01]
  (h=000 p=9, fb=873 +9 = 8TH
  straight ≥+8 → tail ×8 NAMED,
  kv=3884 +33 = re-arm ×8,
  apc=1402 FLAT = flat ×6 HOLD
  NAMED (6th consecutive flat),
  ev 423749/104379, bytes
  +50727 SUB-54K = THE FIRST
  COOLING ROW INSIDE THE STICK —
  the 10-straight ≥54K heat-run
  CLOSED AT 10, rmax=1) run 7 →
  [20:06:22] (h=000 p=9, fb=883
  +10 = 9TH straight ≥+8 → tail
  ×9, kv=3922 +38 = re-arm ×9,
  apc=1405 +3 = THE FLAT BROKE —
  DRIP-CHAIN RE-ARMS AT ×17, ev
  426029/104981 (ev pace
  recovered +2280), bytes
  +55932 ≥54K = NEW HEAT-RUN
  STREAK 1 (heat-on-heat form),
  rmax=1) run 8. NOT RELEASED —
  5 further straight tagged rows
  to run 13 = RECORD BREAK
  (epoch 12-row record); 14+ =
  DEEP RECORD. First cooling row
  inside the stick: heat-on-heat
  form BROKE at [20:01:01]
  (+50727) — the 10-straight
  landmark stands; a new ≥54K
  band opens at [20:06:22]
  (+55932). apc: flat ×6 hold
  named, then broken by +3 →
  chain re-armed ×17. kv: re-arm
  ×8 → ×9 (+33/+38; no sub-+30
  in nine straight ≥+30 deltas
  since the ×3 re-arm). fb: tail
  ×8 → ×9 (+9/+10). On-tag
  PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 180 of 273 clean (93
  wedge rows). ROWLOG_N=602.
  Cycle 429 dispatching (fresh
  lane).
- 2026-09-30 20:25 — cycle 429
  (fix-283) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  LIVE STICK OPEN AT RUN 10
  (still NO release since the
  19:28:59 open): [20:11:42]
  (h=000 p=9, fb=892 +9 = 10TH
  straight ≥+8 → TAIL ×10
  NAMED, kv=3955 +33 = RE-ARM
  ×10 NAMED, apc=1408 +3 =
  DRIP-CHAIN ×18 NAMED (2nd
  consecutive movement), ev
  428145/105537 (flower +2116),
  bytes +52959 SUB-54K TAGGED =
  NO-HEAT FIRE — NEW VARIANT
  INSIDE THIS STICK (the
  new-heat-band streak from the
  armed +55932 closed at 1),
  rmax=1) run 9 → [20:17:02]
  (h=000 p=9, fb=901 +9 (11th
  straight ≥+8), kv=3982 +27 =
  FIRST SUB-+30 → MELT WATCH
  NAMED (breaks the ten-straight
  ≥+30 chain), apc=1408 Δ0 =
  FLAT HOLD (chain PAUSED), ev
  430057/105987 (flower +1912),
  bytes +47246 sub-54K TAGGED =
  2ND consecutive NO-HEAT FIRE
  row (2 straight sub-54K fires
  in-stick), rmax=1) run 10.
  NOT RELEASED — 3 further
  tagged rows to run 13 = RECORD
  BREAK (epoch 12-row record; 2
  more = run 12). No-heat fire
  variant ×2 consecutive (stick
  was all heat-on-heat until the
  [20:01:01] cooling row opened
  the door). apc: chain ×18 then
  flat pause. kv: re-arm ×10
  then melt watch. fb: tail ×10
  → 11th straight. On-tag
  PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 180 of 275 clean (95
  wedge rows). ROWLOG_N=604.
  Cycle 430 dispatching (fresh
  lane).
- 2026-09-30 20:40 — cycle 430
  (fix-284) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  7TH-TOUCH-AND-RELEASE LANDMARK:
  the run-10 stick RELEASED at
  the first untagged row
  [20:22:21] (h=200 untagged:
  fb=909 +8 = TAIL ×12 NAMED —
  12th straight ≥+8, on the
  release row; kv=4007 +25 =
  melt-chain (2nd sub-+30);
  apc=1408 FLAT ×2; ev
  432033/106521; bytes +45878)
  with the run CLOSED AT 10 — NO
  RECORD BREAK (the 12-row epoch
  record [08:39:29]→[09:38:14]
  stands; this release TIES the
  10-row deep-release lineage
  record). Then a clean row
  [20:27:25] (post-release streak
  2: fb=915 +6 mid-gap unnamed
  (tail broke), kv=4027 +20 = 3rd
  sub-+30 (melt-chain ×3 in
  view), apc=1408 flat ×3, ev
  432721/106661) before the wedge
  RE-OPENED at [20:32:45] (h=000
  p=9, fb=925 +10, kv=4059 +32 ≥
  +30 = RE-ARM ×11 NAMED, apc=
  1409 +1 = DRIP-CHAIN RE-ARMS
  ×19, ev 434049/106980, bytes
  +48041 sub-54K TAGGED = NO-HEAT
  FIRE again (1st sub-54K tagged
  row of the NEW stick; the prior
  ×2 cannot extend as ×3 — the
  untagged release rows break
  tagged-row consecutiveness),
  rmax=1) — NEW STICK OPEN AT RUN
  1. The ≥54K band stays CLOSED
  (last ≥54K tagged row
  [20:06:22]). apc: chain
  re-armed ×19. kv: melt-chain ×3
  then re-arm ×11. fb: tail ×12
  on the release row, broken by
  +6 mid-gap, then +10. On-tag
  PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 182 of 278 clean (96
  wedge rows). ROWLOG_N=607.
  Cycle 431 dispatching (fresh
  lane).
- 2026-09-30 20:53 — cycle 431
  (fix-285) reconciled:
  CYCLE_ALERT: SERVE-WEDGE —
  LIVE STICK OPEN AT RUN 4 =
  ALARM TOUCH (8th): the
  NO-HEAT stick EXTENDED —
  [20:38:06] (h=000 p=9, fb=934
  +9 = STEP-UP PAIR NAMED (2nd
  consecutive ≥+8 after the
  tail-×12 break), kv=4090 +31
  = RE-ARM ×12 NAMED, apc=1410
  +1 = DRIP-CHAIN ×20 NAMED
  (2nd consecutive movement —
  chain re-grows), ev
  434601/107115, bytes +51487
  sub-54K TAGGED = NO-HEAT FIRE
  ×2 NAMED, rmax=1) run 2 →
  [20:43:26] (h=000 p=9, fb=943
  +9 (3rd straight ≥+8),
  kv=4122 +32 (re-arm ×12
  continues), apc=1410 FLAT
  (chain PAUSED), ev
  435469/107269, bytes +48229
  sub = NO-HEAT ×3, rmax=1) run
  3 = WORST-LINEAGE WATCH →
  [20:48:46] (h=000 p=9, fb=947
  +4 (breaks the ≥+8 chain,
  sub-8 unnamed), kv=4134 +12 =
  FIRST SUB-+30 → MELT WATCH
  NAMED (after the ×12 re-arm),
  apc=1425 +15 = movement after
  pause (consecutive counter
  resets to 1; LARGEST APC STEP
  SEEN (+15 ≥+10); a 2nd
  consecutive ≥+10 movement
  names the big-step pair
  watch), ev 438465/108008
  (flower pace ELEVATED +2996 —
  1st ≥+3K slot; a 2nd names an
  ev pace jump), bytes +49102
  sub = NO-HEAT ×4, rmax=1) run
  4 = ALARM TOUCH (8th) LOUD.
  NOT RELEASED — no untagged row
  since the stick opened at
  20:32:45. NO-HEAT FIRE ×4
  straight sub-54K (the ≥54K
  band stays CLOSED — last ≥54K
  tagged row [20:06:22]; the
  10-straight heat-run record
  stands). apc: chain ×20 paused
  then +15 break-out step. kv:
  re-arm ×12 (+31/+32) then
  melt watch. fb: step-up pair
  (+9/+9) then +4. On-tag
  PERFECT; no WALL LOST /
  wall=000; no probe failures.
  Epoch I: 182 of 281 clean (99
  wedge rows). ROWLOG_N=610.
  Cycle 432 dispatching (fresh
  lane).
- 2026-09-30 21:06 — cycle 432
  (fix-286) reconciled: CYCLE_OK —
  touch-and-release (8th-touch-
  and-release): the NO-HEAT
  stick's 5TH tagged row
  [20:54:07] (h=000 p=9, fb=953
  +6 sub-8 unnamed, kv=4158 +24
  = 2ND consecutive sub-+30 →
  MELT-CHAIN NAMED (melt ×2),
  apc=1429 +4 sub-+10 = chain
  continues ×21 (BIG-STEP PAIR
  did NOT land — the +15 was not
  repeated ≥+10), ev 439705/
  108322 (flower +1240 sub-3K),
  bytes +56415 ≥54K = BAND
  RE-OPENED named (heat-run
  streak 1 of the new band;
  NO-HEAT ×5 did NOT form),
  rmax=1) ALERT → then RELEASED
  in-window at [20:59:07] (h=200
  untagged: fb=963 +10 ≥+8 =
  RE-ARM NAMED (first of the
  regime watch forms after the
  +4 break), kv=4192 +34 ≥+30 =
  RE-ARM ×13 LANDS (breaking the
  melt-chain), apc=1430 +1
  sub-+10 = chain ×22 OPEN, ev
  441341/108781 (+1636 sub-3K —
  EV PACE JUMP did NOT land),
  bytes +47639 sub-54K (the band
  closed again on the release row
  — the +56415 heat's lag-1
  successor cleaned:
  armed-and-un-fired)). Closed
  run 5 (no record break; touch
  lineage retired at 8);
  post-release streak 1. apc:
  chain ×22 open (sub-+10
  movements; big-step watch armed
  again on any ≥+10). kv: re-arm
  ×13 (+34). fb: re-arm named
  (+10). On-tag PERFECT; no WALL
  LOST / wall=000; no probe
  failures. Epoch I: 183 of 283
  clean (100 wedge rows).
   ROWLOG_N=612. Cycle 433
   dispatching (fresh lane).

   Cycle 433 (backfill, PROBE-FAIL
   lane). The c433 lane errored
   (fix-287) and the relay paused at
   ~21:36 yesterday — record closed
   late with the full span. In the
   gap: incident investigation (all
   source-verified) named the wedge
   class: a promotion lookup that
   hangs (phantom HIT_PENDING, no
   timeout upstream) plus a stale-
   hit ESCAPE LOOP — request
   8273541c (hit=195584) re-claimed
   the same stale hit every escape
   cycle for ~25 min (6 loops ≈260s:
   fallback ×10s + SERVERKILL ->
   ESCWALL release 240s "until
   boundary change" never came) on
   boot 20261001-143345 (14:33).
   User waited 20 min, killed it at
   [15:26:55]; epoch I ends (100
   wedge rows + this incident).
   Patch suite landed FIRST: commit
   7b44fc4 (promotion-TTL retract +
   failure surfacing + flush
   routing + reset events +
   tombstone hygiene; 7 files) with
   PARITY_GATE_PASS 19/19 into BOTH
   patched-files/ and the venv
   (two earlier aborted boots
   13:44/14:02 = shm_broadcast
   warmup hangs, parent exit, NO
   engine error — cold-reboot
   flakiness, patch NOT implicated;
   third boot healthy). Fixer
   round 2 (fix-288) added late-
   report tolerance + drain_flush
   routing before the assert
   surface; all green. Then user
   directive: "try 1 time, wait 3
   seconds max, no multi-minute
   fallbacks" -> launcher turbo
   (commit 89b660e): five exports
   re-defaulted on BOTH launchers
   (stale-hit TTL 60->3s, fallback
   limit 2->1, escape wall 240->3s,
   promotion TTL 30->3s, fail TTL
   10->3s; backups
   .bak-20261001-turbo; md5 e2f57a
   18 / dc7ec008 replace
   2cab139c/e1ca15f9). Station died
   [14:19:59] rc=0 in reboot chaos;
   restarted [20:10] pid=60644,
   first row NEWBOOTLOG h=200
   hb=200 apc=2 (ROWLOG_N grew 612
   -> ~817 across the outage).
   Turbo boot verified 20:00:11
   (engine-20261001-200011): engine
   environ carry all five knobs
   =3/1/3/3/3; 0 tracebacks/
   assertion errors; traffic
   serving, stores flowing; no
   promotion-timeout/fail lines yet
   (fresh cache, nothing stuck).
   Verdict: PROBE-FAIL cycles 433
   (lane errored; incident
   root-caused and resolved via
   patch suite + turbo knobs). 
   Epoch II declared: new-boot
   watch over the turbo regime
   (expected fast syndrome: with
   wall=3s a bad lookup re-escapes
   every ~4-5s instead of ~260s —
   ESCWALL release density is
   EXPECTED; alert only on engine
   condition, not cadence). Cycle
   434 dispatching (fresh lane).

   Cycle 434 (fresh clean lane,
   epoch II turbo regime): CYCLE_OK.
   First full watched window after
   the turbo boot is green end to
   end, probes 20:11:51 ->
   20:17:00. Station CANARY_ALIVE
   pid=60644 on both probes, no
   restart; HEALTH=200 both
   probes. ROWLOG_N 817 -> 818,
   exactly +1: probe 1 ended on the
   [20:10:34] NEWBOOTLOG row; the
   new row [20:15:35] h=200 p=9
   500=0 oom=0 shm=0 gpu=0 hb=200
   fb=0 kv=1 rmax=0 apc=7
   bytes=+20583, alert-free (no
   SERVE-WEDGE tag); zero
   h=000/hb=000 rows in the window
   (no watch / ALARM TOUCH
   layering, no CYCLE_ALERT).
   Syndrome grep on
   engine-20261001-200011.log: 0
   Traceback / AssertionError /
   ImportError lines.
   Regime health: boot 20:00:11
   now ~17 min stable with serving
   traffic continuous (log tail
   20:17:00-20:17:02: POST
   /v1/chat/completions 200 OK
   pairs + APC-HIT nhash=9/10).
   Promotion retractions
   (promotion timeout / fail /
   suppressed + CPU-TIER-RESET):
   ZERO. Nothing has run into the
   3 s TTLs long enough to need
   retracting; the designed
   outcome is that this surface
   stays quiet while lookups
   resolve normally, so its
   absence is itself proof the
   patch+turbo state machine holds.
   Escape-loop recurrence: NONE.
   No ESCWALL release rows, no
   SERVERKILL churn; the fresh log
   carries one KV-LOOKUP line with
   no re-escape repeat, consistent
   with wall=3 s turning any bad
   lookup into a fast re-escape
   (seconds) instead of the
   epoch-I ~260 s wedge; the 4-min
   stupefying loop did not come
   back. apc 2 -> 7 and stores
   flowing (bytes +20583); nothing
   stuck. ROWLOG_N=818.
   Cycle 435 dispatch pending (orchestrator).

   Cycle 435 (epoch II turbo regime,
   watch lane): CYCLE_OK. Full quiet
   window, probes 20:19:26 ->
   20:24:34. Station CANARY_ALIVE
   pid=60644 on both probes, no
   restart; HEALTH=200 both probes.
   ROWLOG_N 818 -> 819, exactly +1:
   probe 1 ended on the [20:15:35]
   row; the new row [20:20:35]
   h=200 p=9 500=0 oom=0 shm=0
   gpu=0 hb=200 fb=0 kv=2 rmax=0
   apc=20 bytes=+34101, alert-free
   (no SERVE-WEDGE tag); zero
   h=000/hb=000 rows in the window
   (no watch / ALARM TOUCH layering,
   no CYCLE_ALERT). Evidence pass on
   engine-20261001-200011.log
   (newest log re-confirmed — still
   the 20:00:11 boot): 0
   Traceback / AssertionError /
   ImportError lines; promotion
   retractions (promotion timeout /
   fail / suppressed +
   CPU-TIER-RESET): ZERO again this
   window — the designed quiet
   surface stays quiet while lookups
   resolve normally; escape-loop
   recurrence NONE (no ESCWALL
   release rows, no SERVERKILL
   churn; KV-LOOKUP total 2, one new
   this window, resolved with no
   re-escape repeat; the epoch-I
   4-min stupefying wedge did not
   come back). Log tail 20:24:37
   healthy: POST /v1/chat/completions
   200 OK pairs, SpecDecoding + KV
   Transfer metrics live (sync_delay
   count=2, touch_keys=2, stores
   flowing, 0 evictions / 0
   tombstones); all rows oom=0
   gpu=0 — no OOM/GPU watchdog
   patterns. ROWLOG_N=819.
   Cycle 436 dispatch pending (orchestrator).

   Cycle 436 (epoch II turbo regime,
   watch lane): CYCLE_OK. Full quiet
   window, probes 20:26:57 ->
   20:32:05. Station CANARY_ALIVE
   pid=60644 on both probes, no
   restart; HEALTH=200 both probes.
   ROWLOG_N 820 -> 821, exactly +1:
   probe 1 ended on the [20:25:36]
   row; the new row [20:30:37]
   h=200 p=9 500=0 oom=0 shm=0
   gpu=0 hb=200 fb=0 kv=3 rep=0
   apc=31 bytes=+21229, alert-free
   (no SERVE-WEDGE tag); zero
   h=000/hb=000 rows in the window
   (no watch / ALARM TOUCH layering,
   no CYCLE_ALERT). Evidence pass on
   engine-20261001-200011.log
   (newest log re-confirmed — still
   the 20:00:11 boot): 0
   Traceback / AssertionError /
   ImportError lines; promotion
   retractions (promotion timeout /
   fail / suppressed +
   CPU-TIER-RESET): ZERO again this
   window — the designed quiet
   surface stays quiet while lookups
   resolve normally, so its absence
   is itself proof the patch+turbo
   state machine holds; escape-loop
   recurrence NONE (no ESCWALL
   release rows, no SERVERKILL
   churn; KV-LOOKUP count 3,
   resolved with no re-escape
   repeat; the epoch-I 4-min
   stupefying wedge did not come
   back). Log tail 20:32:17
   healthy: POST /v1/chat/completions
   200 OK (192.168.2.208), GET
   /metrics 200 OK, SpecDecoding +
   KV Transfer metrics live
   (mean acceptance 2.33, mirrored
   keys=9738, 0 evictions / 0
   tombstones); apc 26 -> 31 and
   stores flowing (bytes +21229);
   all rows oom=0 gpu=0 — no
   OOM/GPU watchdog patterns.
   ROWLOG_N=821.
    Cycle 437 dispatch pending (orchestrator).

    Cycle 437 (epoch II turbo regime,
    watch lane): CYCLE_OK with one
    documented watch-layer row.
    Window probes 20:33:50 ->
    20:39:21. Station CANARY_ALIVE
    pid=60644 on both probes, no
    restart; HEALTH=200 both probes.
    ROWLOG_N 821 -> 822, exactly +1:
    probe 1 ended on the [20:30:37]
    row; the new row [20:35:57]
    h=200 p=9 500=0 oom=0 shm=0
    gpu=0 hb=000 fb=0 kv=12 rmax=1
    rep=0 apc=55 bytes=+47232
    carries ALERT: SERVE-WEDGE — its
    own serving gate h=200 held on
    that row, and the hb=000 run is
    exactly ONE row: per the
    consecutive-row rule 1 = watch,
    2 = ALARM TOUCH, 3+ =
    CYCLE_ALERT, this lands at watch
    level, not an alarm and not an
    alert. Evidence pass on
    engine-20261001-200011.log
    (newest log re-confirmed — still
    the 20:00:11 boot, no restart,
    ~40 min stabilised uptime, no
    NEWBOOTLOG rows since 20:10:34):
    0 Traceback / AssertionError /
    ImportError lines; 0 OOM / GPU
    watchdog patterns (all rows
    oom=0 gpu=0). Promotion
    retractions (promotion timeout /
    fail / suppressed +
    CPU-TIER-RESET): ZERO again this
    window — nothing stuck to
    retract, the patch surface idle
    by design while lookups churned
    (KV-LOOKUP count 3 -> 35, each
    event resolving with no
    companion ESCWALL / SERVERKILL
    or retraction line);
    escape-loop recurrence NONE (no
    same-request re-escape across
    minutes, no WEDGE-LOOP, the
    epoch-I 4-min stupefying wedge
    did not come back). Log tail
    20:40:07 healthy: POST
    /v1/chat/completions 200 OK
    (192.168.2.208), GET /metrics
    200 OK, SpecDecoding + KV
    Transfer metrics live (mean
    acceptance 2.52, mirrored
    keys=11578, 0 evictions / 0
    tombstones); apc 31 -> 55 and
    stores flowing (bytes +47232).
    ROWLOG_N=822.
     Cycle 438 (backfill, folded with
     incident). First c438 watch
     dispatch (fix-293) died AT
     DISPATCH on infra Bad Gateway
     (~20:44) — no probe, no row,
     repo untouched (verified clean).
     Wedge RECURRENCE documented:
     the 20:00:11 boot wedged again
     [20:44:20] (request
     bb090b5ca2c425a5, ~10 min of
     3 s-wall escape loops pasted by
     the user) -> user rebooted ->
     fresh boot 20:57:34
     (engine-20261001-205734; API
     pid 67769, EngineCore 68312).
     Structural root cause CONFIRMED
     on the fresh boot: turbo knobs
     compressed the wait (fallbacks=1
     wall=0.1 + 3 s ESCWALL) but the
     escape does NOT change the tier
     report for the parked request —
     claim persists (hit=99328
     cov=0/99; groups g0FAE/g2SW1EM/
     g3SW1EM/g4SW1EM each claiming
     98/99) -> release -> re-claim ->
     identical escape, ~4-5 s loop
     forever; fresh boot re-wedged
     within ~15 min (request
     b9211e3a5fe8966f, 60 lines by
     [21:13]) while other users kept
     200-OKs (health 200 throughout —
     not total freeze, but the wedged
     request never terminates).
     Verdict: CYCLE_ALERT:
     WEDGE-LOOP-RECURRENCE (epochs II
     window 434-437 clean; 438
     recurrence before its lane ever
     ran).      request never terminates).
     Patch v2 implemented DIRECTLY
     by the orchestrator (fix-294
     lane infra-failed: upstream
     closed, file untouched; spec
     executed in-line for speed):
     per-request ESCAPE-REFUSAL
     window — knob
     VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_
     SECS (default 60 s, 0 = off):
     on SERVERKILL escape the
     request's tier claims are
     REFUSED for the window (pure
     miss) so the request re-prefills
     ACTIVELY instead of parking;
     refusal clears on request
     teardown and re-arms on repeat
     escapes. ROWLOG_N stays 822
     (first c438 lane died pre-probe;
     station pid 60644 alive).
     Legacy-gloss note: wall=3 s made
     the loop FAST, not SAFE — the
     decision layer was still
     missing. Patch v2 deploy awaits
     orchestrator (user restart
     after, with the new knob
     exported on both launchers).
   Cycle 439 (post-patch-v2 boot lane):
   CYCLE_OK. Window probes 21:50:15 ->
   21:57:31. Station CANARY_ALIVE
   pid=60644 on both probes, no restart;
   HEALTH=200 both probes. ROWLOG_N
   836 -> 837, exactly +1 (md5
   02e4af7c296f3a1d2e611cfb7b68c7df ->
   d2653a05f84d5c0a5719e3db6a67647a);
   the new row [21:53:14] h=200 p=9
   500=0 oom=0 shm=0 gpu=0 hb=200 fb=0
   kv=4 rmax=1 rep=0 apc=6 bytes=
   +121377 ev=1884.0,0.0,0.0, alert-
   free (no SERVE-WEDGE tag). Post-v2
   boot hb ladder: [21:43:13] =
   NEWBOOTLOG informational rebase
   (h=000 p=1, boot 21:42:03; not in
   the miss ladder) and [21:48:14]
   h=000 p=9 hb=000 shm=2 ev=?,?,? =
   ONE isolated fresh hb=000 row
   (boot-settling class; HEALTH probes
   200 at 21:50 and 21:57 straddle it)
   -> watch level only per ladder rule
   (1 = watch), closed by the clean
   successor row — no ALARM TOUCH, no
   CYCLE_ALERT. Evidence pass on
   engine-20261001-214203.log (newest
   log re-confirmed, mtime 21:58:53):
   Traceback=0; ImportError|
   AssertionError=0; OOM-class grep
   8 hits ALL = the benign boot banner
   "If OOM'ed, check ..." x8 ranks at
   21:45:37 ("Initial free memory
   30.53 GiB, reserved 1.97 GiB
   memory for KV Cache") — zero real
   OOM / GPU-watchdog events, canary
   oom=0 on every row. Patch v2 regime
   engaged as designed: 4 isolated
   SERVERKILL escapes at 21:56:21,
   21:56:53, 21:57:31, 21:58:02 on
   FOUR distinct req ids (1 each,
   fallbacks=1 wall=0.2, ~30-40 s
   apart); "KV-LOOKUP escape refusal
   req=... window=60" fired twice
   (21:56:56 req-af6595, 21:58:05
   req-85ba17) — NOT alerts, that is
   patch v2 engaging (quiet hold);
   per-request repeat tally 1 each,
   no rapid repeated escape cycles —
   WEDGE-LOOP-POSTV2 absent; ESCWALL
   release=0; KVL total 23. Log tail
   21:58:53 healthy: POST 200 OKs
   streaming, SpecDecoding + KV
   Transfer metrics live (mirrored
   keys=333864, stores flowing, 0
   evictions / 0 tombstones). Verdict:
   CYCLE_OK — post-v2 boot healthy
   (escapes release fast, refusals
   hold, serving continuous).
   ROWLOG_N=837.
   Cycle 440 (post-patch-v2 boot lane):
   CYCLE_OK. Window probes 22:04:00 ->
   22:17:29, station CANARY_ALIVE pid=60644
   on both, no restart; HEALTH timings 200
   @ 22:04:00 and 200 @ 22:17:29.
   ROWLOG_N 837 -> 839 -> 841 (md5
   d2653a05f84d5c0a5719e3db6a67647a ->
   1fc43af7d1e35eac0a6d96fc7b8a39f7 ->
   d76649a57e4cd5e49372ad29f9ed679a),
   append-only +4 rows across the window,
   no rebase (the [21:43:13] NEWBOOTLOG
   row predates this window; both probes
   saw pure appends). Fresh hb rows:
   [21:58:35] and [22:09:15] hb=000
   SERVE-WEDGE, each ISOLATED between
   clean hb=200 rows ([22:03:55] and
   [22:14:15]) -> watch level only, never
   2 consecutive -> no ALARM TOUCH, no
   CYCLE_ALERT; envelope clean on every
   row (500=0 oom=0 shm=0 gpu=0 rep=0
   rmax=1) and counters climb through all
   windows (fb 0->18, kv 22->135, apc
   6->26, ev 5044->9792->11772). Evidence
   pass on engine-20261001-214203.log
   (newest re-confirmed, mtime 22:17:32,
   430131 B): Traceback=0;
   ImportError|AssertionError=0; real
   OOM / GPU-watchdog = 0. Patch v2
   regime engaged as designed: SERVERKILL
   total 18 on the log = 4 (c439's) + 14
   new (22:01:39 -> 22:10:55) on 17
   distinct req ids, all fallbacks=1
   wall=0.2 spaced 20-90 s; one id
   (84c3892f-978c11f2) escaped twice 60 s
   apart (22:07:08 + 22:08:08) = the
   refusal-window re-arm on re-escape,
   NOT a rapid loop; other repeats 1 each
   and rep=0 on every row ->
   WEDGE-LOOP-POSTV2 absent. Refusals 10
   on the log (2 c439's + 8 new), ALL
   window=60 -> knob still 60 on this
   boot (newest log unchanged, no user
   restart yet; the 5 s default applies
   only after a user restart); ESCWALL
   release=1 (first release recorded on
   this boot); KVL 23 -> 135 (+112
   tier-probe lines in ~19 min, healthy
   volume). Log tail 22:17:30 healthy:
   POST 200 OK streaming, Running 1,
   prompt 6727 tok/s, SpecDecoding metrics
   live, KV stores 12.12 GB, mirrored
   keys 15333, tier full with churn
   (evictable 12412, evicted 652,
   tombstones 208). Verdict: CYCLE_OK
   -> post-v2 boot healthy (escapes
   release fast, refusals hold at
   window=60, serving continuous, miss
   ladder watch-level only). ROWLOG_N=841.
   Cycle 441 (epoch II post-v2 boot):
   CYCLE_OK. Window probes 22:22:11 ->
   22:29:09, station CANARY_ALIVE_60644
   on both, no restart; HEALTH timings
   200 @ 22:22:11 and 200 @ 22:29:09.
   ROWLOG_N 841 (tip) -> 842 -> 843
   (md5 d76649a57e4cd5e49372ad29f9ed679a
   -> 663a82127bbcffb68bce23110eeebb09
   -> f57d79031c7cbe68a293b227e486271e),
   append-only +2 rows across the
   handoff ([22:19:23] pre-window
   append + [22:24:24] in-window), no
   rebase, no NEWBOOTLOG. Boot log
   engine-20261001-214203.log still
   newest (APIPID 82175 unchanged, mtime
   22:29:58, 547471 B) -> no mid-window
   boot change; refusal knob still 60
   (environ VLLM_KV_OFFLOAD_ESCAPE_
   REFUSAL_SECS=60; the user restart to
   the 5 s launcher default has not
   happened yet). Envelope clean on
   every row (500=0 oom=0 shm=0 gpu=0
   rep=0 rmax=1); both window rows
   h=200 hb=200, zero hb=000 rows ->
   watch ladder untouched, no ALARM
   TOUCH; counters climb (fb 18->25,
   kv 136->172, apc 32->36, ev
   12412,1261,357 -> 12412,2832,729 =
   tier full with churn). Evidence pass:
   Traceback=0; ImportError|
   AssertionError=0; real OOM/GPU-
   watchdog=0. SERVERKILL total 28 =
   18 baseline + 10 new (22:19:25 ->
   22:28:39) on 27 distinct log ids,
   the 10 new all fallbacks=1 wall
   0.1-0.2 except one wall=21.2
   (22:22:30, isolated between clean
   rows); all 10 new ids unique -> no
   same-id recape in-window, only
   multi-escape id remains 84c3892f-
   978c11f2 (c440's pair 60 s apart =
   benign re-arm) -> WEDGE-LOOP-POSTV2
   absent. Refusals 16 total (+6 new),
   ALL window=60. ESCWALL release=1
   (the 22:08:08 cold-re-arm line; no
   new release this window). Log tail
   22:29:57 healthy: POST 200 OK
   streaming x3, Running 1, KV-LOOKUP
   hit, APC-HIT nhash=26, KV store
   14.32 GB, mirrored keys 12740,
   evictable 12404, tombstones 1586.
   Verdict: CYCLE_OK -> post-v2 boot
   healthy (refusals hold window=60,
   escapes release fast, serving
   continuous, hb ladder clean, no
   loop alerts). ROWLOG_N=843.
   Cycle 442 (epoch II post-v2 boot):
   CYCLE_OK. Window probes 22:34:32 ->
   22:41:26, station CANARY_ALIVE_60644
   on both, no restart (pid 60644
   alive); HEALTH timings 200 @
   22:34:32 and 200 @ 22:41:26.
   ROWLOG_N 843 -> 845 -> 846 (md5
   f57d79031c7cbe68a293b227e486271e ->
   5c9b4f3d09233b802e897054cb0cf910 ->
   cccc141b756bacee4ad1e8f6b01c9621),
   append-only +3 rows from the c441
   tip ([22:29:27] and [22:34:30]
   landed after c441's read; one true
   window row), no rebase, no
   NEWBOOTLOG. Window row [22:39:50]
   h=200 p=9 500=0 oom=0 shm=0 gpu=0
   hb=000 fb=35 kv=232 rmax=1 rep=0
   apc=56 bytes=+50626 ev=12404.0,
   11503.0,2878.0 carries ALERT:
   SERVE-WEDGE; the hb=000 run is
   exactly ONE row, isolated between
   clean hb=200 rows ([22:34:30] and
   the successor) -> watch level only
   per ladder rule; no 2-consecutive
   -> no ALARM TOUCH, no CYCLE_ALERT.
   Envelope clean on every row (500=0
   oom=0 shm=0 gpu=0 rep=0 rmax=1);
   counters climb (fb 28->35, kv
   188->232, apc 45->56, ev
   12412,8296,2091 -> 12404,11503,2878
   = tier full with churn). Boot log
   engine-20261001-214203.log still
   newest (mtime 22:48:41, 752075 B)
   -> no mid-window rebase; refusal
   knob still 60 with direct environ
   proof from the scripted pgrep -f
   'vllm serve' read: API id 82175's
   environ holds
   VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS
   =60 (the user restart to the 5 s
   launcher default has not happened
   yet; the 21:42:10 envs.py WARNING
   calling the var unknown is benign
   boot noise, the connector reads it
   directly and every refusal line
   proves it active). Evidence pass on
   the boot log: Traceback=0;
   ImportError|AssertionError=0;
   OOM/GPU-watchdog raw=0, real=0.
   Patch v2 regime engaged as
   designed: SERVERKILL total 41 = 28
   (c441 baseline) + 13 new (22:31:41
   -> 22:48:22), all fallbacks=1, wall
   0.2-0.3 except one wall=0.9
   (22:48:22), on 13 NEW distinct req
   ids with every event a distinct id
   (inter-event spacing 16-295 s) ->
   no same-id re-escape in-window;
   log-wide per-id tally: 41 escape
   lines over 40 distinct ids, only
   84c3892f-978c11f2 ever escaped
   twice (c440's pair 60 s apart =
   benign refusal-window re-arm) ->
   WEDGE-LOOP-POSTV2 absent. Refusals
   24 total (+8 new, 22:31:44 ->
   22:45:34), ALL window=60. ESCWALL
   release=1 (still only the 22:08:08
   cold re-arm line; no new release
   this window). KVL total 272. Log
   tail 22:48:41 healthy: POST 200 OK
   x2 (192.168.2.208), Running 1 /
   Waiting 1, prompt 2875.6 tok/s,
   SpecDecoding live (mean acceptance
   3.18), per-position acceptance
   0.862/0.707/0.614, KV Transfer
   metrics live (mirrored keys=33894,
   cpu_allocated=12412,
   evictable=12412, free list=0), MFU
   7.6 TF/s/GPU 170.8 GB/s/GPU.
   Verdict: CYCLE_OK -> post-v2 boot
   healthy (refusals hold window=60,
   escapes single-strike and spaced,
   serving continuous, hb ladder
   watch-level only, no loop alerts).
   ROWLOG_N=846.
   Cycle 443, attempt 2 after the
   infra-timeout attempt 1 (fix-299;
   no side effects, repo clean):
   CYCLE_ALERT. Window probes
   00:15:57 -> 00:22:49, station
   CANARY_ALIVE_60644 on both, no
   restart (pid 60644 alive);
   HEALTH timings 200 @ 00:15:57
   and 200 @ 00:22:49. ROWLOG_N
   864 -> 865 -> 868 (md5
   ca702f3e1bfdaa93a326615c465b622c ->
   b91125aa63702d45a1c28b07c367c6a6 ->
   90ee8b82ee011848b9c0cefee722fc33),
   append-only +22 rows from the
   c442 tip 846, no mid-window
   rebase at the rowlog (18 rows
   landed pre-probe, one true
   window row [00:18:47], +3 at the
   00:36:17 evidence read). Epoch
   event in-span (landed
   pre-dispatch): old epoch
   wall-died ~23:11 ([23:11:17]
   h=000 p=7 ev=?,?,?), restart
   minted engine-20261001-231328
   at 23:13:28; rowlog marks
   [23:16:17] ALERT: NEWBOOTLOG
   (fb=0 kv=0 ev=?,?,?), [23:21:18]
   h=000 warm-up, recovery
   [23:26:38] h=200. Post-restart
   deltas: counters reset rows
   853-854 then climb (fb 4->303,
   kv 16->1354, apc 1->42, ev churn
   at tier-full 12412); station pid
   60644 survived the epoch change;
   the 23:41 baselines were taken
   on this same boot so none need
   rebasing (c442-era SK 41 / KVL
   272 / ESCWALL 1 totals died with
   the 214203 log). Boot state:
   API id 100422 alive (lstart Thu
   Oct 1 23:13:27 = the 23:41-fact
   boot), EngineCore 100982,
   refusal knob direct environ
   proof VLLM_KV_OFFLOAD_ESCAPE_
   REFUSAL_SECS=2 (scripted pgrep
   read); newest engine log
   re-confirmed 231328 (mtime
   00:36:14, 1159733 B) so patch
   v3.1 (fee6cbf, both dests md5
   da376d25, .bak-20261001-claim2)
   is staged on disk but NOT
   loaded - the running engine
   predates it, regime shift pends
   the user restart. Envelope
   clean on every h=200 row (500=0
   oom=0 shm=0 gpu=0 rep=0).
   Evidence pass (date-aware
   recount vs 23:41:30; the 23:41
   manual baseline SK 76 / REF 0 /
   ESCWALL 64 / KVL 112 reproduces
   exactly - my first since-23:41
   crawl mis-grouped post-midnight
   lines and was corrected):
   Traceback=0; ImportError|
   AssertionError=0; OOM raw=0
   real=0 (no banner hits this
   boot). SERVERKILL 305 = 76 +
   229 NEW (23:41:44 -> 00:35:10);
   KVL 504 = 112 + 392; ESCWALL
   release 252 = 64 + 188; escape
   refusal lines 1 (00:15:58
   window=2 = designed inertness
   under v2-era code: the 2 s knob
   expires before the 3 s wall
   check so the quarantine never
   engages); 'escape claim
   blocked' lines 0 (debug-only,
   not at INFO - the v3.1
   signature appears only
   post-restart).
   WEDGE-LOOP-POSTV31 ALERT FIRED:
   rapid same-req escapes at
   few-second intervals:
   a8f5a2303ab27c9c 3-s cadence
   run from 23:42:12;
   abf4c15de42aa3fb 14 escapes
   exactly 3 s apart
   00:32:07 -> 00:32:46;
   b97d21e2a88ed193 escape pair
   00:35:07/00:35:10 with ESCWALL
   releases '(no boundary change;
   cold re-arm)' wall=3.0 x2 - the
   classic loop under v2-era
   wall-cycling. Top wedged req
   ids (whole-log):
   a55730aa2436f326=43 (all
   post-baseline, newest hot
   wedge), a44edd70f7347af3=39
   (unchanged), abf4c15de42aa3fb=31
   (all new), ac6b62d8a85ce6f4=17,
   a8f5a2303ab27c9c=17,
   a8d54557bd6e244d=16 (unchanged).
   Ladder over span rows 847-868:
   2-run [22:50:18]+[22:55:39] =
   ALARM TOUCH level, closed by
   [23:00:56] hb=200; restart
   cluster [23:06:17]-[23:26:38]
   carries the epoch-fact h=000
   rows; isolated W rows
   [23:42:11]/[23:52:35] = watch;
   then 5-consecutive hb=000 rows
   [00:08:07][00:13:27][00:18:47]
   [00:24:08][00:29:28] (SERVE-WEDGE
   on each) -> 3+ = CYCLE_ALERT per
   ladder, closed by [00:34:32]
   hb=200. Serving never dropped
   post-warm-up: h=200 every
   h-column row, HEALTH 200 both
   probes, log tail 00:36 healthy
   (POST 200 OK x6 from
   192.168.2.208, Running 1 /
   Waiting 1, SpecDecoding mean
   acceptance 2.89-3.11, KV
   Transfer mirrored 55008-56456
   keys, MFU 0.6-2.1 TF/s).
   Verdict: CYCLE_ALERT ->
   pre-v3.1 wedge-wall loop
   intensification (same-id 3 s
   re-escape bursts + 5-run
   hb=000) while serving held; NOT
   the v3.1 regime signature (per
   new fingerprint one SERVERKILL
   + optional release, then
   silence) - the user restart
   must load v3.1 to break the
   loops; WEDGE-LOOP-POSTV31 stays
   armed for the 231328 boot.
   ROWLOG_N=868.
    Cycle 444 (one-shot, epoch II, no
    restart): CYCLE_ALERT. Probes
    00:43:05 -> 00:49:26, station
    CANARY_ALIVE_60644 on both,
    pid 60644 alive throughout, no
    restart; HEALTH timings 200 @
    00:43:05 and 200 @ 00:49:26.
    ROWLOG_N 869 -> 871 -> 872 (md5
    299ec35a25d96d66ceb93c3ee9a79814
    -> 5bfe18b02edf331cedead55c9e14ed8a
    -> e8fe59e782d65bd007ab5fe7f9f42c48),
    append-only. Regime: STILL
    baseline boot engine-20261001-231328
    (v2-era; user did not restart
    this cycle - API pid 100422
    lstart Thu Oct 1 23:13:27,
    refusal knob = 2 environ-proven,
    'escape claim blocked' count = 0
    so the staged v3.1 was never
    loaded; WEDGE-LOOP-POSTV31 stays
    armed; no post-v3.1 incident to
    report). Engine evidence passes
    00:54:11 + 00:56:42 + ~01:00
    census: SERVERKILL 305 -> 432
    total (+75 by 00:54, +127 by
    ~01:00; corrected new-window
    00:36+ = 81 at 00:56), while
    the c443 top wedge ids stayed
    frozen (a55730aa2436f326=43,
    a44edd70f7347af3=39,
    abf4c15de42aa3fb=31: 0 new
    each) and the burst load moved
    to fresh ids led by NEW top
    wedge chatcmpl-ad87dfae2667eaf5
    =25 with the classic 3-s
    cadence escape run
    00:53:14 -> 00:53:29
    (hit=179200 fallbacks=1
    wall=0.1 -> escape; ESCWALL
    release wall=3.0 'no boundary
    change; cold re-arm' pairs),
    self-cleared ~00:54, then more
    same-pattern bursts from other
    fresh ids through 00:59:00
    (last KV-LOOKUP SERVERKILL
    chatcmpl-baf918c29f8971ac
    hit=15360; the SK rapid-cluster
    flagger only echoes one
    historical burst on
    chatcmpl-bffdc5f4ccf992c7).
    ESCWALL release 252 -> 361
    (+109), KV-LOOKUP hit 504 ->
    674 (+170), escape refusal 1 ->
    2 (+1 isolated benign line
    00:54:13 chatcmpl-a4608872af39eb0e
    window=2; no same-id refusal
    pairs). Traceback=0,
    ImportError/AssertionError=0,
    real CUDA OOM/GPU watchdog=0
    (OOM_RAW=0). Rowlog ladder:
    after the [00:34:32] hb=200
    close, 3-consecutive hb=000
    rows [00:39:53][00:45:13]
    [00:50:33], each ALERT:
    SERVE-WEDGE -> 3+ = CYCLE_ALERT
    per the ladder. h=200, 500=0,
    oom=0 on every row: the wedges
    stayed wall-cycling yet
    self-clearing with serving
    intact (log tail POST 200 OK,
    Running 1, gen ~39-113 tok/s,
    SpecDec accept 2.44-3.46, MFU
    2.8-15.6 TF/s, KV Transfer
    mirrored 20k-333k keys).
    Verdict: CYCLE_ALERT -> the
    v2-era wedge-wall loop regime
    persists AND widened across
    fresh req ids (+127 SK new)
    while serving held; still NOT
    the v3.1 per-id
    one-escape-then-silence
    fingerprint (claim block never
    logged) - only the user restart
    into staged v3.1 can break the
    loops. ROWLOG_N=872.
     Cycle 445 (one-shot, epoch II, no
     restart): CYCLE_ALERT (standing,
     expected pre-restart wall-cycling
     regime). Probes 01:09:49 ->
     01:16:26 on testcomp2, station
     CANARY_ALIVE_60644 both, pid
     60644 alive throughout, no
     restart; HEALTH timings 200 @
     01:09:49 and 200 @ 01:16:26.
     ROWLOG_N 874 -> 875 (md5
     1d8de84b9bf74ab0546dbeb28afd962b
     -> 303d00af81cc7dd4eb0e27d2cc524429;
     append-only). Regime: STILL
     baseline boot engine-20261001-231328.
     log (v2-era; user did not restart
     this cycle - API pid 100422 lstart
     Thu Oct 1 23:13:27, refusal knob
     = 2 environ-proven,
     VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS,
     claim blockout count = 0 so the
     staged v3.1 was never loaded;
     WEDGE-LOOP-POSTV31 stays armed;
     no post-v3.1 incident to report).
     Engine evidence pass ~01:26
     census: SERVERKILL 432 -> 518
     total, +86 new in the 01:00 ->
     01:26:31 window, while the c444
     top wedge ids stayed frozen exact
     (a55730aa2436f326=43,
     a44edd70f7347af3=39,
     abf4c15de42aa3fb=31,
     ad87dfae2667eaf5=25; 0 new each)
     and baf918c29f8971ac=47 closed
     out 00:50-01:00; post-census load
     moved to fresh ids led by NEW top
     wedge chatcmpl-82eccb5ddd9a76bd
     =35 (35 SK all >=01:00 plus one
     benign escape refusal at
     01:04:56 window=2), then
     chatcmpl-a4929a2bb818a4c8=10,
     chatcmpl-a99f2a15f6a0f05d=9 with
     the classic 3-s cadence escape
     run 01:20:07 -> 01:20:22
     (hit=192512 fallbacks=1 wall=0.1
     -> escape; ESCWALL release
     wall=3.0 'no boundary change;
     cold re-arm' pairs), then
     chatcmpl-9820eab3f3db553a 3-s run
     01:21:30 -> 01:21:38 (last SK
     01:21:38; its first escape hit
     the refusal window=2 at 01:21:30
     with the long-held ESCWALL
     release wall=37.8; no refused-id
     same-id re-escape). escape
     refusal 2 -> 5 (+3 isolated
     benign lines: 01:01:02
     chatcmpl-80f4147d65c1e510,
     01:04:56 chatcmpl-82eccb5ddd9a76bd,
     01:21:30 chatcmpl-9820eab3f3db553a;
     exactly one refusal line per id
     - no same-id refusal pairs).
     ESCWALL release 361 -> 427
     (+66), KV-LOOKUP hit 674 -> 800
     (+126). Traceback=0,
     ImportError/AssertionError=0,
     real CUDA OOM/GPU watchdog=0
     (OOM_RAW=0, claim blockout=0).
     Rowlog ladder: after the c444
     standing 3-run [00:39:53]
     [00:45:13][00:50:33] the hb=000
     rows continued with NO recovery
     - 6-consecutive hb=000 rows
     00:39:53 -> 01:06:34, each
     ALERT: SERVE-WEDGE, finally
     closed by [01:11:37] hb=200 ->
     3+ = CYCLE_ALERT per the ladder.
     h=200, 500=0, oom=0 on every
     row: serving intact through the
     engine tail (POST 200 OK total
     356, latest 01:23; SpecDec
     accept 2.24-2.80, MFU 0.6-5.9
     TF/s, healthy KV-LOOKUP hits
     through 01:23:28). Verdict:
     CYCLE_ALERT (standing,
     recorded-but-expected) -> the
     v2-era wedge-wall loop regime
     persists (3-s re-escape bursts
     + 6-run hb=000) while serving
     held; still NOT the v3.1 per-id
     one-escape-then-silence
     fingerprint (claim block never
     logged) - only the user restart
     into staged v3.1 can break the
     loops. ROWLOG_N=875.
      Cycle 446 (one-shot, epoch II, no restart):
      CYCLE_ALERT (standing, expected pre-restart
      wall-cycling regime). Probes 01:31:16 ->
      01:41:35 on testcomp2, station
      CANARY_ALIVE_60644 both, pid 60644 alive
      throughout, no restart needed; HEALTH
      timings 200 @ 01:31:16 and 200 @ 01:41:35.
      ROWLOG_N 875 -> 880 (md5
      303d00af81cc7dd4eb0e27d2cc524429 ->
      ce1670e44113e708b6a694ccb3399929;
      append-only). Regime: STILL baseline boot
      engine-20261001-231328.log (v2-era; user
      did not restart this cycle - API pid
      100422 lstart Thu Oct 1 23:13:27 2026,
      refusal knob = 2 environ-proven
      VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS,
      claim blockout count = 0 so the staged
      v3.1 was never loaded; WEDGE-LOOP-POSTV31
      stays armed; no post-v3.1 incident to
      report). Engine evidence pass ~01:43
      census: SERVERKILL 518 -> 605 total, +87
      all after the 01:26:31 census, on fresh
      ids led by NEW top wedge
      chatcmpl-b85345868b69b2db=41 (3-s cadence
      escape run 01:39:02 -> 01:41:02,
      hit=134144 fallbacks=1 wall=0.1 -> escape;
      ESCWALL release wall=3.0 'no boundary
      change; cold re-arm' pairs), then
      chatcmpl-a7d7ca4522378833=17 (burst
      01:32:51 -> 01:33:39),
      chatcmpl-b2394986999297bd=14 (01:29:20 ->
      01:29:59), then small tail
      (8fccced6aecc6ac7=5, 8809088992bea04b=4,
      869da31df487bda3=3, b15c20533888ad07=2,
      97ef7d5132f24645=1); frozen ids stayed
      frozen exact (a55730aa2436f326=43,
      a44edd70f7347af3=39, abf4c15de42aa3fb=31,
      ad87dfae2667eaf5=25; baf918c29f8971ac=47,
      82eccb5ddd9a76bd=35, a4929a2bb818a4c8=10,
      a99f2a15f6a0f05d=9, 9820eab3f3db553a=4
      all closed: 0 new each). escape refusal 5
      -> 5 (+0, no new refusal lines). ESCWALL
      release 427 -> 506 (+79), KV-LOOKUP hit
      800 -> 933 (+133). Traceback=0,
      ImportError/AssertionError=0, real CUDA
      OOM/GPU watchdog=0 (OOM_RAW=0, claim
      blockout=0). Rowlog ladder: c445's closed
      6-run held; isolated hb=000 [01:16:57]
      bracketed by closes [01:22:10][01:27:23]
      (watch level), then fresh 2-consecutive
      hb=000 rows [01:32:43][01:38:04] landed
      this cycle = ALARM TOUCH level within the
      standing regime, no close row yet by
      probe 2 (01:41:35). h=200, 500=0, oom=0
      on every row: serving intact through the
      engine tail (POST 200 OK total 407, +51
      since c445, latest in the 01:43 tail;
      SpecDec accept 3.17, MFU 2.6 TF/s,
      Running 1). Verdict: CYCLE_ALERT
      (standing, recorded-but-expected) -> the
      v2-era wedge-wall loop regime persists
      AND broadened to fresh ids (+87 SK new)
      while serving held; still NOT the v3.1
      per-id one-escape-then-silence
      fingerprint (claim block never logged) -
      only the user restart into staged v3.1
      can break the loops. ROWLOG_N=880.
      Cycle 447 (one-shot, epoch II, no
      restart): CYCLE_ALERT (standing, expected
      pre-restart wall-cycling regime). Probes
      01:49:34 -> 01:58:03 on testcomp2,
      station CANARY_ALIVE_60644 both, pid
      60644 alive throughout, no restart
      needed; HEALTH timings 200 @ 01:49:34 and
      200 @ 01:58:03. ROWLOG_N 882 -> 883 (md5
      1b9e8539a50f36b84060f8c9caebba4d ->
      3e4c71d5e84265ae04ed87af4f9f2264;
      append-only). Regime: STILL baseline boot
      engine-20261001-231328.log (v2-era; user
      did not restart this cycle - API pid
      100422 unchanged, refusal knob = 2
      environ-proven
      VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS,
      claim blockout count = 0, fd1/2 still on
      the 231328 file, so the staged v3.1 was
      never loaded; WEDGE-LOOP-POSTV31 stays
      armed; no post-v3.1 incident to report).
      Engine evidence pass ~02:12 census
      (9616-line 231328 log): SERVERKILL 605 ->
      677 total, +72 since c446, growth all on
      fresh distributed small ids (top of the
      current census: acd4af385c026201=21,
      997c5c6e07d21b11=20, ac6b62d8a85ce6f4=17,
      a8f5a2303ab27c9c=16, a8d54557bd6e244d=16;
      3-s cadence loops seen 02:06:49 -> :55
      [997c5c6e... hit=71680 fallbacks=1] and
      02:08:28 -> :34 [8bb2074a... hit=180224]
      -> escape until boundary change);
      every tracked top id from c446 froze
      exact (b85345868b69b2db=41,
      a7d7ca4522378833=17,
      b2394986999297bd=14, a55730aa2436f326=43,
      a44edd70f7347af3=39, abf4c15de42aa3fb=31,
      ad87dfae2667eaf5=25, baf918c29f8971ac=47,
      82eccb5ddd9a76bd=35). escape refusal 5 ->
      7 (+2: 01:49:35, 02:02:40, window=2,
      benign). ESCWALL release 506 -> 569
      (+63), KV-LOOKUP hit 933 -> 1063 (+130).
      Traceback=0, ImportError/AssertionError
      =0, real CUDA OOM/GPU watchdog=0,
      0 5xx in error-tail grep. Rowlog ladder:
      c446's pair closed with hb=200
      [01:43:07], then fresh 2-consecutive
      hb=000 rows [01:48:27][01:53:47] landed
      this cycle = 2nd ALARM TOUCH level of
      epoch II, watch level, no 3rd strike.
      h=200, 500=0, oom=0 on every row:
      serving intact through the engine tail
      (POST 200 OK total 466, +59 since c446,
      latest ~02:12; SpecDec accept 2.67, MFU
      0.6 TF/s). Verdict: CYCLE_ALERT
      (standing, recorded-but-expected) -> the
      v2-era wedge-wall loop regime persists
      and spread to fresh small ids (+72 SK
      new) while serving held; still NOT the
      v3.1 per-id one-escape-then-silence
      fingerprint (claim block never logged) -
      only the user restart into staged v3.1
      can break the loops. ROWLOG_N=883.

      Cycle 448 (one-shot, epoch II, no
      restart): CYCLE_ALERT (standing, expected
      pre-restart wall-cycling regime; hb
      ladder deepened then recovered). Probes
      02:18:42 -> 02:24:55 on testcomp2,
      station CANARY_ALIVE_60644 both, pid
      60644 alive throughout, no restart
      needed; HEALTH timings 200 @ 02:18:42 and
      200 @ 02:24:55. ROWLOG_N 887 -> 888 (md5
      8c9b759cc028cf2d21d47a6e90cd4b05 ->
      ef1b1d60d161e6c946017a085e3da09a;
      append-only). Regime: STILL baseline boot
      engine-20261001-231328.log (v2-era; user
      did not restart this cycle - API pid
      100422 unchanged, refusal knob = 2
      environ-proven
      VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS,
      claim blockout count = 0, so the staged
      v3.1 was never loaded;
      WEDGE-LOOP-POSTV31 stays armed; no
      post-v3.1 incident to report). Engine
      evidence pass ~02:26 census (10390-line
      231328 log): SERVERKILL 677 -> 687
      total, +10 since c447, growth all on
      fresh distributed small ids; every
      tracked top id froze exact except one
      +1 (baf918c29f8971ac=47,
      a55730aa2436f326=43, b85345868b69b2db=41,
      a44edd70f7347af3=39, 82eccb5ddd9a76bd=35,
      abf4c15de42aa3fb=31, ad87dfae2667eaf5=25,
      acd4af385c026201=21, 997c5c6e07d21b11=20,
      ac6b62d8a85ce6f4=17, a7d7ca4522378833=17,
      a8f5a2303ab27c9c 16 -> 17,
      a8d54557bd6e244d=16, b2394986999297bd=14).
      Fresh burst seen live in the tail:
      3-s cadence loop 02:11:50 -> :59 on
      802890cea860a673 (hit=28672 fallbacks=1,
      SERVERKILL x4 in 9 s -> escape until
      boundary change + ESCWALL release cold
      re-arm; a wedge-burst cycle, EXPECTED
      under v2-era rules), plus one-off SKs
      aa372013de3001ad 02:15:38,
      88e7ed6b624f5d56 02:16:13,
      a023988faf70586d 02:17:50,
      827af552a769784c 02:19:13 wall=33.9.
      30-s-window same-id escape detector:
      CLEAN (no 3+ same-id escape-refusal
      window). escape refusal 7 -> 9 (+2:
      83e5eb1ce6c1f7f3 02:11:15,
      827af552a769784c 02:19:13, window=2,
      benign; every tracked escape id
      re-checked = single-escape, no
      re-escape anywhere). ESCWALL release
      569 -> 573 (+4), KV-LOOKUP hit 1063 ->
      1133 (+70). Traceback=0,
      ImportError/AssertionError=0, real CUDA
      OOM/GPU watchdog=0, HTTP 5xx = 0 whole
      log. Rowlog ladder: c447's pending
      ALARM-TOUCH pair [01:48:27][01:53:47]
      DEEPENED this cycle to the epoch's
      deepest streak - 5 consecutive hb=000
      rows [01:48:27][01:53:47][01:59:08]
      [02:04:28][02:09:48] all SERVE-WEDGE
      with progress-and-envelope clean, then
      recovered: hb=200 [02:15:00] and clean
      [02:20:00]. h=200, 500=0, oom=0, shm=0,
      gpu=0 on every row: serving intact
      through the engine tail (POST 200 OK
      total 545, +79 since c447, latest in
      the 02:25 tail; SpecDec accept 2.56,
      MFU 0.6 TF/s). Verdict: CYCLE_ALERT
      (standing, recorded-but-expected) -> the
      v2-era wedge-wall loop regime persists
      at its deepest hb ladder yet (5-strike,
      self-recovered) while serving held;
      still NOT the v3.1 per-id one-escape-
      then-silence fingerprint (claim block
      never logged) - only the user restart
      into staged v3.1 can break the loops.
      ROWLOG_N=888.
      Cycle 449 (one-shot, epoch II, no
      restart): CYCLE_ALERT (standing,
      expected pre-restart wall-cycling
      regime; no new escalation, ladder
      watch-grade). Probes 02:29:21 ->
      02:37:39 on testcomp2, station
      CANARY_ALIVE_60644 both, pid 60644
      alive throughout, no restart needed;
      HEALTH 200 @ 02:29:21 and 200 @
      02:37:39. ROWLOG_N 889 -> 891 (md5
      bf3c28e2e72fff337babe6ca35d61608 ->
      0b4a2d4e28e6c31c468cd03f0ced3ef2;
      c448-end baseline 888, md5
      ef1b1d60d161e6c946017a085e3da09a;
      append-only). Fresh rows [02:25:01]
      and [02:30:11] clean hb=200, then a
      single isolated SERVE-WEDGE row
      [02:35:31] hb=000 - watch-grade
      solo, no 2+/3+ zero-ladder, so the
      epoch's deepest mark stays the c448
      5-strike run [01:48:27]..[02:09:48],
      recovered [02:15:00], clean
      [02:20:00]. h=200, 500=0, oom=0,
      shm=0, gpu=0 on every row: serving
      intact. Regime: STILL baseline boot
      engine-20261001-231328.log (v2-era;
      user did not restart this cycle -
      API pid 100422 unchanged, refusal
      knob = 2 environ-proven
      VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS;
      claim blockout count = 0, so the
      staged v3.1 was never loaded;
      WEDGE-LOOP-POSTV31 stays armed; no
      post-v3.1 incident to report).
      Engine evidence pass ~02:38 census:
      SERVERKILL 687 -> 754 total, +67
      since c448; the long-frozen veteran
      wall froze EXACT again:
      (baf918c29f8971ac=47,
      a55730aa2436f326=43,
      b85345868b69b2db=41,
      a44edd70f7347af3=39,
      82eccb5ddd9a76bd=35,
      abf4c15de42aa3fb=31,
      ad87dfae2667eaf5=25,
      acd4af385c026201=21,
      997c5c6e07d21b11=20,
      a7d7ca4522378833=17,
      a4929a2bb818a4c8=10,
      a99f2a15f6a0f05d=9,
      b2394986999297bd=14; interior
      a8f5a2303ab27c9c=17,
      ac6b62d8a85ce6f4=17,
      a8d54557bd6e244d=16 also froze) -
      no loop growth on any tracked id.
      Fresh live v2-era wedge-burst this
      cycle, EXPECTED class:
      858a90beec8840c1, 8x SERVERKILL in
      ~21 s 02:42:01 -> :22 (hit=94208
      fallbacks=1 wall=0.1, SK -> ESCWALL
      release cold re-arm no boundary
      change -> re-hit -> re-SK, 3-s
      cadence loop); c448's burst id
      802890cea860a673 stayed silent =4
      (last event 02:11:59); one-off
      b30ae33b2777629a 02:45:15
      wall=151.0 SK + escape refusal,
      closed while observed. 30-s-window
      same-id detector: at escape-refusal
      granularity CLEAN (fresh refusals
      are isolated singles); at coarser
      SK-line sweep only the historical
      vet-burst set plus the live
      858a90bee loop surface, nothing new
      beyond it - expected pre-restart,
      NOT post-v3.1. escape refusal 9 ->
      11 (+2: 85ef4b4730794ea9 02:37:41,
      b30ae33b2777629a 02:45:15, window=2
      knob-matched, benign, every fresh
      escape id single-escape, no
      re-escape loop beyond the vetted
      v2-era class). ESCWALL release 573
      -> 626 (+53), KV-LOOKUP hit 1133 ->
      1252 (+119). Traceback=0,
      ImportError/AssertionError=0, real
      CUDA OOM/GPU watchdog=0, HTTP 5xx =
      0 whole log, POST 200 OK total 611
      (+66 since c448, latest in the
      02:45 tail with APC-HIT flow).
      Verdict: CYCLE_ALERT (standing,
      recorded-but-expected) -> the
      v2-era wedge-wall loop regime
      persists at watch-grade hb activity
      while serving held; still NOT the
      v3.1 per-id one-escape-then-silence
      fingerprint (claim block never
      logged) - only the user restart
      into staged v3.1 can break the
      loops. ROWLOG_N=891.
      Cycle 450 (one-shot, epoch II, no
      restart): CYCLE_ALERT (standing,
      expected pre-restart wall-cycling
      regime; no new escalation, rowlog
      ladder watch-plus, self-recovered
      twice). Probes 02:59:05 ->
      03:04:27 on testcomp2, station
      CANARY_ALIVE_60644 both, pid 60644
      alive throughout, no restart
      needed; HEALTH 200 @ 02:59:05 and
      200 @ 03:04:27. ROWLOG_N 895 ->
      896 (md5
      f4b1e80d919402aa3d58ebbd95782791
      -> 6ac8fb3495c1cd8b21f886c93cb99
      9e9; c449-end baseline 891, md5
      0b4a2d4e28e6c31c468cd03f0ced3ef2
      ; append-only). Fresh rows:
      [02:40:51] hb=000 completed an
      ALARM-TOUCH-grade 2-run with
      c449's [02:35:31] solo, recovered
      [02:46:00]; isolated [02:51:20]
      hb=000 tracking the engine's
      fresh wedge cycle, recovered
      [02:56:32]; clean [03:01:34]. No
      3+ zero-ladder; h=200, 500=0,
      oom=0, shm=0, gpu=0 on every row:
      serving intact through the engine
      tail (POST 200 OK total 710, +99
      since c449, latest in the 03:06
      tail; SpecDec accept 2.75, MFU
      1.6 TF/s). Regime: STILL baseline
      boot engine-20261001-231328.log
      (v2-era; user did not restart
      this cycle - API pid 100422
      unchanged, refusal knob = 2
      environ-proven
      VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_S
      ECS; claim blockout count = 0
      whole log, so the staged v3.1
      was never loaded;
      WEDGE-LOOP-POSTV31 stays armed;
      no post-v3.1 incident to
      report). Engine evidence pass
      ~03:07 census (log 12549 lines):
      SERVERKILL 754 -> 774 total, +20;
      frozen veteran wall froze EXACT
      again:
      (baf918c29f8971ac=47,
      a55730aa2436f326=43,
      b85345868b69b2db=41,
      a44edd70f7347af3=39,
      82eccb5ddd9a76bd=35,
      abf4c15de42aa3fb=31,
      ad87dfae2667eaf5=25,
      acd4af385c026201=21,
      997c5c6e07d21b11=20,
      a7d7ca4522378833=17, interior
      ac6b62d8a85ce6f4=17,
      a8f5a2303ab27c9c=17,
      a8d54557bd6e244d=16,
      a4929a2bb818a4c8=10,
      a99f2a15f6a0f05d=9,
      b2394986999297bd=14;
      802890cea860a673 silent =4) - no
      loop growth on any tracked id.
      Fresh live v2-era wedge-bursts,
      EXPECTED class:
      8a08cf7bbe4b9e37 6x SK 02:49:55
      -> :09 and ac080e587a1e6e56 6x
      SK 02:51:28 -> :42 (3-s cadence,
      hit-boundary SK -> escape
      refusal window=2 -> ESCWALL
      release cold re-arm -> re-hit ->
      re-SK -> silence), plus singles
      and 2x: 929425484d8f8d9b
      02:48:19-:22, 92e5cb18792402cd
      02:49:11 (wall=1.3),
      9aefa7fe9b5b455e 02:53:21
      (wall=69.9 SK + refusal),
      b6b7ea1e93dc49e4 02:55:56,
      a59305ee34cd2a61 02:56:29,
      9c8633c13fe51342 03:06:43-:46 -
      total +20; c449's burst id
      858a90beec8840c1 stayed
      historical 8x (last 02:42:22),
      and 802890cea860a673 stayed
      silent =4 (last event 02:11:59).
      30-s-window same-id detector: at
      escape-refusal granularity CLEAN
      (11 -> 14 lines, +3 fresh ids
      8a08cf7bbe4b9e37 02:49:55,
      ac080e587a1e6e56 02:51:28,
      9aefa7fe9b5b455e 02:53:21, all
      window=2 knob-matched, benign,
      every fresh escape id
      single-escape, no re-escape
      loop); at coarser SK-line sweep
      only the historical vet-burst
      set plus the two fresh 6x loops
      above - expected pre-restart,
      NOT post-v3.1. ESCWALL release
      626 -> 639 (+13), KV-LOOKUP hit
      1252 -> 1333 (+81). Traceback=0,
      ImportError/AssertionError=0,
      real CUDA OOM/GPU watchdog=0,
      HTTP 5xx = 0 whole log.
      Verdict: CYCLE_ALERT (standing,
      recorded-but-expected) -> the
      v2-era wedge-wall loop regime
      persists at watch-plus hb
      activity while serving held;
      still NOT the v3.1 per-id
      one-escape-then-silence
      fingerprint (claim block never
      logged) - only the user restart
      into staged v3.1 can break the
      loops. ROWLOG_N=896.
      Cycle 451 (one-shot, epoch II, no
      restart): CYCLE_ALERT (standing,
      expected pre-restart wall-cycling
      regime; no new escalation, rowlog
      ladder single-rung watch). Probes
      03:10:41 -> 03:16:00 on testcomp2,
      station CANARY_ALIVE_60644 both,
      pid 60644 alive throughout, no
      restart needed; HEALTH 200 @
      03:10:41 and 200 @ 03:16:00.
      ROWLOG_N 897 -> 898 (md5
      64c7348921cad840568e918f9cf5cad6
      -> ec5231efa08e6649b7396ad72fe558
      37; c450-end baseline 896, md5
      6ac8fb3495c1cd8b21f886c93cb99
      9e9; append-only, NEWBOOTLOG
      never appeared). Fresh rows:
      [03:06:38] clean h=200 closing
      c450, then [03:11:58] hb=000
      solo isolated watch rung
      tracking the engine's fresh
      wedge cycle (prior rungs
      03:01:34/03:06:38 clean;
      recovery rung not yet posted at
      the 03:16:00 probe); h=200,
      500=0, oom=0, shm=0, gpu=0 on
      every row: serving intact
      through the engine tail (POST
      200 OK total 752, +42 since
      c450, live GET /metrics 200
      tail 03:20:10; SpecDec accept
      2.10). Regime: STILL baseline
      boot engine-20261001-231328.log
      (v2-era; user did not restart
      this cycle - API pid 100422
      unchanged, refusal knob = 2
      environ-proven
      VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_S
      ECS=2; claim blockout count = 0
      whole log, so the staged v3.1
      was never loaded;
      WEDGE-LOOP-POSTV31 stays armed;
      no post-v3.1 incident to
      report). Engine evidence pass
      ~03:20 census (log 13117
      lines): SERVERKILL 774 -> 797
      total, +23, every pre-existing
      id EXACT-frozen again, 16/16
      vet match incl. burst ids
      (baf918c29f8971ac=47,
      a55730aa2436f326=43,
      b85345868b69b2db=41,
      a44edd70f7347af3=39,
      82eccb5ddd9a76bd=35,
      abf4c15de42aa3fb=31,
      ad87dfae2667eaf5=25,
      acd4af385c026201=21,
      997c5c6e07d21b11=20,
      a7d7ca4522378833=17,
      b2394986999297bd=14,
      a4929a2bb818a4c8=10,
      a99f2a15f6a0f05d=9;
      802890cea860a673 silent =4;
      858a90beec8840c1 healed =8;
      c450 loops 8a08cf7bbe4b9e37=6,
      ac080e587a1e6e56=6 stay
      closed). Fresh live v2-era
      wedge-bursts, EXPECTED class,
      all 03:16:52 -> 03:19:44
      post-c450 tail, new ids:
      a7a50b84974f629a 4x SK
      03:16:52 -> :01 refusal-free
      (ESCWALL cold re-arm releases
      16:55-17:01),
      b2a12fc9c31fa513 single SK
      03:18:34 wall=8.7 hit=192512
      (refusal 03:18:34 window=2),
      af177fd6260e0c5f 5x SK
      03:17:52 -> 03:18:04, first
      wall=30.1 then 0.1-cadence,
      hit=120832 local=0, refusal
      03:17:53 window=2, releases
      17:54-18:03, and new TOP burst
      8e007ca1da212703 13x SK
      03:19:08 -> :44 within 36 s,
      hit=121856, refusal-free,
      ESCWALL chain every 3 s -
      total +23. Escape refusal 14 ->
      16 (+2), both new lines
      window=2 knob-matched, every
      fresh escape id single-escape,
      no re-escape loop on any id.
      30-s-window same-id detector:
      at escape-refusal granularity
      CLEAN (0 wedge windows whole
      log); SK-line sweep shows
      8e007ca1da212703 13x and
      af177fd6260e0c5f 5x fresh, plus
      the historical vet-burst set -
      expected pre-restart, NOT
      post-v3.1. ESCWALL release 639
      -> 658 (+19), KV-LOOKUP hit
      1333 -> 1381 (+48). Traceback=0,
      ImportError/AssertionError=0,
      real CUDA OOM/GPU watchdog=0,
      HTTP 5xx = 0 whole log.
      Verdict: CYCLE_ALERT (standing,
      recorded-but-expected) -> the
      v2-era wedge-wall loop regime
      persists at watch-grade hb
      activity while serving held;
      still NOT the v3.1 per-id
      one-escape-then-silence
      fingerprint (claim block never
      logged) - only the user restart
      into staged v3.1 can break the
      loops. ROWLOG_N=898.
      Cycle 452 (one-shot, epoch II, no
      restart): CYCLE_ALERT (standing,
      expected pre-restart wall-cycling
      regime; no new escalation, though
      the rowlog ladder touched its
      3-rung CYCLE_ALERT grade at
      03:22:39 and healed clean well
      before probe 2). Probes 03:29:40
      -> 03:38:51 on testcomp2, station
      CANARY_ALIVE_60644 both, pid 60644
      alive throughout, no restart
      needed; HEALTH 200 @ 03:29:40 and
      200 @ 03:38:51. ROWLOG_N 898 ->
      903 (md5
      ec5231efa08e6649b7396ad72fe55837
      -> c4b04808684d630da00d7b96f96b05
      b0; c451-end baseline 898 as
      recorded; probe1 @03:29:40 already
      read 901, md5
      c3ecd2d44691a3c296ff06b2f38f25fe,
      so those rows predate my census;
      append-only, NEWBOOTLOG never
      appeared). Fresh rows: [03:17:19]
      hb=000 (2nd consecutive with
      c451's 03:11:58 -> ALARM TOUCH
      rung), [03:22:39] hb=000 (3rd
      consecutive -> ladder CYCLE_ALERT
      grade), then recovery [03:27:54]
      hb=200 and clean rows [03:33:07] +
      [03:38:08]; h=200, 500=0, oom=0,
      shm=0, gpu=0 every row: serving
      intact (POST 200 OK total 828, +76
      since c451, live GET /metrics 200
      tail 03:39:11; SpecDec accept
      2.46). Regime: STILL baseline boot
      engine-20261001-231328.log
      (v2-era; user did not restart this
      cycle - API pid 100422 unchanged,
      refusal knob = 2 environ-proven
      VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_S
      ECS=2; claim blockout count = 0
      whole log, so the staged v3.1 was
      never loaded; WEDGE-LOOP-POSTV31
      stays armed; no post-v3.1 incident
      to report). Engine evidence pass
      ~03:39 census (log 14156 lines,
      +1039): SERVERKILL 797 -> 861
      total, +64, every pre-existing id
      EXACT-frozen again, 21/21 vet
      match incl. c451-frozen ids
      (8e007ca1da212703=13,
      af177fd6260e0c5f=5,
      a7a50b84974f629a=4,
      b2a12fc9c31fa513=1; also
      87cf48eadb6324ea=27 confirmed
      pre-existing, zero fresh lines).
      Fresh live v2-era wedge-bursts,
      EXPECTED class, all 03:20:32 ->
      03:36:37 post-c451 tail: new TOP
      burst bdb1bd99838f808b 18x SK
      03:20:32 -> :23 within 51 s,
      hit=122880 fallbacks=1,
      refusal-free, ESCWALL cold re-arm
      chain x17; bc063ec677ac9dfb 7x
      03:35:29 -> :47 hit=75776;
      a9ff5d4319db5c25 4x (03:28:56 ->
      03:29:04, refusal 03:28:56
      window=2), 8fbb9f96e1f49e34 4x
      (03:30:15 -> :23), ad1d8df0393957af
      4x (03:32:46 -> :55),
      a769fdab66b6a184 3x (03:27:57 ->
      03:28:03, refusal 03:27:57),
      b8630fa5ea974417 3x (03:25:25 ->
      :31) and 16 shorter fillers -
      total +64 exact. Escape refusal
      16 -> 19 (+3), all new lines
      window=2 knob-matched
      (b4dce4e0ab47315d 03:27:16,
      a769fdab66b6a184 03:27:57,
      a9ff5d4319db5c25 03:28:56), every
      fresh escape id single-escape, no
      re-escape loop on any id,
      30-s-window same-id refusal
      detector CLEAN (0 wedge windows
      whole log). ESCWALL release 658 ->
      700 (+42, cold re-arm class),
      KV-LOOKUP hit 1381 -> 1517 (+136).
      Traceback=0, ImportError/
      AssertionError=0, real CUDA
      OOM/GPU watchdog=0, HTTP 5xx = 0
      whole log. Verdict: CYCLE_ALERT
      (standing, recorded-but-expected)
      -> the v2-era wedge-wall loop
      regime persists at watch-grade hb
      activity while serving held; 3
      consecutive hb=000 rungs (c451's
      03:11:58 + 03:17:19 + 03:22:39)
      reached the ladder CYCLE_ALERT
      grade, then healed clean
      (03:27:54, 03:33:07, 03:38:08)
      through both probes; still NOT the
      v3.1 per-id
      one-escape-then-silence
      fingerprint (claim block never
      logged) - only the user restart
      into staged v3.1 can break the
      loops. ROWLOG_N=903.
      Cycle 453 (one-shot, epoch II, no
      restart): CYCLE_ALERT (standing,
      recorded-but-expected; no new
      escalation, rowlog ladder stayed
      healed clean). Probes 03:44:25 ->
      03:51:35 on testcomp2, station
      CANARY_ALIVE_60644 both, pid
      60644 alive throughout, no
      restart needed; HEALTH 200 @
      03:44:25 and 200 @ 03:51:35.
      ROWLOG_N 904 -> 905 (md5
      2242398831fe17edb92916326d898a9c
      -> 1fc4cebf93ebd8dbb311caf18931e37d;
      c452-end baseline 903 / md5
      c4b04808684d630da00d7b96f96b05b0
      taken from record text (md5 strings
      split by the wrap; probe md5s are
      authoritative start/end); probe1
      @03:44:25 already read 904, so one
      row predates my census; append-only,
      NEWLOG never appeared). The
      pre-census row [03:43:12] and my
      fresh row [03:48:13] are both
      hb=200 clean; h=200, 500=0, oom=0,
      shm=0, gpu=0 every row; NO new
      hb=000 rungs - the c452 3-rung
      ladder (03:11:58 / 03:17:19 /
      03:22:39) stays the last ladder and
      stayed healed through both probes
      (03:27:54 -> 03:38:08 clean, then
      [03:43:12] + [03:48:13]): watch
      grade only. Regime: STILL baseline
      boot engine-20261001-231328.log
      (v2-era; user did not restart this
      cycle - API pid 100422 unchanged,
      refusal knob = 2 environ-proven
      VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_S
      ECS=2; claim blockout count = 0
      whole log, so the staged v3.1 was
      never loaded; no post-restart
      incident, first post-v3.1
      microdetail stays pending;
      WEDGE-LOOP-POSTV31 stays armed).
      Engine evidence pass ~03:52 census
      (log 14640 lines, +484):
      SERVERKILL 861 -> 861, ZERO fresh
      SK lines this window, every
      pre-existing id EXACT-frozen, 21/21
      vet match (incl
      8e007ca1da212703=13,
      af177fd6260e0c5f=5,
      baf918c29f8971ac=47), all 8 c452
      burst ids delta-0
      (bdb1bd99838f808b=18 still top,
      refusal-free, ESCWALL chain x17,
      closed; bc063ec677ac9dfb=7; 4x
      a9ff5d4319db5c25 /
      8fbb9f96e1f49e34 /
      ad1d8df0393957af; 3x
      a769fdab66b6a184 /
      b8630fa5ea974417;
      87cf48eadb6324ea=27). Escape
      refusal 19 -> 19 (+0), last line
      still 03:28:56, every escape id
      single-escape, no re-escape on any
      id, 30-s-window same-id refusal
      detector CLEAN whole log. ESCWALL
      release 700 -> 700 (+0, context
      number cross-checked against
      record), KV-LOOKUP hit 1517 ->
      1533 (+16 benign serving lookups
      03:40:19 -> 03:52:00, local=0).
      Traceback=0, ImportError/
      AssertionError=0, real CUDA
      OOM/GPU watchdog=0, HTTP 5xx = 0
      whole log. Serving intact: POST
      200 OK total 869 (+41 since
      c452), live GET /metrics 200 tail
      03:51:55; SpecDec mean accept len
      3.60. Verdict: CYCLE_ALERT
      (standing, recorded-but-expected)
      -> the v2-era wedge-wall loop
      regime persists at watch-grade hb
      activity while serving held;
      still NOT the v3.1 per-id
      one-escape-then-silence
      fingerprint (claim block never
      logged) - only the user restart
      into staged v3.1 can break the
      loops. ROWLOG_N=905.
Cycle 454 (one-shot, epoch II, no
restart): CYCLE_ALERT (standing,
recorded-but-expected; same standing
grade - no new escalation) -> a
fresh 3-rung hb=000 ladder
(03:53:33 + 03:58:53 + 04:04:14)
re-reached the ladder CYCLE_ALERT
grade and was still un-healed at
probe end (heal watch next cycle);
v2-era wedge-wall loop regime
persists while serving held; still
NOT the v3.1 per-id
one-escape-then-silence fingerprint
(claim blockout count = 0 whole
log) - only the user restart into
staged v3.1 can break the loops.
Probes 03:56:46 -> 04:04:37 on
testcomp2, station
CANARY_ALIVE_60644 both, pid 60644
alive throughout, no restart
needed; HEALTH 200 @ 03:56:46 and
200 @ 04:04:37. ROWLOG_N 906 -> 908
(md5 5a6b4aa71908c6b149ec7087e94d4280
-> efb907549c0d3caee919b6e28a4aaffc;
probe md5s authoritative start/end;
handoff's N=905 predates rows; row
906 = [03:53:33] hb=000 rung came
before my probe1 census). Rows: h=
200, 500=0, oom=0, shm=0, gpu=0
every row; fresh rungs [03:58:53]
(907) + [04:04:14] (908); c452
3-rung ladder stayed healed through
03:48:13, new 000s start 03:53:33;
SERVE-WEDGE tag on 000 rows as
before (v2-era pattern). Regime:
STILL baseline boot
engine-20261001-231328.log (v2-era;
API pid 100422 unchanged, refusal
knob = 2 environ-proven
VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS
=2; claim blockout count = 0 whole
log so the staged v3.1 was never
loaded; no post-restart incident,
first post-v3.1 microdetail stays
pending; WEDGE-LOOP-POSTV31 stays
armed). Path note: engine logs live
at /mnt/data/shared/models/
vllm-glm-5.3-flash-nvfp4/logs
(sm120-logs dir = canary station
files only). Engine evidence census
@04:13:02 (log 15906 lines, +1266):
SERVERKILL 861 -> 930 (+69 fresh,
wedge-burst window 04:02 - 04:12,
every fresh event on a NEW window
id, every vetted pre-existing id
frozen EXACT with counts unchanged
incl baf918c29f8971ac=47,
a55730aa2436f326=43,
b85345868b69b2db=41,
a44edd70f7347af3=39,
82eccb5ddd9a76bd=35,
abf4c15de42aa3fb=31,
87cf48eadb6324ea=27,
ad87dfae2667eaf5=25,
acd4af385c026201=21,
997c5c6e07d21b11=20,
bdb1bd99838f808b=18,
8e007ca1da212703=13,
858a90beec8840c1=8; window burst id
tally closes exact: 17 ids = 69).
Top window bursts (v2-era expected
class): a1cb9c778c6c0671 17x
(04:11:05 -> :53, ~3s ladder,
ESCWALL-chained loop), baea315
b3e82f190 13x (04:10:08 -> :44, 3s),
b156fc9e9a21b886 8x (04:03:25 ->
:46), bd66140596ac36c5 6x (04:06:42
-> :56), b4c4c064624d3035 4x,
af9b32cf71a29611 3x,
a6dd8f47b3043c47 3x, b34c9b
639593716e 3x, plus 2x id fills
(993d0e8c0a5a4144, aa5c90
c692158b7a, ae500239c8d066a7) and 6
singleton ids; same-id SK loops
recorded expected pre-restart.
Escape refusal 19 -> 21 (+2 fresh:
bd66140596ac36c5 04:06:42,
af9b32cf71a29611 04:08:58, both
window=2 knob-matched), every
escape id single-refusal, no
re-escape refusal on any id,
30-s-window same-id refusal
detector CLEAN (0 wedge windows
whole log). ESCWALL release 700 ->
752 (+52, all cold re-arm class, no
boundary change, local=0).
KV-LOOKUP hit 1533 -> 1690 (+157
benign serving lookups through
census, local=0). Traceback=0,
ImportError/AssertionError=0, real
CUDA OOM/GPU watchdog=0, HTTP 5xx =
0 whole log. Serving intact: POST
200 OK total 984 (+115 since c453),
live GET /metrics 200 through
census; SpecDec mean accept len
2.99 @04:13:01. ROWLOG_N=908.
       Cycle 455 (epoch II; v2-era boot
engine-20261001-231328 unchanged, API
pid 100422, knob
VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
environ-proven again, fd-resolved live
log matches newest; no restart, no
rebase): station pid 60644 ALIVE at
both probes, HEALTH=200 @04:18:20 and
HEALTH=200 @04:24:20, one 300 s wait
between. ROWLOG_N=910 -> 911
(MD5 aeae24973ce335968c5c8ada725dfab5 ->
e34538f60dc14578c2d3edf6b3f9ba3e; the
new 04:19:39 row is hb=200 and heals
the 04:14:39 isolated hb=000 run).
Engine pass (log 16518 lines) is
v2-era: SERVERKILL 930 -> 952 (+22
fresh, all one-off class: top in-window
frames abfc57ff1db69e9e 2x
(04:17:51 -> :54 ESCWALL-chained),
86e288aa7bbfbfe6 2x (04:23:28 -> :30),
a3ba586ec5335993 and a2951fa6f2a71546
1x, remainder singleton; no new burst
id, ledger vets frozen EXACT incl
a1cb9c778c6c0671=17 kept, baf918c29
f8971ac=47, a55730aa2436f326=43,
b85345868b69b2db=41, a44edd70f7347
af3=39). Escape refusal 21 -> 22 (+1
fresh: abfc57ff1db69e9e @04:17:49
window=2), every id single-refusal, 30
-s same-id wedge detector CLEAN whole
log. ESCWALL release 752 -> 767 (+15,
cold re-arm, no boundary change,
local=0). KV-LOOKUP hit 1690 -> 1733
(+43, benign serving lookups).
Traceback=0, ImportError/
AssertionError=0, real CUDA OOM/GPU
watchdog=0 whole log. Serving intact:
POST /v1 200 OK plus live
GET /metrics 200 through 04:26, SpecDec
mean accept len 2.05 @04:26:21. hb
ladder state: prior un-healed 3-rung
(03:53 / 03:58 / 04:04) closed here
(04:09 recovery audited, 04:19:39
healed follow-up), the 04:14:39 run
stayed isolated, no 2-consecutive
window run, ALARM not touched. Standing
CYCLE_ALERT persists unchanged until
the user restart flips the regime;
zero-delta rest keeps the standing
verdict, no escalation. Verdict: QUIET
(clean cycle, boot still pre-v3.1).
       Cycle 456 (epoch II; v2-era boot
engine-20261001-231328 unchanged, API
pid 100422, fd-resolved live log
matches newest file by mtime under
/mnt/data/shared/models/
vllm-glm-5.3-flash-nvfp4/logs, knob
VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
environ-proven again on the running
pid; no restart, no rebase, no regime
flip): station pid 60644 ALIVE at both
probes, HEALTH=200 @04:33:58 and
HEALTH=200 @04:40:20, one 300 s wait
between. ROWLOG_N=911 -> 915 (c455 md5
e34538f6..., 913 bf6f42be..., 915
1af9c031...; new rows 04:24:41 and
04:29:50 hb=200, 04:35:10 isolated
hb=000 rung, healed by 04:40:19
hb=200, no 2-run window, ALARM not
touched). Engine pass @04:43:29
(scripted, fd-resolved): SERVERKILL
952 -> 1015 (+63 through census) and
re-census @04:49:06 SK=1025 (+10
further live serving growth; ledger
delta +73 exact = corrected in-window
frames >=04:26 on 10-02; window: 23
distinct fresh ids, 9 singleton + 4
diads, no ledger vet moved). Top
window bursts (v2-era expected
ESCWALL-chained class):
963164aa471a2b63 11x (04:35:08 ->
:38), ba2c157b3d7aac75 8x
(04:31:57 -> 04:32:18), bd205ad8ce7e
99c0 7x (04:36:12 -> :30),
9399e65cbb67f0fe 7x (04:44:06 ->
:23), 96e734145190642a 5x
(04:26:37 -> :49), b41c16f
b4a867335 4x; remainder diads and
singleton ids. Frozen vets EXACT
whole-log check re-run: all 20 ids
unchanged incl baf918c29f8971ac=47,
a55730aa2436f326=43,
b85345868b69b2db=41,
a44edd70f7347af3=39, 82eccb5ddd
9a76bd=35, abf4c15de42aa3fb=31,
87cf48eadb6324ea=27, ad87dfae2667
eaf5=25, acd4af385c026201=21,
997c5c6e07d21b11=20, bdb1bd99838f
808b=18, 8e007ca1da212703=13,
a1cb9c778c6c0671=17;
midscale ac6b62d8a85ce6f4=17
unchanged; none of the 20 moved.
Escape refusal 22 -> 25 (+3 through
census: b0046a5eeb03cd1b @04:33:58,
85a14409d356e457 @04:41:12,
94c51ad5eb962d1a @04:42:10, plus +2
further one-offs by re-census:
9399e65cbb67f0fe @04:44:06,
b7485652de1371fe @04:46:17, every
line window=2 knob-matched). Per-id
refusal max = 1 over the whole grown
log (no re-escape on any id); 30-s
same-id refusal detector CLEAN at two
independent checks. One-escape
microdetail (94c51ad5eb962d1a): hit
local=0 hit=151552 @04:42:07 ->
stale-hit fallback + SERVERKILL
@04:42:09 -> refusal @04:42:10 ->
ESCWALL release wall=3.7 cold re-arm
@04:42:11 -> hits resume, no
re-refusal (silence). ESCWALL release
767 -> 810 (+43 cold re-arm through
census; 818 by re-census). KV-LOOKUP
hit 1733 -> 1871 (+138 through
census; 1893 by re-census, benign
serving lookups, local=0).
Traceback=0, ImportError/
AssertionError=0, real CUDA OOM/GPU
watchdog=0 whole log; probe rows
500=0 oom=0 gpu=0 throughout; POST
class 200 OK 1170 lines, GET /metrics
200 OK 4001 in log, live self-metrics
200 @04:51, last log line metrics
200. Serving intact: h=200 p=9 both
probes; SpecDec counters alive
(drafts 615556, accepted 942463,
~1.53 accept/draft @04:51). hb ladder
state (this cycle): 04:24:41 and
04:29:50 hb=200, 04:35:10 isolated
rung, 04:40:19 recovery = closed;
prior 03-rung stayed closed from
c455; no 2-consecutive window, ALARM
not touched. No restart happened, so
no post-restart incident and no
WEDGE-LOOP-POSTV31 evaluation
applicable; first post-v3.1
microdetail stays pending. Standing
CYCLE_ALERT persists unchanged until
the user restart flips the regime;
this cycle stayed inside the standing
regime with no new trigger, no
escalation, no rebase. Verdict: QUIET
(clean cycle, boot still pre-v3.1;
wedge-burst SK volume higher than
c455 but vet-frozen, single-refusal,
wedge-clean, healed hb row).
       Cycle 457 (epoch II; v2-era boot
       engine-20261001-231328 unchanged,
       API pid 100422, fd-resolved live
       log = newest file (FDMATCH=yes on
       both scripted passes), knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS
       =2 environ-proven again; no restart,
       no rebase, no regime flip): station
       pid 60644 ALIVE at both probes,
       HEALTH=200 @04:51:57 and HEALTH=200
       @04:57:18, one 300 s wait between.
       ROWLOG_N=915 -> 920 (probe1 917
       md5 7e83a27c..., probe2 918 md5
       10f875ce..., end 920 md5 c9c59407
       across canary5m.log; new rows
       04:45:39 hb=000 isolated rung,
       healed by 04:50:40 hb=200;
       04:55:41 hb=200; 05:01:01 second
       isolated hb=000 rung, healed by
       05:06:02 hb=200; no 2-run window,
       ALARM not touched).
       Baseline note (discrepancy): the
       c456 refusal arrow said end=25 but
       its own named incidents (+3 then
       +2) imply 27 and the live log
       agrees: 27 whole log, ZERO fresh
       after 04:49:06, so true c456 end
       RF=27; engine log authoritative;
       handoff cross-check said 25.
       Engine census @05:00:23 (scripted,
       fd-resolved, log 18283 lines):
       SERVERKILL 1025 -> 1026 (+1 through
       census, +0 by re-census @05:06:49;
       window 04:49:06 -> 05:06:49 closes
       exact: one fresh frame 04:59:30 =
       one-off id a4825dcf5f9794d7 1x; no
       burst id, no wedge-burst window
       this cycle, top bursts = none).
       One-escape microdetail
       (a4825dcf5f9794d7): KV-LOOKUP hit
       local=0 hit=1024 @04:59:29 ->
       stale-hit fallback wall=0.2 ->
       SERVERKILL @04:59:30 -> escape
       (0,False) -> NO refusal follows, NO
       ESCWALL release yet, hits resume
       (silence); rowlog rung 05:01:01
       straddles it, healed 05:06:02.
       Per-id refusal max = 1 over whole
       log (no re-escape on any id);
       30-s same-id wedge detector CLEAN
       at both scripted checks
       (WEDGE30_COUNT=0 whole log).
       Frozen vets EXACT whole log:
       baf918c29f8971ac=47,
       a55730aa2436f326=43,
       b85345868b69b2db=41,
       a44edd70f7347af3=39,
       82eccb5ddd9a76bd=35,
       abf4c15de42aa3fb=31,
       87cf48eadb6324ea=27,
       ad87dfae2667eaf5=25,
       acd4af385c026201=21,
       997c5c6e07d21b11=20,
       bdb1bd99838f808b=18,
       a1cb9c778c6c0671=17,
       8e007ca1da212703=13; midscale
       ac6b62d8a85ce6f4=17 and wedge-era
       id 963164aa471a2b63=11 all hold
       (none of the ledger-frozen moved).
       Escape refusal 27 -> 27 (+0 fresh;
       all 27 lines window=2 knob-matched).
       ESCWALL release 818 -> 818 (+0;
       last releases b7485652de1371fe
       @04:46:18 wall=3.6 and @04:46:21
       wall=3.0, cold re-arm class).
       KV-LOOKUP hit 1893 -> 1910 (+17
       benign serving lookups; sample
       lines local=0 except one
       local=1024 @05:02:44).
       Traceback=0, ImportError/
       AssertionError=0, real CUDA OOM
       /GPU watchdog=0 whole log.
       Serving intact: POST 200 OK 1170
       -> 1253 (+83), GET /metrics 200
       OK 4001 -> 4209 (+208) in log;
       probe rows 500=0 oom=0 gpu=0
       throughout; last log line POST
       /v1 200 OK; apc counter alive
       (364 -> 430 by 05:06).
       No restart happened, so no
       post-restart incident and no
       WEDGE-LOOP-POSTV31 evaluation
       applicable; first post-v3.1
       microdetail stays pending.
       Standing CYCLE_ALERT persists
       unchanged until the user restart
       flips the regime; this cycle
       stayed inside the standing
       regime, no new trigger, no
       escalation, no rebase. Verdict:
       QUIET (clean cycle, boot still
       pre-v3.1; wedge output collapsed
       to a single one-off frame;
       detector CLEAN whole log).
       Cycle 458 (epoch II; pre-restart
       regime held through the whole
       cycle; user did not restart):
       station pid 60644 alive at both
       probes, no restart needed;
       probe1 @05:09:52 HEALTH=200;
       single 300 s wait; probe2
       @05:14:56 HEALTH=200.
       ROWLOG_N 920 -> 921 (+1 row
       05:11:03 h=200 p=9 500=0);
       ROWLOG_MD5 920-end
       c9c59407377a94d95cb924dcaf41cc0b
       -> 921-end
       5d609e41710997ad5cd926c56bf5512e.
       Regime: newest engine log still
       engine-20261001-231328.log;
       LIVELOG fd of API PID 100422
       resolves /mnt/data/shared/
       models/vllm-glm-5.3-flash-nvfp4/
       logs/engine-20261001-231328.log
       (19042 lines @05:16:25 census);
       boot still v2-era; refusal-knob
       environ re-read N/A (no new
       boot appeared through 05:16).
       Frozen-census deltas whole log:
       SERVERKILL 1026 -> 1026 (+0);
       escape refusal 27 -> 27 (+0);
       ESCWALL release 818 -> 818 (+0);
       KV-LOOKUP hit 1910 -> 1923 (+13
       benign serving hits, samples
       local=0 @05:17:44, @05:18:20);
       Traceback=0, ImportError/
       AssertionError=0, real CUDA OOM
       /GPU watchdog=0 whole log.
       Last escape line in log remains
       the ledgered one-off
       a4825dcf5f9794d7 @04:59:30; zero
       escape events after it -> cycle
       window 05:06 -> 05:18 incident-
       clean (no burst ids this cycle).
       Frozen vets EXACT (SK-only
       counts; 82eccb... = 35 SK + 1
       refusal explains raw-36):
       baf918c29f8971ac=47,
       a55730aa2436f326=43,
       b85345868b69b2db=41,
       a44edd70f7347af3=39,
       abf4c15de42aa3fb=31,
       87cf48eadb6324ea=27,
       ad87dfae2667eaf5=25,
       acd4af385c026201=21,
       997c5c6e07d21b11=20,
       bdb1bd99838f808b=18,
       a1cb9c778c6c0671=17,
       8e007ca1da212703=13,
       ac6b62d8a85ce6f4=17,
       963164aa471a2b63=11: none moved.
       Scripted 30-s same-id detector:
       all FLAG30 windows predate 05:06
       (burst frames 23:26-23:42,
       00:27-02:40, 04:03-04:46 =
       ledgered wedge-burst cycles,
       expected pre-restart; top
       repeats 12-hex:
       baf918c29f89=47, a55730aa2436=43,
       b85345868b69=41, a44edd70f734=39,
       abf4c15de42a=31); zero new flags
       after 04:59:30 (c457
       WEDGE30_COUNT=0 frame stands).
       hb ladder unchanged: two isolated
       rungs (04:45:39, 05:01:01) both
       healed by next row; 05:11:03
       row clean; ALARM untouched;
       TB/OOM/GPU-watchdog/5xx=0 whole
       way; serving intact (POST /v1
       200 OK; APC counter 430 -> 444).
       Post-restart incident: none (no
       restart; first post-v3.1
       microdetail stays pending;
       WEDGE-LOOP-POSTV31 not
       applicable under v2-era boot).
       Zero-delta quiet cycle ->
       standing regime verdict
       persists: QUIET (no new wedges;
       frozen vets EXACT; ladder rungs
       isolated and healed; staged
       v3.1 claim-block still
       post-restart-gated).
       Cycle 459 (epoch II; pre-restart v2-era boot
       held through the whole cycle; station pid
       60644 alive at both probes, no restart, no
       rebase): probe1 @05:21:52 HEALTH=200; single
       300 s wait; probe2 @05:28:02 HEALTH=200.
       ROWLOG_N 923 -> 924 (new rows: 05:16:22 h=200
       p=9 500=0 clean; 05:21:43 hb=000 isolated
       wedge rung with ALERT: SERVE-WEDGE marker, fb
       frozen 1026, healed by 05:26:43 hb=200; no
       2-run window, ALARM untouched); ROWLOG_MD5
       923-end 5d57a2d0e5be862f30beb0e8e8461bcf ->
       924-end 38e1e3880ac1c4f5965d3ae03e8826a4;
       handoff baseline said 921/5d609e41:
       superseded by two later aprows, probe
       authoritative (append-only, NEWBOOTLOG not
       counted).
       Regime: newest engine log still
       engine-20261001-231328.log; LIVELOG fd of API
       PID 100422 resolves
       /mnt/data/shared/models/vllm-glm-5.3-flash-
       nvfp4/logs/engine-20261001-231328.log;
       FDMATCH=yes; log 19042 lines @05:16:25 ->
       19759 @05:32:21; boot v2-era; knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       environ-proven on pid 100422; no new boot
       through 05:32.
       Whole-log counts: SERVERKILL 1026 -> 1026
       (+0); escape refusal 27 -> 27 (+0); ESCWALL
       release 818 -> 818 (+0); KV-LOOKUP hit 1923
       -> 1950 (+27 benign serving lookups: 1947
       @05:29:28, 1950 @05:32:50; one local=2048
       @05:21:16 line, all others local=0; no
       escape/refusal/SERVERKILL follow-up; engine
       log matches handoff cross-counter 1923).
       Fresh-window (10-02, t>=05:06:49, date-aware
       scripted recheck): FW_EVENTS=0 whole:
       SERVERKILL=0, refusal=0, ESCWALL=0; no burst
       ids this cycle; ALERTSTRINGS_FRESH=0;
       last escape event in log remains the
       ledgered one-off a4825dcf5f9794d7 @04:59:30
       (single frame, no repeat, silence pattern
       intact after it).
       Scripted 30-s same-id wedge detector: 141
       FLAG30 frames ALL predate 04:59:30; last
       flag window b7485652de1371fe 04:46:16 ->
       04:46:17; burst clusters 23:25-23:53,
       00:27-02:53, 03:06-04:46 = ledgered
       wedge-burst cycles, expected pre-restart;
       zero flags after 04:59:30; detector CLEAN
       for this cycle's window (c457 WEDGE30=0
       frame stands).
       Census-tool note: first scripted pass used a
       wrong seconds-cut (189409s) and a later pass
       a date-bug (10-01 23:xx passing a 10-02
       cut); both only inflated fresh-window
       counters, never hid events; whole-log totals
       plus the date-aware recheck are authoritative
       and reconcile (KVL fresh 40 = c458 tail 13 +
       24 by 05:29 + 3 by 05:32 = 1950 end total).
       Frozen vets EXACT whole log (SK-only counts,
       none moved): baf918c29f8971ac=47,
       a55730aa2436f326=43, b85345868b69b2db=41,
       a44edd70f7347af3=39, 82eccb5ddd9a76bd=35,
       abf4c15de42aa3fb=31, 87cf48eadb6324ea=27,
       ad87dfae2667eaf5=25, acd4af385c026201=21,
       997c5c6e07d21b11=20, bdb1bd99838f808b=18,
       ac6b62d8a85ce6f4=17, a8f5a2303ab27c9c=17,
       a7d7ca4522378833=17, a1cb9c778c6c0671=17.
       Traceback=0, ImportError/AssertionError=0,
       real CUDA OOM/GPU watchdog=0 whole log and
       in the fresh window; probe rows 500=0 oom=0
       gpu=0 throughout.
       Serving intact: POST /v1 200 OK 1253 -> 1370
       (+117 @05:29:28), GET /metrics 200 OK 4209
       -> 4485 (+276); apc 444 -> 491; fb frozen
       1026 since 04:50:40; kv rowlog counter 4820
       -> 4840; last raw log lines POST 200 OK +
       GET 200 OK.
       Post-restart incident: none (no restart;
       first post-v3.1 microdetail stays pending;
       WEDGE-LOOP-POSTV31 not applicable under the
       v2-era boot).
       Zero-delta quiet cycle -> standing regime
       verdict persists: QUIET (no new wedges;
       frozen vets EXACT; ladder rungs isolated and
       healed; boot still pre-v3.1; staged v3.1
       claim-blockout still post-restart-gated).
       Cycle 460 (epoch II; v2-era boot held
       through the whole cycle; user did not
       restart, no rebase): station pid 60644
       alive at both probes, no restart
       needed; probe1 @05:37:13 HEALTH=200;
       single 300 s wait; probe2 @05:43:50
       HEALTH=200.
       ROWLOG_N 926 -> 927 (probe rows:
       05:26:43 and 05:31:51 clean; 05:37:10
       first burst row fb 1026 -> 1045
       kv=4926; 05:42:10 fb=1046 kv=4938;
       all four rows h=200 p=9 500=0 oom=0
       shm=0 gpu=0); ROWLOG_MD5 926-end
       1f0829ead26f3483f6270146c9a435f8 ->
       927-end
       9e871683701fe32ae9abed8e9c2ba3d5;
       handoff baseline 924 superseded by
       later aprows, probe authoritative
       (append-only, NEWBOOTLOG not
       counted).
       Regime: newest engine log still
       engine-20261001-231328.log; LIVELOG
       fd of API PID 100422 resolves
       /mnt/data/shared/models/vllm-glm-5.3-
       flash-nvfp4/logs/
       engine-20261001-231328.log;
       FDMATCH=yes; 19759 lines @05:32:21 ->
       20406 @05:46 census; boot v2-era;
       knob VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_
       SECS=2 environ-proven on pid 100422;
       no new boot through census.
       Fresh wedge burst RESUMED in a
       104-second wave plus two trailing
       one-offs (scripted whole-log +
       fresh-window census; window after
       the ledgered one-off
       a4825dcf5f9794d7 @04:59:30):
       8d3fc8142b88 = 13 SERVERKILLs
       05:35:02 -> 05:35:38 (3-s cadence;
       hit=1024 local=0 trace g0FAE hit=2;
       stale-hit wall=0.2 then 0.1; escape
       (0, False) every cycle; ESCWALL
       release wall=3.0 cold re-arm x12;
       NO refusal follows - the knob=2
       refusal path did not fire even
       inside the loop; last escape
       @05:35:38 then silence through
       census).
       bb6576aa418c = 4 SERVERKILLs
       05:36:38 -> 05:36:46 (hit=137216
       local=0; releases wall=3.0 x3).
       One-offs: 9f243dcb69e0 @05:35:56
       (hit=136192); ae38dac1aea4 @05:36:20
       (hit=106496); a2c6487d8c0a @05:37:14
       (hit=3072); a814c5ca1fdccd03
       @05:44:00 (hit=20480); none drew a
       release line within census (no
       re-hit; wall expiry silent).
       All six burst ids are new
       fingerprints, disjoint from the
       ledger-frozen veteran list.
       Counter deltas whole log:
       SERVERKILL 1026 -> 1047 (+21 = 13
       + 4 + 1x4 fresh, exactly the six
       ids above); rowlog fb 1026 -> 1046
       through the 05:42:10 row (+20)
       with the a814c SK @05:44:00 landing
       on the next row: whole-log +21
       closes exact; escape refusal 27 ->
       27 (+0 whole log, none fresh, none
       in the 13x loop); ESCWALL release
       818 -> 833 (+15 = 12x 8d3f + 3x
       bb6576); KV-LOOKUP hit 1950 ->
       1995 (+45 benign serving lookups,
       doubled-line events as ever;
       fresh local!=0 only the ledgered
       05:02:44 local=1024 and 05:21:16
       local=2048, both inert).
       Frozen vets EXACT none moved:
       fresh +21 is fully attributed to
       the six new ids; large veterans
       re-read unchanged: baf918c29f89=47,
       a55730aa2436=43, b85345868b69=41,
       a44edd70f734=39, 82eccb5ddd9a=35,
       abf4c15de42a=31, 87cf48eadb63=27,
       ad87dfae2667=25, acd4af385c02=21,
       997c5c6e07d2=20, bdb1bd99838f=18,
       a1cb9c778c6c=17; new id 8d3fc8142b88
       at 13 slots above frozen minima.
       Traceback=0, ImportError/
       AssertionError=0, real CUDA OOM /
       GPU watchdog=0 whole log; HTTP 5xx
       =0 (the only 500-token matches are
       SpecDecoding metric lines); the
       only non-200 serving lines are two
       POST 400 Bad Request lines (client
       192.168.2.208); probe rows 500=0
       oom=0 shm=0 gpu=0 all four.
       hb ladder unchanged: last rung
       stays 05:21:43, healed by 05:26:43
       hb=200; rows 05:26:43 -> 05:42:10
       all hb=200, no 2-run window, ALARM
       untouched.
       Serving intact: POST /v1/chat/
       completions 200 OK 1434 whole log
       @05:47 postcheck (+64 vs c459
       census 1370), GET /metrics 200 OK
       4721 (+236 vs 4485); apc 491 ->
       524; last raw log lines GET 200
       OK + POST 200 OK @05:46:32.
       Post-restart incident: none (no
       restart; first post-v3.1
       microdetail stays pending;
       WEDGE-LOOP-POSTV31 not applicable
       under the v2-era boot).
       Standing pre-restart regime: the
       same-id detector is NOT clean this
       cycle: 11 escapes of 8d3fc8142b88
       inside 05:35:02 -> 05:35:32 and 4
       escapes of bb6576aa418c inside 8 s
       = burst trigger, so the c458/c459
       zero-delta QUIET chain breaks.
       Verdict: CYCLE_ALERT (fresh wedge
       burst 8d3fc8142b88x13 +
       bb6576aa418cx4 + four one-offs; SK
       +21 RF +0 SW +15 KVL +45; boot
       still 231328 v2-era, knob=2, API
       pid 100422 alive; staged v3.1
       claim-blockout still
       post-restart-gated).
       Cycle 461 (epoch II; v2-era boot held through the
       whole cycle; user did not restart, no rebase):
       station pid 60644 alive at both probes, no
       restart needed; probe1 @05:52:45 HEALTH=200;
       single 300 s wait; probe2 @05:59:42 HEALTH=200.
       ROWLOG_N 929 -> 930 (probe rows: fresh row
       [05:57:44] h=200 p=9 500=0 oom=0 shm=0 gpu=0
       clean); ROWLOG_MD5 929-end
       f7c4b479b43b4fe2189e646c77354b07 -> 930-end
       d337a20b79d3258c7975c1eee8a47293; c460-end
       baseline 927 (9e871683701fe32ae9abed8e9c2ba3d5)
       superseded by two later aprows, probe
       authoritative (append-only, NEWBOOTLOG never
       appeared). The pre-census row [05:47:31] hb=000
       (fb 1046 -> 1047, the ledgered a814c SK landing
       on the next row) stayed an isolated SERVE-WEDGE
       rung, healed by [05:52:44] hb=200; fresh row
       05:57:44 clean; no 2-run window, ALARM
       untouched.
       Regime: newest engine log still
       engine-20261001-231328.log; LIVELOG fd1/fd2 of
       API PID 100422 both resolve it (FDMATCH=yes by
       readlink); boot still v2-era; knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       environ-proven on pid 100422; no new boot
       through the 06:00 census (log 20938 lines,
       +532).
       Engine census (scripted, whole log): SERVERKILL
       1047 -> 1047 (+0); escape refusal 27 -> 27 (+0);
       ESCWALL release 833 -> 833 (+0); KV-LOOKUP hit
       1995 -> 2010 (+15 benign serving lookups,
       samples local=0 @05:54:23, @05:59:17,
       @05:59:51); Traceback=0,
       ImportError/AssertionError=0, real CUDA OOM /
       GPU watchdog=0 whole log.
       Fresh window 10-02 t>=05:46:32 (after c460's
       last raw line): FRESH_SK=0, FRESH_RF=0,
       FRESH_SW=0, FRESH_KVL=15 -> zero fresh wedge
       machinery this cycle. Last SK still the
       ledgered one-off a814c5ca1fdccd03 @05:44:00;
       c460 burst ids stayed closed/silent:
       8d3fc8142b88=13, bb6576aa418c=4, one-offs
       9f243dcb69e0 / ae38dac1aea4 / a2c6487d8c0a;
       no new top burst id. Last refusal still
       b7485652de1371fe @04:46:17; last SW still
       bb6576aa418c @05:36:46.
       Vets frozen EXACT whole log, none moved (fresh
       +0 fully attributed): baf918c29f89=47,
       a55730aa2436=43, b85345868b69=41,
       a44edd70f734=39, 82eccb5ddd9a=35,
       abf4c15de42a=31, 87cf48eadb63=27,
       ad87dfae2667=25, acd4af385c02=21,
       997c5c6e07d2=20, bdb1bd99838f=18,
       a1cb9c778c6c=17; midscale and wedge-era ids
       hold.
       30-s same-id refusal detector: CLEAN whole log
       (RF_FLAG30_COUNT=0); fresh SK sweep trivially
       clean (0 SK events in the window).
       Serving intact: POST 200 OK 1474 whole log
       (+40 vs 1434), GET /metrics 200 OK 4974 (+253
       vs 4721); apc 533 -> 546; last raw log lines
       GET 200 OK + POST 200 OK @05:59:52; BAD400
       still the 2 ledgered client-400 lines; probe
       rows 500=0 oom=0 shm=0 gpu=0.
       Post-restart incident: none (no restart; first
       post-v3.1 microdetail stays pending;
       WEDGE-LOOP-POSTV31 not applicable under the
       v2-era boot).
       Verdict: QUIET (clean zero-wedge cycle;
       c460's burst trigger did not recur; ladder
       watch-grade solo rung only; boot still
       pre-v3.1; staged v3.1 claim-blockout still
       post-restart-gated).
       Cycle 462 (epoch II; boot stayed 231328 v2-era
       through the whole cycle; user did not restart,
       no rebase): station pid 60644 alive at both
       probes; probe1 @06:02:30 HEALTH=200; single
       300 s wait; probe2 @06:07:35 HEALTH=200.
       ROWLOG_N 930 -> 931 (fresh row [06:02:45]
       h=200 p=9 500=0 oom=0 shm=0 gpu=0 clean);
       ROWLOG_MD5 930-end
       d337a20b79d3258c7975c1eee8a47293 -> 931-end
       44cffa56414269e2f043c56fa6171321 (append-only,
       probe authoritative, NEWBOOTLOG never
       appeared). Regime: newest engine log still
       engine-20261001-231328.log; LIVELOG fd1/fd2
       of API PID 100422 both resolve it; boot
       still v2-era; knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2 carried
       environ-proven on the same pid; no restart.
       Engine census (scripted, whole log):
       SERVERKILL 1047 -> 1047 (+0; fresh sweep
       since 05:46:32 trivially 0; last SK still
       the ledgered one-off a814c5ca1fdccd03
       @05:44:00); escape refusal 27 -> 27 (+0,
       last refusal still b7485652de1371fe
       @04:46:17); ESCWALL release 833 -> 833 (+0,
       last release still bb6576aa418c @05:36:46);
       KV-LOOKUP hit 2010 -> 2015 (+5 benign;
       fresh ts 06:00:09, 06:02:27, 06:07:34,
       06:10:09, 06:10:26, all local=0);
       Traceback=0, ImportError/AssertionError=0,
       real CUDA OOM / GPU watchdog=0 whole log.
       Frozen vets EXACT none moved (delta +0):
       8d3fc8142b88=13, bb6576aa418c=4, one-offs
       9f243dcb69e0 / ae38dac1aea4 /
       a2c6487d8c0a / a814c5ca1fdccd03, SK=1 each;
       long-history SK table re-read unchanged:
       baf918c29f89=47, a55730aa2436=43,
       b85345868b69=41, a44edd70f734=39,
       82eccb5ddd9a=35, abf4c15de42a=31,
       87cf48eadb63=27, ad87dfae2667=25,
       acd4af385c02=21, 997c5c6e07d2=20,
       bdb1bd99838f=18, a1cb9c778c6c=17;
       midscale and wedge-era ids hold; no new
       top burst ids.
       30-s same-id refusal detector (fixed id
       extraction) CLEAN whole log: all 27
       refusals are unique-fingerprint
       singletons (max30s=1 per id); fresh SK
       sweep trivially clean.
       Serving intact: POST 200 OK 1505 whole log
       (+31 vs 1474), GET /metrics 200 OK 4997
       (+23 vs 4974); BAD400 still the 2 ledgered
       client-400 lines; probe rows 500=0 oom=0
       shm=0 gpu=0; last raw lines POST 200 OK at
       the 06:10 tail (serving and lookups live).
       hb ladder: no fresh rungs; c461 rung
       05:47:31 stays healed 05:52:44 hb=200;
       fresh row 06:02:45 clean; no 2-run
       window, ALARM untouched.
       Log lines 20938 -> 21328 (+390; benign
       probe traffic). Post-restart incident: none
       (no restart; first post-v3.1 microdetail
       stays pending; WEDGE-LOOP-POSTV31 not
       applicable under the v2-era boot).
       Verdict: QUIET (clean zero-wedge cycle;
       zero-delta wedges everywhere; KVL +5
       benign; detector CLEAN; ladder healed
       watch-grade; boot still pre-v3.1; staged
       v3.1 claim-blockout still
       post-restart-gated).
       Cycle 463 (epoch II; boot stayed 231328
       v2-era through the whole cycle; user did
       not restart, no rebase): station pid
       60644 alive at both probes, no restart
       needed; probe1 @06:15:45 HEALTH=200;
       single 300 s wait; probe2 @06:22:32
       HEALTH=200.
       ROWLOG_N 933 -> 934 (c462 ledger 931
       superseded by two later aprows
       [06:07:45], [06:12:46], probe
       authoritative); fresh row [06:18:05]
       h=200 p=9 500=0 oom=0 shm=0 gpu=0
       clean; ROWLOG_MD5 933-end
       c5b3429c93fc4c8dfcc441e88022d049 ->
       934-end 516b8f6d8680ed2d5617af6823bf6d1b
       (append-only, NEWBOOTLOG never
       appeared).
       Regime: newest engine log still
       engine-20261001-231328.log; LIVELOG
       fd1/fd2 of API PID 100422 both resolve
       it; boot still v2-era; knobs
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       (+ESCAPE_WALL_SECS=3) environ-proven on
       pid 100422; no restart.
       Fresh wedge burst RESUMED (scripted
       whole-log + fresh-window census; window
       t>=06:10:27 after c462's last raw
       line): new top burst id
       82e1d5343e41c872 = 4 SERVERKILLs
       06:14:33 -> 06:14:42 (4 same-id escapes
       inside 9 s; hit=112640 local=0
       fallbacks=1; escape (0, False) every
       cycle; ESCWALL release wall=3.0 cold
       re-arm x3 @06:14:35/38/41 between runs;
       last escape @06:14:42 then silence).
       4 trailing one-off brand-new ids:
       bb169dd13f243d25 @06:13:33 (hit=3072),
       9cb820ab4bcdb6e1 @06:13:53 (hit=112640),
       8cd9b74db28048af @06:14:13 (hit=118784),
       9a3150ec795436af @06:15:25 (hit=113664);
       112-s wave 06:13:33 -> 06:15:25, all
       large-prefix stale-hit re-escapes
       (local=0), the exact v2 loop shape the
       staged v3.1 claim-blockout targets.
       Counter deltas whole log: SERVERKILL
       1047 -> 1055 (+8 = 4 + 1x4, exactly
       the 5 new ids); ESCWALL release
       833 -> 836 (+3, all 82e1d cold
       re-arms); escape refusal 27 -> 27
       (+0 whole log, none fresh, none
       inside the loop); KV-LOOKUP hit
       2015 -> 2039 (+24 benign serving
       lookups, doubled-line events as
       ever, fresh local=0 all, first
       06:12:27, last 06:22:33); log lines
       21328 -> 21848 (+520). Rowlog fb
       counter corroborates: 1047 -> 1055 on
       the 06:18:05 row (whole-log +8 closes
       exact).
       Frozen vets EXACT none moved: fresh +8
       fully attributed to the 5 new ids;
       long-history table re-read unchanged:
       baf918c29f89=47, a55730aa2436=43,
       b85345868b69=41, a44edd70f734=39,
       82eccb5ddd9a=35, abf4c15de42a=31,
       87cf48eadb63=27, ad87dfae2667=25,
       acd4af385c02=21, 997c5c6e07d2=20,
       bdb1bd99838f=18, a1cb9c778c6c=17;
       c460 burst ids stay closed/silent:
       8d3fc8142b88=13, bb6576aa418c=4,
       one-offs 9f243dcb69e0 / ae38dac1aea4 /
       a2c6487d8c0a / a814c5ca1fdccd03;
       midscale and wedge-era ids hold; no
       veteran moved (fresh SK lines are all
       on the 5 new ids).
       30-s same-id detector: SK side NOT
       clean this cycle (SK_FRESH_MAX30S=4
       within the fresh window:
       82e1d5343e41c872 escapes at
       06:14:33/35/39/42); refusal side CLEAN
       whole log (all 27 refusals
       unique-fingerprint singletons,
       RF_MAX30S=1; last refusal still
       b7485652de1371fe @04:46:17).
       Traceback=0, ImportError/
       AssertionError=0, real CUDA OOM /
       GPU watchdog=0 whole log; HTTP 5xx=0;
       BAD400 still the 2 ledgered client-400
       lines (192.168.2.208).
       Serving intact: POST 200 OK 1555 whole
       log (+50 vs 1505), GET /metrics 200 OK
       5154 (+157 vs 4997); last raw log
       lines POST 200 OK + GET 200 OK
       @06:23:12; probe rows 500=0 oom=0
       shm=0 gpu=0; apc 573 -> 581.
       hb ladder: no fresh rungs; last rung
       stays 05:47:31, healed 05:52:44
       hb=200; fresh rows 05:52:44 ->
       06:18:05 all hb=200, no 2-run window,
       ALARM untouched.
       Post-restart incident: none (no
       restart; first post-v3.1 microdetail
       stays pending; WEDGE-LOOP-POSTV31 not
       applicable under the v2-era boot).
       Standing pre-restart regime: the
       same-id SK detector is NOT clean this
       cycle: 4 escapes of
       82e1d5343e41c872 inside 9 s = burst
       trigger, so the c461/c462 zero-delta
       QUIET chain breaks.
       Verdict: CYCLE_ALERT (fresh wedge
       burst 82e1d5343e41c872x4 + four
       one-offs; SK +8 RF +0 SW +3 KVL +24;
       boot still 231328 v2-era, knob=2, API
       pid 100422 alive; wedge detector and
       hb ladder clean; serving intact;
       staged v3.1 claim-blockout still
       post-restart-gated).
       Cycle 464 (epoch II; boot stayed 231328
       v2-era through the whole cycle; user did
       not restart, no rebase): station pid
       60644 alive at both probes, no restart
       needed; probe1 @06:28:25 HEALTH=200;
       single 300 s wait; probe2 @06:34:30
       HEALTH=200.
       ROWLOG_N 934 -> 937: row 935
       [06:23:16] (landed right after c463's
       06:23:12 last-raw anchor), row 936
       [06:28:25], row 937 [06:33:36]; all
       h=200 p=9 500=0 oom=0 shm=0 gpu=0
       clean. ROWLOG_MD5 934-end
       516b8f6d8680ed2d5617af6823bf6d1b ->
       935-end 29b88eff25baaf48455307273da42074
       -> 937-end bb124a9bc7d67660688364af23a54576
       (append-only, NEWBOOTLOG never appeared).
       Wedge census whole log, scripted x3
       while serving continued: SERVERKILL
       1055 -> 1055 (+0, no fresh escapes,
       all five c463 ids silent); escape
       refusal 27 -> 27 (+0, none fresh);
       ESCWALL release 836 -> 836 (+0, c463
       run stays closed); KV-LOOKUP hit
       2039 -> 2062 (+23 benign serving
       lookups, local=0; none large-prefix,
       last fresh 06:35:35).
       Top burst ids unchanged from c463:
       82e1d5343e41c872=4 escapes
       06:14:33-:42 (3 ESCWALL cold re-arms,
       run closed); one-offs bb169dd13f243d25,
       9cb820ab4bcdb6e1, 8cd9b74db28048af,
       9a3150ec795436af closed; frozen vets
       unmoved.
       30-s same-id detector: SK side CLEAN
       this cycle (zero new SK lines whole
       log); refusal side CLEAN whole log
       (all 27 unique-fingerprint singletons,
       RF_MAX30S=1; last refusal still
       b7485652de1371fe @04:46:17).
       Traceback=0, ImportError/
       AssertionError=0, real CUDA OOM /
       GPU watchdog=0 whole log; HTTP 5xx=0;
       BAD400 still the 2 ledgered client-400
       lines (192.168.2.208; whole-log 4xx
       total=2).
       Serving intact: POST 200 OK 1631
       whole log (+76 vs 1555), GET 200 OK
       5439 (+285 vs 5154); log lines
       21848 -> 22494; last raw lines
       EngineCore 06:35:33 APC-HIT +
       06:35:35 benign KVL then
       POST 200 OK x2; rowlog apc 590 ->
       618, kv 5012 -> 5029, fb flat 1055.
       hb ladder: no fresh rungs; last rung
       stays 05:47:31, healed 05:52:44
       hb=200; fresh rows 06:23:16 ->
       06:33:36 all hb=200, no 2-run
       window, ALARM untouched.
       Post-restart incident: none (no
       restart; first post-v3.1 microdetail
       stays pending; WEDGE-LOOP-POSTV31 not
       applicable under the v2-era boot).
       Regime: newest engine log still
       engine-20261001-231328.log; LIVELOG
       fd1/fd2 of API PID 100422 resolve it;
       boot v2-era; knobs
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       (+ESCAPE_WALL_SECS=3) environ-proven on
       pid 100422; no restart; staged v3.1
       claim-blockout commit fee6cbf stays
       deployed (PARITY_OK md5 da376d25 both
       dests) but post-restart-gated.
       Standing pre-restart regime: zero
       wedge-delta whole log + clean ladder +
       clean detector -> QUIET per standing
       semantics; c463 CYCLE_ALERT stays
       closed within c463.
       Verdict: QUIET (SK +0, burst ids
       silent, RF +0 SW +0 KVL +23 benign;
       boot still 231328 v2-era, knob=2, API
       pid 100422 alive, station pid 60644
       alive; wedge detector and hb ladder
       clean; serving intact; staged v3.1
       claim-blockout still
       post-restart-gated).
       Cycle 465 (epoch II; boot stayed 231328
       v2-era through the whole cycle; user did
       not restart, no rebase): station pid
       60644 alive at both probes, no restart
       needed; probe1 @06:40:45 HEALTH=200;
       single 300 s wait; probe2 @06:46:13
       HEALTH=200.
       ROWLOG_N 937 -> 939: row 938 [06:38:44]
       and row 939 [06:43:45] landed inside
       the cycle window; both h=200 p=9 500=0
       oom=0 shm=0 gpu=0 hb=200 clean.
       ROWLOG_MD5 937-end
       bb124a9bc7d67660688364af23a54576
       (c464) -> 938-end
       3d25a4d46f3f9bacda1765f1381de867
       (probe1) -> 939-end
       36c653f392e7f90a44c36f3c296bf10a
       (probe2) (append-only, NEWBOOTLOG
       never appeared).
       Wedge census whole log, scripted
       evidence pass: SERVERKILL
       1055 -> 1055 (+0, no fresh escapes,
       all five c463 ids silent); escape
       refusal 27 -> 27 (+0, none fresh,
       last refusal still b7485652de1371fe
       @04:46:17 window=2); ESCWALL release
       836 -> 836 (+0, c463 run stays
       closed); KV-LOOKUP hit
       2062 -> 2081 (+19 benign serving
       lookups: 18 local=0, 1 local=2048
       hit=125952 @06:40:42 req
       a440b330a46d1f8a routine local-tier
       load, zero wedge-class lines for
       that id; last fresh 06:46:12
       local=0).
       Top burst ids unchanged from c463:
       82e1d5343e41c872=4 escapes
       06:14:33-:42 (3 ESCWALL cold re-arms,
       run closed); one-offs
       bb169dd13f243d25, 9cb820ab4bcdb6e1,
       8cd9b74db28048af, 9a3150ec795436af
       closed (each sk=1, lastTags
       06:13:33-06:15:25); frozen vets
       unmoved.
       30-s same-id detector: SK side CLEAN
       this cycle (zero new SK lines whole
       log); refusal side CLEAN whole log
       (all 27 unique-fingerprint
       singletons, dup-id-count=0).
       Traceback=0, ImportError/
       AssertionError=0, real CUDA OOM /
       GPU watchdog=0 whole log; HTTP
       5xx=0; HTTP 4xx still the 2
       ledgered client-400 lines
       (192.168.2.208 POST 400 Bad
       Request).
       Serving intact: POST 200 OK 1686
       whole log (+55 vs 1631), GET 200 OK
       5571 (+132 vs 5439); log lines
       22494 -> 22925; last raw lines
       APC-HIT + POST 200 OK x2 + GET
       /metrics 200 OK; rowlog apc 618 ->
       646 -> 659, kv 5029 -> 5038 ->
       5053, fb flat 1055.
       hb ladder: no fresh rungs; last rung
       stays 05:47:31, healed 05:52:44
       hb=200; fresh rows 06:38:44 ->
       06:43:45 all hb=200, no 2-run
       window, ALARM untouched.
       Post-restart incident: none (no
       restart; first post-v3.1 microdetail
       stays pending; WEDGE-LOOP-POSTV31
       not applicable under the v2-era
       boot).
       Regime: newest engine log still
       engine-20261001-231328.log; LIVELOG
       fd of API PID 100422 resolves it;
       boot v2-era; knobs
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       (+ESCAPE_WALL_SECS=3) environ-proven
       on pid 100422; no restart; no
       NEWBOOTLOG in the logs dir; staged
       v3.1 claim-blockout commit fee6cbf
       stays deployed (PARITY_OK md5
       da376d25 both dests) but
       post-restart-gated.
       Standing pre-restart regime: zero
       wedge-delta whole log + clean
       ladder + clean detector -> QUIET
       per standing semantics; c463
       CYCLE_ALERT stays closed within
       c463.
       Verdict: QUIET (SK +0, burst ids
       silent, RF +0 SW +0 KVL +19 benign;
       boot still 231328 v2-era, knob=2,
       API pid 100422 alive, station pid
       60644 alive; wedge detector and hb
       ladder clean; serving intact;
       staged v3.1 claim-blockout still
       post-restart-gated).
       Cycle 466 (epoch II; boot stayed 231328
       v2-era through the whole cycle; user did
       not restart, no rebase): station pid
       60644 alive at both probes, no restart
       needed; probe1 @06:50:06 HEALTH=200;
       single 300 s wait; probe2 @06:55:13
       HEALTH=200.
       ROWLOG_N 939 -> 940 -> 941: row 940
       [06:48:46] and row 941 [06:53:46]
       landed inside the cycle window; both
       h=200 p=9 500=0 oom=0 shm=0 gpu=0
       hb=200 clean.
       ROWLOG_MD5 939-end
       36c653f392e7f90a44c36f3c296bf10a
       (c465 probe2) -> 940-end
       9f507eabc89fc65bcbc6e606ec43a712
       (probe1) -> 941-end
       4d6b5140877a5d47c482286e99a44b48
       (probe2) (append-only, NEWBOOTLOG
       never appeared).
       Wedge census whole log, scripted
       evidence pass: SERVERKILL
       1055 -> 1055 (+0, no fresh escapes,
       all five c463 burst ids silent);
       escape refusal 27 -> 27 (+0, none
       fresh, last refusal still
       b7485652de1371fe @04:46:17);
       ESCWALL release 836 -> 836 (+0,
       c463 run stays closed); KV-LOOKUP
       hit 2081 -> 2083 (+2 benign routine
       local=0 hit=1024 loads @06:49:03
       and @06:50:04, zero wedge-class
       lines; running-total form restored
       per c466 dispatch: prior hard total
       2062 at c464, c465 +19 -> 2081,
       live normalized).
       Top burst ids unchanged from c463:
       82e1d5343e41c872 (run closed, sk=4,
       total ref-lines 16) and one-offs
       bb169dd13f243d25,
       9cb820ab4bcdb6e1, 8cd9b74db28048af,
       9a3150ec795436af (each sk=1,
       ref-lines 4); zero new id lines
       this cycle; last SK 06:15, last
       refusal 04:46, last ESCWALL
       release 06:14, all pre-window.
       30-s same-id detector: SK side
       CLEAN (zero new SK lines whole
       log); refusal side CLEAN whole log
       (27 unique-fingerprint singletons,
       dup-id-count=0).
       Traceback=0, ImportError/
       AssertionError=0, real CUDA OOM /
       GPU watchdog=0 whole log; HTTP
       5xx=0; HTTP 4xx still the 2
       ledgered client-400 lines.
       Serving intact: POST 200 OK 1686 ->
       1714 (+28); GET /metrics 200 OK
       5542 (narrower /metrics-only
       pattern, not comparable to the
       c465 all-GET figure); log lines
       22925 -> 23180; engine log md5
       prefix ffebfac2; last raw lines
       APC-HIT + POST 200 OK x2 + GET
       /metrics 200 OK; rowlog apc 659 ->
       670 -> 679, kv 5053 -> 5054 ->
       5056, fb flat 1055.
       hb ladder: no fresh rungs; last
       rung stays 05:47:31, healed
       05:52:44 hb=200; fresh rows
       06:48:46 -> 06:53:46 all hb=200,
       no 2-run window, ALARM untouched.
       Post-restart incident: none (no
       restart; first post-v3.1
       microdetail stays pending;
       WEDGE-LOOP-POSTV31 not applicable
       under the v2-era boot).
       Regime: newest engine log still
       engine-20261001-231328.log; LIVELOG
       fd of API PID 100422 resolves it;
       boot v2-era; knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       environ-proven on pid 100422
       (WALL_SECS=3 ambient per ledger);
       no restart; no NEWBOOTLOG in the
       logs dir; staged v3.1
       claim-blockout commit fee6cbf stays
       deployed (PARITY_OK md5 da376d25
       both dests) but post-restart-gated.
       Standing pre-restart regime: zero
       wedge-delta whole log + clean
       ladder + clean detector -> QUIET
       per standing semantics; zero-delta
       cycle keeps the standing verdict;
       c463 CYCLE_ALERT stays closed.
       Verdict: QUIET (SK +0, burst ids
       silent, RF +0 SW +0 KVL +2 benign
       running 2083; boot still 231328
       v2-era, knob=2, API pid 100422
       alive, station pid 60644 alive;
       wedge detector and hb ladder
       clean; serving intact; staged v3.1
       claim-blockout still
       post-restart-gated).
       Cycle 467 (epoch II; boot stayed 231328
       v2-era through the whole cycle; user did
       not restart, no rebase): station pid
       60644 alive at both probes, no restart
       needed; probe1 @06:57:48 HEALTH=200;
       single 300 s wait; probe2 @07:04:17
       HEALTH=200.
       ROWLOG_N 941 -> 943 (probe1 held the
       c466-end 941 anchor): row 942
       [06:58:47] and row 943 [07:03:47]
       landed inside the cycle window; both
       h=200 p=9 500=0 oom=0 shm=0 gpu=0
       hb=200 clean.
       ROWLOG_MD5 941-end
       4d6b5140877a5d47c482286e99a44b48
       (c466 probe2 = c467 probe1 anchor,
       byte-identical re-read) -> 943-end
       a65720f8edcff35aa549de50bde65a3f
       (probe2; rows 942+943 landed between
       probes; append-only chain intact).
       Wedge census whole log, scripted
       evidence pass: SERVERKILL
       1055 -> 1055 (+0, no fresh escapes,
       all five c463 burst ids silent);
       escape refusal 27 -> 27 (+0, none
       fresh, last refusal still
       b7485652de1371fe @04:46:17);
       ESCWALL release 836 -> 836 (+0,
       c463 run stays closed, releases
       frozen at 06:14 pre-window);
       KV-LOOKUP hit 2083 -> 2085 (+2
       benign local=0 loads @06:56:44
       hit=2048 req b118e02c3ee9b9b9
       and @06:57:46 hit=1024 req
       88f994fce6f2dbac; zero wedge-class
       lines for either id; running
       total continues).
       Top burst ids unchanged from
       c463: 82e1d5343e41c872 (run
       closed, sk=4) and one-offs
       bb169dd13f243d25,
       9cb820ab4bcdb6e1,
       8cd9b74db28048af,
       9a3150ec795436af (each sk=1);
       zero new id lines this cycle;
       last SK 06:15, last refusal
       04:46, last ESCWALL release
       06:14, all pre-window.
       30-s same-id detector: SK side
       CLEAN (zero new SK lines whole
       log); refusal side CLEAN whole
       log (27 unique-fingerprint
       singletons, dup-id-count=0).
       Traceback=0, ImportError/
       AssertionError=0, real CUDA
       OOM / GPU watchdog=0 whole
       log; HTTP 5xx=0; HTTP 4xx
       still the 2 ledgered
       client-400 lines.
       Serving intact: POST 200 OK
       1714 -> 1734 (+20); GET
       /metrics 200 OK 5542 -> 5642
       (+100); log lines 23180 ->
       23393; engine log md5 prefix
       b77b6bd1; last raw lines GET
       /metrics + POST + GET /metrics
       + GET /health all 200 OK;
       rowlog apc 679 -> 701 -> 701,
       kv 5056 -> 5058 -> 5058, fb
       flat 1055.
       hb ladder: no fresh rungs;
       last rung stays 05:47:31,
       healed 05:52:44 hb=200; fresh
       rows 06:58:47 -> 07:03:47 all
       hb=200, no 2-run window,
       ALARM untouched.
       Post-restart incident: none (no
       restart; first post-v3.1
       microdetail stays pending;
       WEDGE-LOOP-POSTV31 not
       applicable under the v2-era
       boot).
       Regime: newest engine log is
       still engine-20261001-231328.log
       (LIVELOG fd of API PID 100422
       resolves it); boot v2-era; knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       environ-proven on pid 100422
       (WALL_SECS=3 ambient per
       ledger); no restart; no
       NEWBOOTLOG in the logs dir;
       staged v3.1 claim-blockout
       commit fee6cbf stays deployed
       (PARITY_OK md5 da376d25 both
       dests) but post-restart-gated.
       Standing pre-restart regime:
       zero wedge-delta whole log +
       clean ladder + clean detector
       -> QUIET per standing
       semantics; zero-delta cycle
       keeps the standing verdict;
       c463 CYCLE_ALERT stays closed.
       Verdict: QUIET (SK +0, burst
       ids silent, RF +0 SW +0 KVL +2
       benign running 2085; boot
       still 231328 v2-era, knob=2,
       API pid 100422 alive, station
       pid 60644 alive; wedge
       detector and hb ladder clean;
       serving intact; staged v3.1
       claim-blockout still
       post-restart-gated).
       Cycle 468 (epoch II; boot stayed 231328
       v2-era through the whole cycle; user did
       not restart, no rebase): station pid
       60644 alive at both probes (ALIVE_60644
       both), no restart needed; probe1
       @07:07:46 HEALTH=200; single 300 s
       wait; probe2 @07:13:14 HEALTH=200.
       ROWLOG_N 943 -> 944 (probe1 held the
       c467-end 943 anchor): row 944
       [07:09:02] landed inside the cycle
       window; h=200 p=9 500=0 oom=0 shm=0
       gpu=0 hb=200 clean.
       ROWLOG_MD5 943-end
       a65720f8edcff35aa549de50bde65a3f
       (c467 probe2 = c468 probe1 anchor,
       byte-identical re-read) -> 944-end
       0c2c96d110dbf528b337700c2a5ef9d2
       (probe2; row 944 landed between
       probes; append-only chain intact).
       Wedge census whole log, scripted
       evidence pass (3 passes: census,
       per-id, serving; lens: per-id
       sk = SERVERKILL-line count,
       chatcmpl-anchored): SERVERKILL
       1055 -> 1064 (+9, five fresh runs,
       every fresh id silent since
       07:07:46 through the scan);
       escape refusal 27 -> 27 (+0, none
       fresh, last refusal still
       b7485652de1371fe @04:46:17);
       ESCWALL release 836 -> 840 (+4:
       b3ccc741 release @07:06:05 and
       aae331b9 releases @07:07:16/
       19/22, all local=0 wall=3.0 no
       boundary change, cold re-arm);
       KV-LOOKUP hit 2085 -> 2113 (+28
       benign local=0 loads; running
       total continues; tail sample:
       ba26a0cd7750cd58 hit=1024
       @07:12:31, bc70962064a0c456
       hit=1024 @07:13:12,
       ba895349e654fcca hit=162816
       @07:14:01, b4e6bf809e0f2736
       hit=9216 @07:14:27,
       8b21cfd501ea3adb hit=1024
       @07:14:32, 93160224bf5c98c9
       hit=20480 @07:14:59).
       Frozen vet per-id sk counts all
       EXACT vs the abridged handoff
       list (every listed id matches);
       additive day-1 fingerprint
       ac6b62d8a85ce6f4 (sk=17, not in
       the abridged list) whole-log
       stable, zero fresh lines this
       cycle; c463 five-id burst group
       re-verified inside the 1055
       baseline: bb169dd13f243d25,
       9cb820ab4bcdb6e1,
       8cd9b74db28048af,
       9a3150ec795436af sk=1 each,
       82e1d5343e41c872 sk=4 (run
       still closed).
       Fresh SK frames this cycle, all
       inside 07:06:02-07:07:46, every
       fresh id silent ever since
       (through the pass-3 scan):
       aae331b9fe417b4f sk=4 @07:07:13/
       16/19/22 hit=113664 fallbacks=1
       (3 s cadence, local=0 per
       releases; 4 sk / 3 sw frame
       shape identical to the c463
       82e1d5343e41c872 run
       @06:14:33-42);
       b3ccc741ef138d2b sk=2 @07:06:02/
       05 (3 s cadence, hit=1024, 1
       release @07:06:05);
       one-offs: 926c00d4519e406e
       @07:06:30 hit=112640,
       ae2bb80c0891bc31 @07:06:50
       hit=138240, ac51e7e455cd37af
       @07:07:46 hit=113664 (each
       sk=1, no release logged, no
       fresh refusal for any fresh
       id; all closed).
       30-s same-id detector: refusal
       side CLEAN whole log (27
       unique-fingerprint singletons,
       dup-id-count=0); SK side shows
       fresh same-id re-escape runs on
       two ids at 3-4 s cadence with
       wall=3.0 releases and no
       boundary change (cold re-arm) =
       the standard pre-restart v2-era
       escape/wall micro-frame shape
       also seen across the frozen
       vets (e.g. a44edd70f7347af3
       day-1 23:26 run); pre-restart
       wedge-burst cycles are expected
       frames, not serve wedges;
       WEDGE-LOOP-POSTV31 not
       applicable under the v2-era
       boot.
       Traceback=0, ImportError/
       AssertionError=0, real CUDA
       OOM / GPU watchdog=0 whole
       log; HTTP 5xx=0; HTTP 4xx
       still the 2 ledgered
       client-400 lines.
       Serving intact: POST 200 OK
       1734 -> 1785 (+51); GET
       /metrics 200 OK 5642 -> 5833
       (+191); log lines 23393 ->
       24175 (pass-3 scan, log still
       growing); engine log md5
       prefix 559b8413; last raw
       lines GET /metrics + POST
       /v1/chat/completions both
       200 OK; rowlog apc 701 -> 710,
       kv 5058 -> 5100, fb 1055 ->
       1064 (fb +9 = fresh SK +9).
       hb ladder: no fresh rungs;
       last rung stays 05:47:31,
       healed 05:52:44 hb=200; fresh
       row 07:09:02 hb=200, no 2-run
       window, ALARM untouched.
       Post-restart incident: none (no
       restart; first post-v3.1
       microdetail stays pending).
       Regime: newest engine log is
       still engine-20261001-231328.log
       (LIVELOG fd of API PID 100422
       resolves it; newest-mtime too
       in the logs dir); boot v2-era;
       knob VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       environ-proven on pid 100422
       (ESCAPE_WALL_SECS=3 ambient,
       environ and ledger agree);
       no restart; no NEWBOOTLOG in
       the logs dir; staged v3.1
       claim-blockout commit fee6cbf
       stays deployed (PARITY_OK md5
       da376d25 both dests) but
       post-restart-gated.
       Standing pre-restart regime:
       fresh closed one-off/short
       frames counted (+9 SK); zero
       new serve wedges this cycle
       (no fresh rowlog ALERT,
       refusal side silent, detector
       and hb ladder clean, serving
       200); pre-restart wedge-burst
       cycles are expected frames;
       standing QUIET verdict
       persists; c463 CYCLE_ALERT
       stays closed.
       Verdict: QUIET (SK +9, all five
       fresh frames closed inside the
       window: aae331b9 x4 run,
       b3ccc741 x2 run, three
       one-offs each sk=1; RF +0,
       SW +4, KVL +28 benign running
       2113; boot still 231328
       v2-era, knob=2, API pid
       100422 alive, station pid
       60644 alive; wedge detector
       and hb ladder clean; serving
       intact; staged v3.1
       claim-blockout still
       post-restart-gated).
       Cycle 469 (epoch II; boot stayed 231328 v2-era
       through the whole cycle; user did not restart,
       no rebase): station pid 60644 alive at both
       probes (ALIVE_60644 both), no restart needed;
       probe1 @07:29:56 HEALTH=200; single 300 s
       wait; probe2 @07:36:27 HEALTH=200.
       ROWLOG_N 947 -> 949 (probe1 held 947 = the
       c468-end 944 anchor plus 3 fresh wedge rows
       it caught 07:14:22/07:19:43/07:25:03, all
       ALERT SERVE-WEDGE hb=000); probe2 landed
       rows 948/949 @07:30:19/07:35:27 hb=200
       clean (detector CLEAN at cycle close).
       ROWLOG_MD5 944-end
       0c2c96d110dbf528b337700c2a5ef9d2 (c468
       close probe2 anchor, per handoff baselines)
       -> 947-end efcceea725f2f257e959c652ae2c548b
       (probe1; 945-947 = the 3 wedge rows) ->
       949-end 966ceccee693a54bbefa439cfeafce7d
       (probe2; append-only chain intact).
       Wedge census vs c468 baselines (live log
       only, matches ledger scope; scripted
       evidence pass + fresh-window mini-pass):
       SERVERKILL 1064 -> 1137 (+73 fresh, 19
       fresh ids, window 07:16:55-07:28:27;
       cross-check rowlog fb 1064 -> 1137 = +73
       exact); escape refusal 27 -> 27 (+0 none
       fresh, last refusal still
       b7485652de1371fe @04:46:17); ESCWALL
       release 840 -> 894 (+54, all fresh-window,
       wall=3.0, local=0, no boundary change,
       cold re-arm); KV-LOOKUP hit 2113 -> 2216
       (+103 benign local=0 loads; tail
       a233f963c777f3ab hit=2048 @07:35:15).
       Fresh SK frames: EngineCore pid 100982,
       fallbacks=1, wall=0.2-0.3, ESCWALL
       release lines at wall=3.0, every fresh
       id silent since 07:28:27 through the
       scan: 97e295a1f5706694 sk=16
       @07:18:33-19:18 and ac387a2b14ed9135
       sk=15 @07:17:05-17:47 (both ~3 s
       cadence); 99c07e0bfceab0b3 sk=9
       @07:26:29-26:53; a90544c89bc6e91a sk=5
       @07:23:04-16; 86c4c45ff9925950 sk=4
       @07:28:18-27 and b84b9c85d61fc447 sk=4
       @07:22:07-16; a6451f36682fd834 and
       a692611ae71bfc12 sk=3; 81c7e27ce5f74bae,
       b4537f00c3128a83, b641e1b3b989b9f2
       sk=2; one-offs sk=1 x8: 8d1b37dd6a9f75be,
       b8eee81d244a8c77, 8d35f9d8b421a30f,
       8c344f8dd5e97b4e, 9332adbb020c77ac,
       814a5e95e01f0e7a, 9af34cdc7590bbb9,
       b4872e5aaab58924; hit 1024-188416.
       30-s same-id lens: SK side shows the
       standard pre-restart v2-era escape/wall
       micro-frame (same-id re-escape runs at
       ~3-5 s with wall=3.0 releases, no
       boundary change, cold re-arm) now at
       wave scale; refusal side CLEAN (+0
       fresh, prior singletons unchanged);
       WEDGE-LOOP-POSTV31 not applicable under
       the v2-era boot.
       hb ladder: fresh rungs - three
       consecutive hb=000 rows
       07:14:22/07:19:43/07:25:03 ALERT
       SERVE-WEDGE sustained by the burst
       (latch: 3+ consecutive hb=000 rungs =
       CYCLE_ALERT trigger); heal: fresh rows
       07:30:19/07:35:27 hb=200 clean;
       detector CLEAN at close; prior last
       rung stays 05:47:31 healed 05:52:44
       hb=200; ALARM untouched at the 07:09:02
       close of c468.
       Traceback=0, ImportError/
       AssertionError=0, real CUDA OOM / GPU
       watchdog=0 (live-log fresh and whole-live
       counts zero; archived 09-15 frames carry
       the ledgered historical Traceback agg=
       651 / OOM agg=469 with zero fresh
       lines); rowlog 500=0 on every row (5xx
       clean on all watched rows).
       Serving intact: POST 200 OK 1785 -> 1865
       (+80); GET /metrics 200 OK 5833 -> 6045
       (+212); log still growing (mtime
       07:38:01, last raw POST 200 OK + GET
       /metrics 200 OK @07:36:33); rowlog apc
       710 -> 762, kv 5100 -> 5411, fb 1064 ->
       1137 (fb +73 = fresh SK +73 exact).
       Regime: newest engine log is still
       engine-20261001-231328.log (newest-mtime
       07:38:01 and LIVELOG fd of API PID
       100422 both resolve it); boot v2-era;
       knob VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_
       SECS=2 environ-proven on pid 100422
       (ESCAPE_WALL_SECS=3 ambient, environ
       and ledger agree); no restart; no
       NEWBOOTLOG in the logs dir; EngineCore
       child pid 100982 observed inside the
       boot window; staged v3.1 claim-blockout
       commit fee6cbf stays deployed
       (PARITY_OK md5 da376d25 both dests) but
       post-restart-gated.
       Post-restart incident: none (no
       restart; first post-v3.1 microdetail
       stays pending).
       Standing pre-restart regime: this is a
       delta cycle - a fresh 73-sk/3-sw burst
       wave (19 new ids, all fresh
       fingerprints, none of them ledgered
       vets, all closed since 07:28:27)
       carried 3 consecutive wedged probe
       rows; this is beyond the one-off/short-
       of-window frame allowance, so the
       ladder/burst trigger governs: 3+
       consecutive hb=000 rungs = CYCLE_ALERT
       (heal with detector CLEAN at close does
       not erase the ladder); c463-class run
       at larger scale; zero-delta standing
       verdict rule not invoked; c463
       CYCLE_ALERT grouping now has a larger
       analog; post-restart verdict still
       requires the zero wedge-loop gate.
       Verdict: CYCLE_ALERT (SK 1064 -> 1137,
       +73 fresh across 19 new ids
       07:16:55-07:28:27, every fresh id
       silent since 07:28:27; 3 consecutive
       hb=000 rows 07:14:22/07:19:43/07:25:03,
       then heals hb=200 @07:30:19/07:35:27
       with detector CLEAN at close; RF +0,
       SW +54, KVL +103 benign running 2216;
       refusal last b7485652de1371fe
       @04:46:17; boot 231328 v2-era whole
       cycle, knob=2/WALL=3, API pid 100422 +
       EngineCore pid 100982 alive, station
       pid 60644 alive; serving intact 200;
       staged v3.1 claim-blockout still
       post-restart-gated).
       Cycle 470 (one-shot): station pid 60644
       ALIVE at both probes (07:44:07 probe1,
       07:53:22 probe2); HEALTH=200 at both
       closes; no station restart; ROWLOG_N
       950 -> 952, ROWLOG_MD5 03e56a66 ->
       20a11aa8 (append-only; no NEWBOOTLOG).
       Regime: newest engine log is still
       engine-20261001-231328.log (newest-
       mtime and LIVELOG fd of API pid 100422
       both resolve it); no restart - boot
       v2-era whole cycle; API pid 100422
       alive (pgrep exact), EngineCore pid
       100982 alive; knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       and VLLM_KV_OFFLOAD_ESCAPE_WALL_SECS=3
       environ-proven on pid 100422; staged
       v3.1 claim-blockout (commit fee6cbf,
       md5 da376d25 both dests PARITY_OK)
       still deployed, post-restart-gated.
       Engine counters: SK 1137 -> 1156
       (+19), RF 27 (+0; last
       b7485652de1371fe @04:46:17 unchanged),
       SW 894 -> 901 (+7 wall=3.0 cold
       re-arm), KVL 2216 -> 2257 (+41 benign
       local=0); Traceback 0,
       ImportError/AssertionError 0,
       CUDA-OOM 0, GPU-watchdog 0.
       Wedge window: 07:40:39 -> 07:48:10,
       +19 SERVERKILL across 12 fresh ids
       (b0ab5778e3c64163=4x,
       be211edd340813fb=3x, 871ec0115a65c7ad=3x,
       9 singletons); ledger vets frozen
       EXACT (baf918c29f8971ac=47,
       a55730aa2436f326=43, b85345868b69b2db=41,
       a44edd70f7347af3=39, 82eccb5ddd9a76bd=35,
       abf4c15de42aa3fb=31, 97e295a1f5706694=16,
       ac387a2b14ed9135=15, 99c07e0bfceab0b3=9,
       ...); every fresh id silent since
       07:48:10 (probe2 07:53:22, CLEAN).
       Rowlog rows this cycle: [07:40:29]
       hb=200 carry-over row, [07:45:49]
       hb=000 one isolated rung ALERT
       SERVE-WEDGE, [07:50:50] hb=200 heal;
       2-consecutive rungs never met;
       apc 798 -> 804, kv 5413 -> 5497,
       fb 1137 -> 1156 (fb +19 = SK +19
       exact); tier-full churn ev 147280 ->
       152352.
       Serving intact: POST 200 OK, GET
       /metrics 200 OK, GET /health 200 OK
       through 07:53:14 (log tail still
       growing).
       Post-restart incident: none (no
       restart; first post-v3.1 microdetail
       stays pending).
       Standing pre-restart regime: wedge
       activity present but inside the
       one-off/short-of-window frame
       allowance - ~8 min window, 12 fresh
       small-count ids (max 4x same id at
       the wall=3.0 re-arm cadence), single
       hb=000 rung, healed, detector CLEAN
       at cycle close; ladder trigger (2+
       consecutive rungs) not reached.
       Verdict: QUIET (SK 1137 -> 1156 +19
       short-window ids silent since
       07:48:10; isolated hb=000 rung
       @07:45:49 healed @07:50:50; RF +0,
       SW +7 re-arm, KVL +41 benign
       local=0; boot 231328 v2-era whole
       cycle knob=2/WALL=3, API pid 100422
       + EngineCore 100982 alive, station
       pid 60644 alive; serving 200 intact;
       staged v3.1 claim-blockout still
       post-restart-gated).
       Cycle 471 (one-shot): station pid 60644 ALIVE
       at both probes (08:00:06 probe1, 08:07:14
       probe2); HEALTH=200 at both closes; no
       station restart; ROWLOG_N 953 -> 955,
       ROWLOG_MD5 6cb41e49 -> a99f0a19
       (append-only; no NEWBOOTLOG).
       Regime: newest engine log is still
       engine-20261001-231328.log (newest-mtime
       and LIVELOG fd of API pid 100422 both
       resolve it); no restart - boot v2-era
       whole cycle; API pid 100422 alive (pgrep
       exact), EngineCore pid 100982 alive; knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2 and
       VLLM_KV_OFFLOAD_ESCAPE_WALL_SECS=3
       environ-proven on pid 100422; staged v3.1
       claim-blockout (commit fee6cbf, md5
       da376d25 both dests PARITY_OK) still
       deployed, post-restart-gated.
       Engine counters: SK 1156 -> 1173 (+17),
       RF 27 (+0; last b7485652de1371fe @04:46:17
       unchanged), SW 901 -> 909 (+8 wall=3.0
       cold re-arm), KVL 2257 -> 2291 (+34 benign
       local=0); Traceback 0,
       ImportError/AssertionError 0, CUDA-OOM 0,
       GPU-watchdog 0.
       Wedge window: 07:54:41 -> 08:00:31, +17
       SERVERKILL across 9 fresh ids
       (bdc371c76360cc58=6x,
       8cb8410b855f8a11=3x,
       a3a4961a89a860b4=2x, 6 singletons); every
       fresh id silent since 08:00:31 (probe2
       08:07:14; log tail 08:07:25, CLEAN).
       Rowlog rows this cycle: [07:50:50] hb=200
       carry-over row, [07:56:10] hb=000 one
       isolated rung ALERT SERVE-WEDGE,
       [08:01:13] hb=200 heal, [08:06:15] hb=200;
       2-consecutive rungs never met; apc 804 ->
       813, kv 5497 -> 5573, fb 1156 -> 1173
       (fb +17 = SK +17 exact); tier-full churn
       ev 152352 -> 159632.
       Serving intact: POST 200 OK @08:07:23 and
       08:07:24 (engine tail); rowlog h=200 rows
       [07:50:50] -> [08:06:15]; log tail still
       growing (mtime 08:07:25).
       Post-restart incident: none (no restart;
       first post-v3.1 microdetail stays
       pending).
       Standing pre-restart regime: wedge
       activity present but inside the
       one-off/short-of-window frame allowance -
       ~6 min window, 9 fresh small-hit-count
       ids (max 6x same id bdc371c76360cc58 at
       the wall=3.0 re-arm cadence), single
       hb=000 rung, healed, detector CLEAN at
       cycle close; ladder trigger (2+
       consecutive rungs) not reached.
       Verdict: QUIET (SK 1156 -> 1173 +17
       short-window ids silent since 08:00:31;
       isolated hb=000 rung @07:56:10 healed
       @08:01:13; RF +0, SW +8 re-arm, KVL +34
       benign local=0; boot 231328 v2-era whole
       cycle knob=2/WALL=3, API pid 100422 +
       EngineCore 100982 alive, station pid
       60644 alive; serving 200 intact; staged
       v3.1 claim-blockout still
       post-restart-gated).
       Cycle 472 (one-shot): station pid 60644 ALIVE
       at both probes (08:10:12 probe1, 08:15:32
       probe2); HEALTH=200 at both closes; no
       station restart; ROWLOG_N 955 -> 957,
       ROWLOG_MD5 a99f0a19 -> 919f3288
       (midcycle 373df838 @probe2; append-only,
       no NEWBOOTLOG).
       Regime: newest engine log is still
       engine-20261001-231328.log (LIVELOG fd
       of API pid 100422 resolves it, size
       6624982, mtime 08:17:16 growing); no
       restart - boot v2-era whole cycle; API
       pid 100422 alive (pgrep exact, etime
       09:03:48), EngineCore pid 100982 alive;
       knob VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       and VLLM_KV_OFFLOAD_ESCAPE_WALL_SECS=3
       environ-proven on pid 100422; staged
       v3.1 claim-blockout (commit fee6cbf, md5
       da376d25 both dests PARITY_OK) still
       deployed, post-restart-gated.
       Engine counters: SK 1173 -> 1174 (+1),
       RF 27 (+0; last b7485652de1371fe
       @04:46:17 unchanged), SW 909 (+0; no
       release after the 08:00:31 re-arm
       batch), KVL 2291 -> 2308 (+17 benign
       local=0 17/17); Traceback 0,
       ImportError/AssertionError 0, CUDA-OOM
       0, GPU-watchdog 0, 5xx 0.
       Wedge events: no window this cycle -
       one fresh id 86144233f44a2cba single
       SERVERKILL @08:10:13 (hit=68608
       fallbacks=1 wall=0.2), silent since;
       c471 window ids all silent since
       08:00:31 (bdc371c76360cc58=6x,
       8cb8410b855f8a11=3x,
       a3a4961a89a860b4=2x, 6 singletons).
       Rowlog rows this cycle: [08:11:15]
       h=200 hb=200, [08:16:26] h=200 hb=200;
       no new 000 rungs (c471 single rung
       @07:56:10 stays healed @08:01:13);
       apc 813 -> 863, kv 5573 -> 5592,
       fb 1173 -> 1174 (fb +1 = SK +1 exact);
       tier-full churn ev 159632 -> 162632.
       Serving intact: POST 200 OK x2
       @08:17:13 (engine tail); rowlog h=200
       rows [08:11:15] -> [08:16:26]; log tail
       still growing.
       Post-restart incident: none (no restart;
       first post-v3.1 microdetail stays
       pending).
       Standing pre-restart regime: single
       one-off fresh id escape (1 event, no
       repeat, silent), detector CLEAN at
       cycle close, no 000 rungs this cycle;
       ladder trigger (2+ consecutive rungs)
       not reached.
       Verdict: QUIET (SK 1173 -> 1174 +1 one
       fresh id silent since 08:10:13; RF +0,
       SW +0, KVL +17 benign local=0; boot
       231328 v2-era whole cycle knob=2/WALL=3,
       API pid 100422 + EngineCore 100982
       alive, station pid 60644 alive; serving
       200 intact; staged v3.1 claim-blockout
       still post-restart-gated).
       Cycle 473 (one-shot): station pid 60644 ALIVE
       at both probes (08:19:28 probe1, 08:26:59
       probe2); HEALTH=200 at both closes; no
       station restart; ROWLOG_N 957 -> 959,
       ROWLOG_MD5 919f3288 -> df427800
       (append-only, no NEWBOOTLOG).
       Regime: newest engine log is still
       engine-20261001-231328.log (size 6673490,
       mtime 08:21:27 growing); no restart -
       boot v2-era whole cycle; API pid 100422
       alive (etime 09:13:29 at 08:26:59) and
       EngineCore pid 100982 active in engine
       lines; knob VLLM_KV_OFFLOAD_ESCAPE_
       REFUSAL_SECS=2 and VLLM_KV_OFFLOAD_ESCAPE_
       WALL_SECS=3 environ-proven on pid 100422;
       staged v3.1 claim-blockout (commit
       fee6cbf, md5 da376d25 both dests
       PARITY_OK) still deployed,
       post-restart-gated.
       Engine counters: SK 1174 (+0; last
       86144233f44a2cba @08:10:13 silent
       since), RF 27 (+0; last b7485652de1371fe
       @04:46:17 unchanged), SW 909 (+0; no
       release since 07:59:22 re-arm batch),
       KVL 2308 -> 2311 (+3 benign local=0
       3/3); Traceback 0,
       ImportError/AssertionError 0, CUDA-OOM 0,
       GPU-watchdog 0, 5xx 0.
       Wedge events: no window this cycle -
       zero fresh SERVERKILL lines, zero
       refusal lines, zero releases; c471
       window ids and 86144233f44a2cba all
       silent.
       Rowlog rows this cycle: [08:21:27]
       h=200 hb=200, [08:26:28] h=200 hb=200;
       no new 000 rungs (c471 single rung
       @07:56:10 stays healed @08:01:13);
       apc 863 -> 919, kv 5592 -> 5601,
       fb 1174 -> 1174 (fb +0 = SK +0 exact);
       tier-full churn ev 162632 -> 167716.
       Serving intact: POST 200 OK @08:21:28
       (engine tail); rowlog h=200 rows
       [08:21:27] -> [08:26:28]; log tail
       still growing.
       Post-restart incident: none (no restart;
       first post-v3.1 microdetail stays
       pending).
       Standing pre-restart regime: zero wedge
       activity this cycle, detector CLEAN at
       cycle close; ladder trigger (2+
       consecutive rungs) not reached.
       Verdict: QUIET (SK 1174 +0 nothing new
       since 08:10:13; RF +0, SW +0, KVL +3
       benign local=0; boot 231328 v2-era whole
       cycle knob=2/WALL=3, API pid 100422 +
       EngineCore 100982 alive, station pid
       60644 alive; serving 200 intact; staged
       v3.1 claim-blockout still
       post-restart-gated).
       Cycle 474 (one-shot): station pid 60644 ALIVE
       at both probes (08:30:49 probe1, 08:38:06
       probe2); HEALTH=200 at both closes; no
       station restart; ROWLOG_N 959 -> 961,
       ROWLOG_MD5 df427800 -> d1c61706
       (append-only, no NEWBOOTLOG).
       Regime: newest engine log is still
       engine-20261001-231328.log (LIVELOG fd
       of API pid 100422 resolves it, size
       6857614, mtime 08:38:34 growing); no
       restart - boot v2-era whole cycle;
       API pid 100422 alive (etime 09:27:12),
       EngineCore pid 100982 alive (etime
       09:26:54); knob
       VLLM_KV_OFFLOAD_ESCAPE_REFUSAL_SECS=2
       and VLLM_KV_OFFLOAD_ESCAPE_WALL_SECS=3
       environ-proven on pid 100422; staged
       v3.1 claim-blockout (commit fee6cbf,
       md5 da376d25 both dests PARITY_OK)
       still deployed, post-restart-gated.
       Engine counters: SK 1174 -> 1180 (+6),
       RF 27 (+0; last b7485652de1371fe
       @04:46:17 unchanged), SW 909 -> 911
       (+2 wall=3.0 release local=0 cold
       re-arm, both adc3b10a9cdd866d),
       KVL 2311 -> 2347 (+36 benign local=0
       36/36); Traceback 0, ImportError/
       AssertionError 0, CUDA-OOM 0,
       GPU-watchdog 0, 5xx 0.
       Wedge window: 08:29:29 -> 08:31:15,
       +6 SERVERKILL across 4 fresh ids
       (b82040ace4072ba9=1 @08:29:29,
       af0867d407472ae3=1 @08:29:50,
       adc3b10a9cdd866d=3x @08:30:16/19/22
       at the wall=3.0 re-arm cadence,
       809d6f26d23a3d60=1 @08:31:15); every
       fresh id silent since 08:31:15
       (0 later SK; probe2 08:38:06; log
       tail 08:40:40, CLEAN).
       Rowlog rows this cycle: [08:31:35]
       h=200 hb=200, [08:36:40] h=200
       hb=200; no new 000 rungs (c471
       single rung @07:56:10 stays healed
       @08:01:13); apc 919 -> 946, kv 5601
       -> 5641, fb 1174 -> 1180 (fb +6 =
       SK +6 exact); tier-full churn ev
       167716 -> 176076.
       Serving intact: POST 200 OK x5 in
       engine tail (APIServer pid=100422);
       rowlog h=200 rows [08:31:35] ->
       [08:36:40]; log tail still growing
       (last KV-TOMBSTONE line 08:40:40).
       Post-restart incident: none (no
       restart; first post-v3.1 microdetail
       stays pending).
       Standing pre-restart regime: wedge
       activity inside the one-off/
       short-of-window frame allowance -
       106 s window, 4 fresh ids, max 3x
       same id adc3b10a9cdd866d at the
       wall=3.0 re-arm cadence, zero new
       000 rungs, detector CLEAN at cycle
       close; ladder trigger (2+
       consecutive rungs) not reached.
       Verdict: QUIET (SK 1174 -> 1180 +6,
       short-window ids all silent since
       08:31:15; RF +0, SW +2 re-arm, KVL
       +36 benign local=0; boot 231328
       v2-era whole cycle knob=2/WALL=3,
       API pid 100422 + EngineCore 100982
       alive, station pid 60644 alive;
       serving 200 intact; staged v3.1
       claim-blockout still
       post-restart-gated).
       Cycle 475 dispatch pending (orchestrator).
