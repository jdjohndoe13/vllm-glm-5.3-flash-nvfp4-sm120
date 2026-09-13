# Patches — provenance and application notes

The shipped `patched-files/` were produced from the vllm package **extracted
from the docker image** (below), with the modifications applied below.

**Note: the committed kit ships only the resulting changed files**
(`patched-files/` + `manifest.txt`, 7 files — see below), not the full tree.
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
