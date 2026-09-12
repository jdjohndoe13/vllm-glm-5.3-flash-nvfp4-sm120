# Patches — provenance and application notes

The shipped `vllm-image-tree/` was built from the vllm package **extracted
from the docker image** (below), with two modifications applied.

**Note: the committed kit ships only the resulting changed files**
(`patched-files/` + `manifest.txt`, 6 files — see below), not the full tree.
Equivalence was verified by diffing the full patched tree against the
image's stock package: exactly the 6 manifest files differ, everything else
is byte-identical to the image.

1. **PR #54743** — "[KV Offload] Scope offload group configs to
   prefix-cacheable KV cache groups" (vllm-project/vllm, **unmerged/open**).
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
   upstream's `prefix_cacheable` attribute introduced/referenced by PR
   #54743. The fork's equivalent property is
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

3. **Overlay files baked into the tree** (also shipped standalone in the repo
   root for the no-offload launcher, which mounts them instead):
   - `vllm-glm-5.3-flash-nvfp4-modelopt.py` →
     `model_executor/layers/quantization/modelopt.py`
     (SM120 NVFP4 serving fix from the cstechdev image's overlay set)
   - `deepseek_v4_mhc_warmup.py` →
     `model_executor/warmup/deepseek_v4_mhc_warmup.py`
     (mHC kernel warmup; removes per-request JIT compile warnings)

## Image / fork provenance

- Image: `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`
  (cu130 build, SM120 fixes baked in by the cstechdev fork)
- The image's vllm is built from the **cstechdev fork, commit `g487ecf187`**
  — NOT an upstream vllm commit. The SM120 overlay fixes (rope-free
  sparse-MLA + kpool; glm5next support per PR #53906 lineage) are part of
  that fork; upstream vLLM could not run this model on sm_120 at image
  build time.
- If you need to inspect the fork's history, the tree was extracted from the
  image exactly as shipped — no need to rebuild anything.
