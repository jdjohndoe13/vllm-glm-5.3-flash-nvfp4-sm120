# GLM-5.3-Flash-NVFP4 on vLLM — 8× RTX 5090 (sm_120) — reproduction kit

This repository reproduces the exact GLM-5.3-Flash serving configuration that
ran on the `testcomp2` machine as of **2026-09-12**, including the KV CPU
offloading feature that was validated, burst-tested, and promoted to
production on that date.

Target: **8× RTX 5090 (32 GB each, sm_120), TP=8, port 1025, vLLM,
GLM-5.3-Flash-NVFP4 (RedHatAI compressed-tensors checkpoint) with KV CPU
offloading (CPU_TIER_GB-sized RAM tier; default 512 GiB — the validated
ceiling on this host/driver).**

---

## 1. Assumptions

- **Model files** (auto-detected): if
  `/mnt/huggingface/RedHatAI/GLM-5.3-Flash-NVFP4` exists and is readable,
  the launcher mounts `/mnt/huggingface` read-only and serves it directly
  ([RedHatAI/GLM-5.3-Flash-NVFP4 on Hugging Face](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)).
  If not, the launcher passes the repo id `RedHatAI/GLM-5.3-Flash-NVFP4`
  instead and mounts `$HOME/.cache/huggingface` into the container — vLLM
  then downloads the checkpoint on first boot (**~198 GB**; pre-seed outside
  the container with `hf download RedHatAI/GLM-5.3-Flash-NVFP4` to skip the
  in-container download).
  (⚠ use the **RedHatAI** compressed-tensors checkpoint — the
  [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
  modelopt checkpoint emits corrupted tokens on sm_120, see
  vllm-project/vllm#54150.)
- **Kernel JIT caches**: `/mnt/data/shared/models/vllm-moet-cache/{jit,tilelang}`
  (Triton/deep_gemm/tilelang disk caches, shared across vllm servers on the
  box). If missing, they regenerate automatically on first boot — first
  startup is just slower (JIT compiles + warmup, see section 7). Don't delete
  them while a server is running. If that folder is missing or not writable,
  the launchers automatically fall back to a local `.cache/` folder created
  next to the launcher script. These are the kernel JIT caches — they have
  nothing to do with the HuggingFace model cache (see the model bullet above).
- **Ports**: the server binds `0.0.0.0:1025`. Nothing else must use it.

## 2. Prerequisites (fresh Ubuntu 26.04)

```bash
# NVIDIA driver (the image is a cu130 build — use a current driver,
# e.g. from Ubuntu's nvidia repo or the .run installer; verify with nvidia-smi)
nvidia-smi   # must show 8 GPUs, driver 580+ recommended for CUDA 13

# Docker (official repo recommended over distro package)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"   # then re-login

# NVIDIA Container Toolkit (required for --gpus all)
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# small utilities used by the scripts
sudo apt-get install -y curl python3

# docker pull the image now to confirm access (see backup-image.sh if you
# want to pre-stage it from a docker-save tarball instead).
# The launchers pin the EXACT build by hash — the tag alone is not trusted:
docker pull cstechdev/vllm@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5

# verify whatever you have locally matches the pin:
docker image inspect --format '{{.Id}}' cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
# must print: sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5
```

Image on Docker Hub: [cstechdev/vllm](https://hub.docker.com/r/cstechdev/vllm)
(tag `glm53-flash-nope-sm120-cu130-20260826-r1`,
[direct tag link](https://hub.docker.com/r/cstechdev/vllm/tags?page=1&name=glm53-flash-nope-sm120-cu130-20260826-r1)).

## 3. Repository layout

| path | what it is |
|---|---|
| `patched-files/` | **The only vllm files that differ from the image's stock package** (vllm-project/vllm#54743 port + boot-fix + overlays + diagnostics instrumentation + the `VLLM_KV_OFFLOAD_MIRROR_LOCAL` mirror patch, 2026-09-15) — 12 files + `manifest.txt`, mounted individually over the image's vllm at runtime. Verified by full diff against the image (see section 5). |
| `vllm-glm-5.3-flash-nvfp4.sh` | **Primary launcher** — production config with KV offloading (port 1025, auto-restart, per-file mounts from `patched-files/`). |
| `vllm-glm-5.3-flash-nvfp4-orig.sh` | Fallback launcher — same server WITHOUT KV offloading (stock image package + the 2 overlay files mounted individually). Same container name/port; the launchers guard against each other. |
| `test.sh` | Needle-battery validation test (boots the offload launcher, 6 tests, tears down). |
| `backup-image.sh` | Saves the docker image to a tarball (run BEFORE wiping the old machine). |
| `patches/` | The raw PR diff + provenance/how-to (`patches/README.md`). |

## 4. Quick start (recommended)

```bash
git clone https://github.com/jdjohndoe13/vllm-glm-5.3-flash-nvfp4-sm120.git
bash vllm-glm-5.3-flash-nvfp4-sm120/vllm-glm-5.3-flash-nvfp4.sh
```

All launcher knobs (model paths, image pin, container name, port, context
length, …) sit in the `EDITABLE SETTINGS` block at the top of the `.sh` file
— edit them there, or override per-run without touching the file:

```bash
MAX_MODEL_LEN=150000 bash vllm-glm-5.3-flash-nvfp4-sm120/vllm-glm-5.3-flash-nvfp4.sh
# or equivalently:  bash vllm-glm-5.3-flash-nvfp4-sm120/vllm-glm-5.3-flash-nvfp4.sh MAX_MODEL_LEN=150000
# bigger GPU KV pool (~502k tokens instead of 414k):
KV_CACHE_MEMORY=4000000000 bash vllm-glm-5.3-flash-nvfp4-sm120/vllm-glm-5.3-flash-nvfp4.sh
# NOTE: `bash MAX_MODEL_LEN=150000 <script>.sh` does NOT work — bash treats
#       the assignment as the script's filename. Put the assignment before
#       'bash' (or after the script path), not between them.
```

- Wait for `Application startup complete` in the output
  (cold start with empty JIT caches: up to ~25 min due to kernel
  compilation/warmup; warm caches: ~4–7 min and grows with `CPU_TIER_GB`
  — the driver walks/pins the whole tier at init; 512 GiB measured
  ~6.5 min; model load itself ~1–2 min).
- Verify:

```bash
curl -s http://localhost:1025/v1/models | head -c 400

# smoke test with a needle check:
curl -s http://localhost:1025/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user",
       "content":"Reply with exactly this token: KVTEST-OK"}],"max_tokens":1600}'
```

You should see text that contains `"content":"KVTEST-OK"`.

- The full validation battery (boots, tests, tears down — restart the
  launcher afterwards): `bash vllm-glm-5.3-flash-nvfp4-sm120/test.sh`. Expected verdict:
  `KV OFFLOADING WORKS` with T5 (repeat-after-eviction) ≥ ~10× faster than
  T1 (cold) and all answers correct. Reference run (2026-09-12):
  T1 15.2 s, T2 1.3 s, T3 11.9 s, T4 11.7 s, **T5 1.3 s (11.3×)**, T6 1.6 s.

- Optional convenience alias (adjust to your clone location; the `SERVED_MODEL_NAMES`
  env prefix makes the server answer under both `glm-5.3-flash` and
  `glm-5.3-flash-nvfp4` instead of the default single name):
  `echo "alias llmglmf='SERVED_MODEL_NAMES=\"glm-5.3-flash glm-5.3-flash-nvfp4\" bash $HOME/vllm-glm-5.3-flash-nvfp4-sm120/vllm-glm-5.3-flash-nvfp4.sh'" >> ~/.bashrc`

## 5. How the per-file mounts work (and why it's safe)

The kit does NOT ship a full patched vllm package — it doesn't need to. The
docker image's stock vllm package is complete and runnable by itself; the
kit's `patched-files/` contains only the 8 files that differ from it, and
the launcher bind-mounts them individually over their stock paths inside
the container (paths listed in `patched-files/manifest.txt`):

- `distributed/kv_transfer/kv_connector/v1/offloading/config.py` — vllm-project/vllm#54743 + boot-fix
- `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` — vllm-project/vllm#54743
  + KV-LOOKUP diagnostics (2026-09-14) + `VLLM_KV_OFFLOAD_MIRROR_LOCAL`
  mirror/touch/bypass patch (2026-09-15)
- `distributed/kv_transfer/kv_connector/v1/offloading/metrics.py` — mirror
  counters `vllm:kv_offload_mirrored_keys_total` /
  `vllm:kv_offload_touch_keys_total` (2026-09-15; clone copy + 2 edits)
- `v1/kv_offload/base.py` — vllm-project/vllm#54743 + `prepare_store`
  `bypass_threshold` kwarg (2026-09-15)
- `v1/kv_offload/config.py` — vllm-project/vllm#54743
- `v1/kv_offload/cpu/shared_offload_region.py` — CPU tier mmap region hint
- `v1/kv_offload/cpu/manager.py` — diagnostics instrumentation (fix-3,
  2026-09-14) + counts-gate `bypass_threshold` (2026-09-15)
- `v1/kv_offload/cpu/common.py`, `v1/kv_offload/cpu/spec.py` — diagnostics
  instrumentation (fix-3, 2026-09-14)
- `model_executor/layers/quantization/modelopt.py` — SM120 NVFP4 overlay
- `model_executor/warmup/deepseek_v4_mhc_warmup.py` — mHC warmup overlay
- `v1/core/kv_cache_coordinator.py` — admission instrumentation overlay
  (APC-HIT log line, added 2026-09-14; diagnostics only, no behavior change)

**Equivalence proof**: the originally-shipped full patched tree (extracted
from this image, patch applied, boot-fix sed, overlays baked) was diffed in
full against the image's stock package — exactly the 8 originally-listed
files differ, everything else is byte-identical. Mounting these files over a
stock image therefore yields the identical runtime to mounting a full patched
tree. Files added later are each a minimal verified diff vs the stock image:
the 2026-09-14 diagnostics trio (`cpu/manager.py`, `cpu/common.py`,
`cpu/spec.py` + the scheduler/base/core overlay edits) and the 2026-09-15
mirror patch (`offloading/metrics.py` = clone copy + 2 counter edits;
scheduler/base/manager edits py_compile-verified).

The image itself remains the one artifact this repo cannot carry (torch/
CUDA/deps substrate, ~29 GB built artifact): pull it from the registry on
the new machine, or pre-stage it with `backup-image.sh`
(`docker save` → `docker load`) before wiping the old machine.

**Image pin**: the validated build of
[cstechdev/vllm](https://hub.docker.com/r/cstechdev/vllm) is
`sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5`
(config digest and registry manifest digest coincide for this image). Both
launchers run that ID directly and refuse to start if it is absent — even
if the tag exists locally with different content (i.e., the tag was
re-published under the same name).

## 6. What the KV offloading config does (and what to expect)

- `--kv-transfer-config OffloadingConnector / kv_both / cpu_bytes_to_use`
  reserves a **CPU tier** sized by the launcher setting `CPU_TIER_GB`
  (default 512 GiB — the validated ceiling, see "Budget physics" below;
  raise/lower via `CPU_TIER_GB=256 ./vllm-glm-5.3-flash-nvfp4.sh`, also
  validated). The extra config also sets `offload_prompt_only: false`
  (2026-09-14): the upstream default is `true`, which stores only prompt
  tokens — the model's own replies were never offloaded, so under
  concurrent multi-turn use each session lost its own 8k reply between
  turns (T2 cached 93.6%) while prompts survived. With `false`, replies
  are stored to the tier as well; validated T1 99.6% and T2 99.7%/99.5%
  (next-turn, interleaved sessions). Tier fill rate is higher with this
  knob on — watch host RAM.
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` — **ATTEMPTED AND REVERTED
  (2026-09-14)**. A `4096` `-e` env var was deployed to stop the whole-path
  cache losses; the 0% ↔ 97% alternation was IDENTICAL with and without it
  and deployed-build tracing proved the knob never gated the entries that
  served the observed hits. Reverted to the default (None = dense). Do not
  re-add it.
- Whole-path cache losses (long sessions alternate `cached=0` full recomputes
  ↔ 97%+ deep hits): current picture after deployed-build source tracing
  (2026-09-14). Every hash-strip path in the scheduler is enumerated and
  none wipes the hash map wholesale (frees RETAIN hashes; scan-miss ⇔ entry
  absent). The observed signature — zero turns with byte-identical request
  heads (probe-verified), collapsed turns hitting ONLY the shallowest states
  (2048/4096), and a total KV budget of 414,634 tokens (~405 blocks of 1024,
  "Maximum concurrency 2.07x for 200k-token requests") — points to
  **mamba/GDN GPU pool pressure**: concurrent 100k+ sessions overflow the
  small mamba pool; its freed states are re-allocated (hash-stripped)
  between turns and the min-intersection hit gate collapses the whole
  admission to 0/shallow. The CPU tier stores heavily (hundreds of GB
  cumulative) and — as of the 2026-09-14 evening align-mode boot — serves
  loads back (see the KV_CACHE_MEMORY bullet above); live-session chain
  storage is the remaining gap. One-line admission instrumentation is deployed
  (patched-files/v1/core/kv_cache_coordinator.py): watch
  `APC-HIT nhash= max= final= per_group=[...] uncached=` in docker logs —
  group order is FA first, mamba/GDN last; `uncached>0 ∧ final=0` = FA
  alive + mamba group missed everything (pool churn confirmed);
  `uncached=0 ∧ final=0` = head-of-chain mismatch.
- **`KV_CACHE_MEMORY` raised 3.3e9 → 4.0e9 (2026-09-14, ~414,634 → ~502k
  tokens)**: the captured APC-HIT evidence settled the loss mechanism —
  concurrent ~140k-token agent sessions (especially parallel-turn bursts,
  `MAX_NUM_SEQS=4`) need more than the total KV pool holds (~405 blocks of
  1024), so whole cached chains get evicted between turns and the
  min-intersection hit gate collapses the admission to `cached=0` (all six
  groups miss, `uncached=0`). 4.0e9 was previously proven to boot standalone;
  per-rank VRAM headroom is ~1.1 GB so do not push past ~4.5e9 without a
  boot test. **The CPU tier now DOES serve loads back (2026-09-14
  evening)**: first loads observed on the stock align-mode config —
  ~9.3 GB restored (`CPU_to_GPU` counter, count 40), including a full
  133,120-token chain at all-group depth (`KV-LOOKUP hit` with the FA group
   and all four mamba/GDN groups at hit=130). The "remaining gap" was
   solved on 2026-09-15 — see the STALL ROOT CAUSE bullet below: live
   agent-session chains are absent from the tier because only
   newly-computed chunks are ever offered for tier store; the locally-hit
   bulk is never mirrored.
- **STALL ROOT CAUSE (2026-09-15, decoded from proxy-captured requests +
  metrics + KV-LOOKUP traces — the multi-turn session "disease")**: a live
  multi-agent session lives ENTIRELY in the GPU radix pool; the CPU tier
  never receives its bulk. Evidence chain (testcomp2, 03:07–03:17):
  1. Proxy capture pair — the 03:09:49 question request and the 03:12:43
     answer ("pick") request have a **byte-identical 42-message prefix and
     identical 142-tool payload** → same agent's next turn; no compaction,
     no context rewrite. (Proxy request logs at
     `B:\intersub\temp\llm-proxy\logs\<epoch_ms>.req/.resp.json`.)
  2. KV-LOOKUP traces — question turn: `local=128000 hit=0` (served ~98%
     from the GPU-local radix pool, tier hit=0). Pick turn after a ~3-min
     user pause: `local=0 hit=0` (chatcmpl-be8618bc, 138,240 tok) → full
     re-prefill 03:13:00–03:13:10 at 13,852 tok/s ≈ **10 s**. Next turn
     03:13:28 was fast again (`local=139264`).
  3. **The tier never gets chain content**: every chain turn in the window
     shows tier `hit=0` while local serves 95–98%; the only CPU-TIER-EVICT
     store batches observed are tiny ~5-key tail      slivers. Metrics confirm
     scope: all 9,366 tier store/allocation batches are ≤16 keys (avg 5.6;
     74,928 one-key store ops, zero large batches) — `_build_store_jobs`
     (offloading/scheduler.py ~:1290) offers the full computed frontier,
     but the store-frontier semantics (`next_stored_chunk_idx` start +
     jump-to-frontier after any accepted sliver + already-stored filter)
     mean locally-hit bulk is never RE-offered once its tier copy is
     evicted. Fresh
     full-prefill requests DO tier fully (03:13:37 → 03:14:01 a 140/140
     full-tier hit), and the soak's repeated contexts survive via
     restore-touches-MRU every iteration.
  4. Churn math: tier = 8,854 keys (≈ 512 GiB, ~30.7 MB/key); soak churn
     ~43,000 evictions in 2 h ≈ 2.5 tier sweeps/h — anything not
     re-touched within ~10–15 min is gone. GPU pool = 4.0e9 ≈ 502k tokens
     shared by `MAX_NUM_SEQS=4`; one concurrent 162k-token soak restore
     alone ≈ a full-pool flush. So: pause + any concurrent traffic →
     local=0, tier=0 → full re-prefill of the whole chain.
  5. Known caveat: lookup can match write-pending tier keys before their
     data lands (03:13:37 attempt showed a full `hit=143360` tier scan for
     keys that vanished by 03:14:27 — prepare_store inserts keys before
     the copy lands; aborted/canceled requests leave false-positive
     entries). Mirroring must not touch write-pending keys.
  6. (2026-09-15, 04:42 decode) **Restored tier keys are re-evicted within
     ~30 s**: the 04:41:33 report turn was tier-served (119/122 chunks,
     121,856 tokens) yet the next turn 34 s later (04:42:07) found tier
     hit=1 — with a byte-identical 91-message shared prefix (proxy-verified;
     no rewrite, no compaction). Mechanism: `prepare_load` pins loaded keys
     and `complete_load` unpins WITHOUT an MRU touch, so restored keys drop
     back to their old (cold) LRU position; the soak's eviction wave
     (~5 evictions/s) re-evicts them almost immediately. The soak's own
     contexts survive only via per-iteration entry-touches of their own
     keys. This also explains the earlier 03:14:01→03:14:27 and
     03:32:39→03:33:17 anomalies (tier-full→0 in 26-38 s). The deployed
     patch's lookup-time touch fires BEFORE the load pins, closing this
     path.
- **Fix direction (user-approved 2026-09-15)**: (a) mirror locally-hit
  (radix-resident) chunks to the tier so RAM always holds a copy, plus
  (c) touch tier keys to MRU on local hits so active sessions keep tier
  residency — and the mirrored copy IS the "properly offloaded to RAM"
  guarantee when VRAM pressure evicts (concurrent request during/after a
  refresh). Enlarging the GPU pool was explicitly REJECTED: it cannot
  survive a +1k-context arrival or an idle past the pool's retention —
  long-term residency must live in the RAM tier. **Patch IMPLEMENTED
  (2026-09-15, in `patched-files/`, takes effect on next restart)**: env
  flag `VLLM_KV_OFFLOAD_MIRROR_LOCAL` (default ON; `0`/`false`/`off`
  disables) — (a) store offers start from chunk 0 so tier-evicted chunks
  are re-mirrored on every step (the manager's already-stored filter makes
  still-resident keys a no-op; BOTH store-job loops use the same start so
  no key is allocated without a copy), (b) ready tier hits are touched to
  MRU on lookup (HIT_PENDING/write-pending keys never touched), (c) the
  counts gate gets a `bypass_threshold` kwarg passed by mirror offers
  behind an introspection guard (the gate is dead in this deployment — the
  launcher passes no `store_threshold`, default 0). New 12th mounted file
  `offloading/metrics.py` exposes
  `vllm:kv_offload_mirrored_keys_total` +
  `vllm:kv_offload_touch_keys_total`; expected post-restart signature:
  mirrored_keys ramps to ≈ full-chain keys per turn, large allocation
  buckets (16-256 keys) become non-zero, chain turns show tier `hit≠0` in
  the KV-LOOKUP trace, and the multi-turn re-prefill stalls disappear.
  Write amplification is bounded by the eviction rate (only tier-evicted
  chunks re-copy; ~137 keys × ~62 MB ≈ 8.5 GB worst case per cold chain —
  far cheaper than the ~10 s re-prefill it replaces).
- `--mamba-cache-mode all` (dense mamba block ids for the tier) —
  **ATTEMPTED AND REVERTED (2026-09-14)**: on this hybrid model every block
  id is charged the FULL per-block byte sum (MLA+indexer pages; the
  kv_cache_utils memory check), so one 200k-token request needs ~7.1 GiB of
  GPU KV versus the ~3.73 GiB/rank budget at `KV_CACHE_MEMORY=4.0e9`;
  `--mamba-block-size 1024` does not change the charge and no launcher knob
  makes "all" fit on 32 GB cards. The deployed config keeps the stock align
  mode (auto-selected when prefix caching is on — boot log: `Mamba cache
  mode is set to 'align' ... by default`). Note the boot behavior: the
  workers auto-bump the attention block size to 1024 tokens so the
  attention page is at least the mamba page (interface.py:926/:950, 23.77%
  padding) — the effective block is 1024 tokens even though `block_size 256`
  is passed.
- It lives in the HOST's
  `/dev/shm` because the launcher runs with `--ipc=host`, so Docker's
  `--shm-size` flag is ignored there; when the tier exceeds the shm mount's
  size (default: half of RAM = 504 GiB), the launcher **remounts /dev/shm
  larger automatically** (tmpfs size is a cap, not a reservation — raising
  it charges nothing; needs passwordless sudo or root, else the gate error
  prints the manual `mount -o remount,size=…G /dev/shm` plus the fstab line
  for persistence). Budget physics: the tier file is **preallocated at full
  size on boot AND pinned via cudaHostRegister** (`PIN_MEMORY` = CUDA
  available → true here) — i.e. tier GiB are unpageable RAM consumed from
  the moment the engine starts (leave ~60+ GiB for OS + engine processes +
  psm/sem files). The tier-size ceiling on this host is NOT RAM — it is the
  NVIDIA driver's per-rank pinned page-table budget (~537–600 MB of driver
  page tables per rank; a hard driver limit with no knob, not a RAM-size
  limit): **512 GiB total (8 ranks × 64 GiB pinned each) is the validated
  maximum**; 576, 640, and 800 GiB tiers all fail on all
  8 ranks with `cudaHostRegister failed (code=2)` +
  `NVRM: failed to allocate page table`, independent of free RAM,
  fragmentation, or compaction (bisected 2026-09-13). Above the ceiling the
  boot completes but leaves the tier UNPINNED (degraded). The tier region
  is registered into every TP rank's GPU context, so the budget is charged
  per rank. The tier file (`vllm_offload_<uuid>.mmap`) is **unlinked
  only on graceful engine exit**:
  `docker stop` SIGKILLs the engine first, so every restart LEAKS the
  tier file (measured: 5 orphaned files = 412 GiB filled host /dev/shm to
  89% and made subsequent starts fail with `RuntimeError: Insufficient
  space in /dev/shm` from `shm_broadcast.check_shm_free_space`). The
  launcher's pre-flight deletes leaked tier mmaps and orphaned torch
  shm files (`psm_*`, `sem.mp-*`) after stopping the container, then
  gates on free space; tier files are root-owned, so a non-root start
  needs passwordless sudo for the auto-wipe — otherwise the gate error
  prints the manual command: `sudo rm -f /dev/shm/vllm_offload_*.mmap`.
- GPU KV pool: ~502,439 tokens (`KV_CACHE_MEMORY=4000000000`, fp8). The
  launcher passes `block_size 256`, but the boot auto-bumps the attention
  block to 1024 tokens to match the hybrid mamba page (23.77% padding) —
  the effective block is 1024 tokens.
  When the pool fills, evicted prompt KV blocks spill to the CPU tier and
  are **restored from it** when you revisit those prompts — measured
  ~11 GB in ~0.35 s (~32 GB/s), i.e. revisit-after-eviction runs ~11–20×
  faster than re-prefill.
- The tier is a **rolling spill buffer, not a durable long-term cache**: if
  you push far more concurrent tokens than the GPU pool (overflow-scale
  churn), the tier degrades gracefully to re-prefill speed. Correctness
  held in all tests (needle checks before/after eviction, under
  preemption, and under concurrent overflow).
- Sustained long-decode-with-store churn (the upstream ~33% penalty claim,
  vllm-project/vllm#55035) was not reproduced in our short-decode tests; your real usage
  pattern is the arbiter.
- Upstream vllm-project/vllm#52656 reported silent serve-crashes at 128 GB
  CPU budgets and boot failure at 256 GB — on OTHER stacks (GLM 5.2,
  `block_size: 1`, B200/MI325X, v0.26–0.27.1). **Re-tested on this stack
  (2026-09-12): 128 GiB booted and served; 256 GiB booted, served, and
  absorbed 138.8 GB of store-counter volume with
  zero failed requests** — disjoint ~160-170k real-session prefills stored
  ~17-22 GB each, cross-conversation restores pulled ~38k shared
  system-prompt tokens from the tier even across days, and a repeat after
  eviction restored 158,720 tokens (TTFT 157 ms). No crash in either
  failure mode. The tier-full edge (tier actually occupied — 256 GiB ≈
  ~3.5M tokens at the measured tier occupancy) was not exercised; expect
  graceful degradation to re-prefill per the design.
- Useful metrics (already exposed on `:1025/metrics`, prometheus-readable):
  - `vllm:kv_offload_cpu_cache_usage_perc` — fraction of the tier **pinned
    by active transfers** (0.0 = idle). NOT tier fill; it reads ~0 even
    when the tier holds GBs.
  - `vllm:kv_offload_store_bytes_total` / `vllm:kv_offload_load_bytes_total`
    — cumulative GB spilled / restored. Rising load counter = the feature
    is earning its keep for your traffic.
- Response-level detail (enabled by launcher flags): the OpenAI-compatible
  response carries `usage.prompt_tokens_details.cached_tokens` (prefix-cache
  hits for that prompt — `--enable-prompt-tokens-details`) and a `metrics`
  object with per-request timing (`time_to_first_token_ms`,
  `generation_time_ms`, `queue_time_ms`, `mean_itl_ms`,
  `tokens_per_second` — `--enable-per-request-metrics`). Measured on
  testcomp2: repeating a 109,876-token prompt reported
  `cached_tokens: 109568` and cut TTFT from ~13.7 s to ~0.15 s. A
  non-destructive self-check for all of this ships with the kit:
  `bash test_kv_cache.sh` (verifies cached_tokens, the metrics object and
  null transfer params, with prometheus counter attribution and an
  idle-stability gate). Caveats:
  - Prefix-cache hits are **block-granular** (`block_size` 256) and this
    hybrid model does **not reuse its first 256-token block** — so prompts
    shorter than ~2 blocks (~513 tokens) legitimately report 0 even when
    repeated verbatim. Use a filler of several thousand tokens to see
    nonzero `cached_tokens`.
  - `created_cache_tokens` is currently always 0 in this build (the
    finalize-time estimate reads 0 after the fact), even for requests that
    demonstrably wrote the cache.
  - `kv_transfer_params` / `ec_transfer_params` stay `null` by design here:
    they are populated only by disaggregated P2P KV-transfer connectors
    (LMCache/NIXL-style flows), not by the local CPU offload connector;
    request-side `kv_transfer_params` IS supported by the patched connector
    as a per-request knob (`max_offload_tokens`, `kv_load_tiers` tier
    matchers).
  - Right after boot, the JIT warmup sweep inflates the `prefix_cache_*`
    prometheus counters (its dummy prefills query the cache repeatedly).
    Any attached LLM agent session re-prefills its whole conversation
    context every turn, so the counters keep moving whenever such a
    session is active. Intermittently the engine also RESETS these
    counters during quiet windows (values decrease; observed twice on
    testcomp2, cadence unexplained) — `test_kv_cache.sh` detects this and
    labels attribution accordingly; per-request counter deltas remain
    exact when no reset lands inside their snapshot pairs.
  - CPU-offload tier behavior with REAL session content (measured by
    replaying llm-proxy-captured 108-165k-token agent requests on
    testcomp2): the connector stores every finished request's KV to the
    tier — a 164k-token request wrote ~40 GB on finish (`store_bytes_total`
    counts ~245 KB/token, higher than the ~63 KB/token load counter — the
    store counter appears to include staging overhead). Requests sharing a
    prefix restore their overlapping blocks FROM THE TIER automatically
    (`external_prefix_cache_hits_total` + `load_bytes_total` deltas), even
    across "sessions". After eviction (three disjoint large prompts pushed
    the 414k-token pool past capacity), repeating the first prompt
    restored ~126k tokens (7.8 GB) from the tier in ~0.15 s of TTFT
    overhead: total wall 4.2 s / TTFT 149 ms versus 25.4 s / TTFT 21.3 s
    fresh (~140x). Tier restores run at RAM speed (~50 GB/s), so a
    restore is nearly free compared to re-prefill.
  - During 140-165k-token prefills colliding with tier restores the CUDA
    caching allocator can emit OOM-retry warnings (~318 MB ask vs ~240 MB
    free per GPU); it flushes its cache and retries — observed twice in
    one hour of heavy traffic (once from a real session), zero failed
    requests. Treat as transient. If hard `CUDA out of memory` errors ever
    appear, lower `kv_cache_memory` in the launcher slightly (e.g. 3.3e9 →
    3.1e9) to leave activation headroom; that shrinks the GPU pool
    proportionally.
- Hybrid-model KV accounting is chunky: ~8 GB tier per ~110k-token payload
  → roughly `CPU_TIER_GB/8` × 110k-token payloads fit in the tier
  (~64 × 110k at the default 512 GiB, more with prefix overlap).

## 7. Known limitations / gotchas

- `MAX_NUM_SEQS=4` and a `KV_CACHE_MEMORY`-sized GPU pool (default 3.3e9
  bytes → 414,634 tokens fp8; `KV_CACHE_MEMORY=4000000000` → ~502k,
  `5000000000` → ~628k, the latter proven to boot with the tier in a
  196k-token 2-concurrent-request test) → with the default pool only
  ~2 × 200k-token conversations fit concurrently; the rest queue (admission
  control).
- Total context limit is 200,000 tokens **including** output tokens —
  generated payloads/tests must keep prompt+output under that
  (a 912k-char filler text ≈ 199.8k tokens will be rejected with a 400).
- First boot on empty JIT caches logs JIT-compilation warnings; the mHC
  warmup sweep (baked overlay) reduces per-request compile warnings to
  benign disk-cache loads.
- `--restart=unless-stopped` means the container comes back automatically
  after reboot — but the **first request after a fresh boot can hit a still
  starting engine**; the engine answers on the port only after
  `Application startup complete`.
- The mounted patched files are `:ro` — edit them deliberately, never casually.

## 8. Switching to the no-offload fallback

```bash
bash vllm-glm-5.3-flash-nvfp4-orig.sh
```

Same name/port; the guard inside stops and removes the running instance
first. Switch back the same way with the primary launcher. (`docker logs -f
vllm-glm-5.3-flash-nvfp4` follows whichever container is up.)

## 9. Patch source preservation (DONE — preserved in a fork)

The offload fix comes from PR vllm-project/vllm#54743 ("[KV Offload] Scope offload
group configs to prefix-cacheable KV cache groups"), which is **open /
unmerged upstream**:

- head commit: `899699c74ae2b8e8adc8726e5c9d0e355935076a`
- base commit: `504bb8b0c39dbde713a7344772bdb7005adbb214`

Because it is unmerged, the commit is **not reachable from vllm master**. It
is preserved in a fork (branches are not removed there):

- Fork: `jdjohndoe13/vllm`
- Branch containing the commit:
  https://github.com/jdjohndoe13/vllm/tree/pr-54743-kv-offload-prefix-cacheable
- SSH clone: `git@github-jdjohndoe13:jdjohndoe13/vllm.git`
  (`github-jdjohndoe13` is an SSH host alias from the original machine's
  `~/.ssh/config`; on a fresh machine recreate that alias or simply use
  `git@github.com:jdjohndoe13/vllm.git` with your own GitHub key)

```bash
git clone git@github.com:jdjohndoe13/vllm.git
cd vllm
git checkout pr-54743-kv-offload-prefix-cacheable
```

The shipped `patches/pr54743.notests.diff` is generated from exactly that
commit, so the kit remains self-contained even without the fork — the fork
is belt-and-suspenders for future archaeology.

Runtime provenance: the docker image
([cstechdev/vllm](https://hub.docker.com/r/cstechdev/vllm) on Docker Hub)
is built from the sources at
https://github.com/chriswritescode-dev/glm-5.3-flash-sm120 — that repo is
the "cstechdev fork" whose vllm commit is `g487ecf187` (NOT an upstream
vLLM commit — it carries the sm_120 rope-free sparse-MLA + kpool fixes for
glm5next, vllm-project/vllm#53906 lineage + fork fixes
vllm-project/vllm#53969). A mirror fork of those image sources is kept at
https://github.com/jdjohndoe13/glm-5.3-flash-sm120-docker-image-sources in
case the original disappears. The image is pinned by digest in every
launcher; back it up with `backup-image.sh` if registry availability is
ever a concern.

## 10. Troubleshooting quick list

| symptom | check |
|---|---|
| `AssertionError ... tokens_per_block ... tokens_per_hash` at boot | the mounted offloading config is stale/wrong — `grep participates_in_prefix_caching patched-files/distributed/kv_transfer/kv_connector/v1/offloading/config.py` must match (line ~62); if not, re-copy from the fork (README §9) |
| `'UniformTypeKVCacheSpecs' object has no attribute 'prefix_cacheable'` | same — boot-fix missing in the mounted config.py |
| launcher aborts with `missing patched file:` | repo incomplete — manifest.txt lists 8 files that must exist under `patched-files/` |
| launcher aborts with `pinned image ... not present locally` | get the pinned build: `docker load` the backup tarball, or `docker pull cstechdev/vllm@sha256:0bd709e8...fde5`. If the tag exists but "points to" a different hash, the tag drifted — do NOT run it |
| container dies at boot, GPUs busy | another LLM process holds VRAM: `nvidia-smi --query-compute-apps=pid --format=csv` and stop it |
| port 1025 in use | previous container alive: `docker rm -f vllm-glm-5.3-flash-nvfp4` |
| boot aborts with `No available memory for the cache` / `estimated maximum model length` far below expected | `KV_CACHE_MEMORY` too small for the per-request KV charge at `MAX_MODEL_LEN` — raise it (≤ ~4.5e9 proven on 32 GB cards) or lower `MAX_MODEL_LEN`/`MAX_NUM_SEQS` |
| answers look corrupted / U+FFFD garbage | wrong checkpoint (LibertAIDAI modelopt) — use RedHatAI compressed-tensors (vllm-project/vllm#54150) |
| offload restores never happen (loads stay 0) | tier present but nothing evicts; check `kv_offload_store_bytes_total` grows when the GPU pool fills |

## 11. Reference upstream items

- vllm-project/vllm#54831 — KV offloading impossible for GLM-5.3
  (DSA indexer) — the architecture-level blocker this kit works around via
  the fork's scratch-group design + vllm-project/vllm#54743 scoping.
- vllm-project/vllm#54150 — modelopt NVFP4 checkpoint corruption on sm_120.
- vllm-project/vllm#53963, vllm-project/vllm#53906, vllm-project/vllm#53969 —
  glm5next-on-sm120 enablement lineage (the fork image's fixes).
- vllm-project/vllm#52656 — CPU offload budgets >64 GiB crash.
- vllm-project/vllm#55035 — offloading throughput A/B (the ~33%
  decode-penalty claim for sustained churn).
