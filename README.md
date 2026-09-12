# GLM-5.3-Flash-NVFP4 on vLLM — 8× RTX 5090 (sm_120) — reproduction kit

This repository reproduces the exact GLM-5.3-Flash serving configuration that
ran on the `testcomp2` machine as of **2026-09-12**, including the KV CPU
offloading feature that was validated, burst-tested, and promoted to
production on that date.

Target: **8× RTX Pro 5090 (32 GB each, sm_120), TP=8, port 1025, vLLM,
GLM-5.3-Flash-NVFP4 (RedHatAI compressed-tensors checkpoint) with KV CPU
offloading (64 GiB in RAM).**

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
# e.g. from ubuntu's nvidia repo or the .run installer; verify with nvidia-smi)
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
| `patched-files/` | **The only vllm files that differ from the image's stock package** (vllm-project/vllm#54743 port + boot-fix + 2 overlays) — 6 files + `manifest.txt`, mounted individually over the image's vllm at runtime. Verified by full diff against the image (see section 5). |
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
# NOTE: `bash MAX_MODEL_LEN=150000 <script>.sh` does NOT work — bash treats
#       the assignment as the script's filename. Put the assignment before
#       'bash' (or after the script path), not between them.
```

- Wait for `Application startup complete` in the output
  (cold start with empty JIT caches: up to ~25 min due to kernel
  compilation/warmup; warm caches: ~3–5 min; model load itself ~1–2 min).
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

- Optional convenience alias (adjust to your clone location):
  `echo "alias llmglmf='bash $HOME/vllm-glm-5.3-flash-nvfp4-sm120/vllm-glm-5.3-flash-nvfp4.sh'" >> ~/.bashrc`

## 5. How the per-file mounts work (and why it's safe)

The kit does NOT ship a full patched vllm package — it doesn't need to. The
docker image's stock vllm package is complete and runnable by itself; the
kit's `patched-files/` contains only the 6 files that differ from it, and
the launcher bind-mounts them individually over their stock paths inside
the container (paths listed in `patched-files/manifest.txt`):

- `distributed/kv_transfer/kv_connector/v1/offloading/config.py` — vllm-project/vllm#54743 + boot-fix
- `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` — vllm-project/vllm#54743
- `v1/kv_offload/base.py` — vllm-project/vllm#54743
- `v1/kv_offload/config.py` — vllm-project/vllm#54743
- `model_executor/layers/quantization/modelopt.py` — SM120 NVFP4 overlay
- `model_executor/warmup/deepseek_v4_mhc_warmup.py` — mHC warmup overlay

**Equivalence proof**: the originally-shipped full patched tree (extracted
from this image, patch applied, boot-fix sed, overlays baked) was diffed in
full against the image's stock package — exactly these 6 files differ,
everything else is byte-identical. Mounting these 6 files over a stock
image therefore yields the identical runtime to mounting a full patched
tree.

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

- `--kv-transfer-config OffloadingConnector / kv_both / cpu_bytes_to_use 68719476736`
  reserves a **64 GiB CPU tier** (shared `mmap` region in `/dev/shm`,
  ~68.67 GB file — RAM-backed, not SSD; freed on clean shutdown).
- GPU KV pool: 414,634 tokens (`kv-cache-memory 3.3e9`, fp8, block 256).
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
- Keep `cpu_bytes_to_use` ≤ 64 GiB: upstream reports crashes with larger
  CPU budgets (vllm-project/vllm#52656).
- Useful metrics (already exposed on `:1025/metrics`, prometheus-readable):
  - `vllm:kv_offload_cpu_cache_usage_perc` — fraction of the tier **pinned
    by active transfers** (0.0 = idle). NOT tier fill; it reads ~0 even
    when the tier holds GBs.
  - `vllm:kv_offload_store_bytes_total` / `vllm:kv_offload_load_bytes_total`
    — cumulative GB spilled / restored. Rising load counter = the feature
    is earning its keep for your traffic.
- Hybrid-model KV accounting is chunky: ~8 GB tier per ~110k-token payload
  → roughly 4 × 200k-token sessions fit in the 64 GiB tier.

## 7. Known limitations / gotchas

- `MAX_NUM_SEQS=4` and a 414k-token GPU pool → only ~2 × 200k-token
  conversations fit concurrently; the rest queue (admission control).
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
| launcher aborts with `missing patched file:` | repo incomplete — manifest.txt lists 6 files that must exist under `patched-files/` |
| launcher aborts with `pinned image ... not present locally` | get the pinned build: `docker load` the backup tarball, or `docker pull cstechdev/vllm@sha256:0bd709e8...fde5`. If the tag exists but "points to" a different hash, the tag drifted — do NOT run it |
| container dies at boot, GPUs busy | another LLM process holds VRAM: `nvidia-smi --query-compute-apps=pid --format=csv` and stop it |
| port 1025 in use | previous container alive: `docker rm -f vllm-glm-5.3-flash-nvfp4` |
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
