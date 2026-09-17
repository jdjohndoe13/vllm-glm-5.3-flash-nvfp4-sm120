# TESTING.md — vLLM KV-Offload / Spec-Decode Test Suite (portable)

Stress + regression suite for **bare-metal long-context LLM servers with KV-offload
(OffloadingConnector), prefix caching, and MTP/speculative decoding**.
Home: `testcomp2:/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/` (GLM-5.3-Flash-NVFP4,
8× RTX 5090, max_model_len 200k, GPU KV pool ~334k tokens, CPU tier 800 GiB).
Adapt for other kits by editing the `CONFIG` block below — everything else is generic.

## CONFIG (edit per kit)

| Var | GLM kit value | Notes |
|---|---|---|
| `BASE` | `http://127.0.0.1:1025` | server root (no trailing path) |
| `MODEL` alias(es) | `glm-5.3-flash`, `qwen-3.8-flash-next` | both served (`--served-model-name`); tests address the qwen alias — leave if served, else change |
| `max_model_len` | 200000 | payloads below must stay under prompt+completion ≤ this |
| GPU KV pool | ~334k tokens (`kv_cache_memory_bytes` 3.0e9) | storm sizing = N × payload_tokens > pool |
| CPU tier | 800 GiB (`cpu_bytes_to_use`) | see Tier math section |
| Scheduler | `max_num_batched_tokens=1024`, `max_num_seqs=4` | contention behavior scales with this |

Scripts read their own constants at head — when porting, update `BASE`, `MODEL`,
and the `build(total_chars, tail_chars)` sizes there (GLM ~3.85 chars/token).

## Preconditions (before ANY test)

1. Engine alive: `curl -s {BASE}/health` → 200; `/v1/models` shows the aliases you'll target.
2. Record boot snapshot (one command):
   `L=$(ls -t logs/engine-*.log | head -1); grep -c OutOfMemoryError $L`
   → the OOM baseline; every test should return it unchanged (or per-test criterion).
3. Engine log residue: `grep -E 'WARNING|ERROR' $L` count (known inventory:
   KVConnectorBase_V1 experimental, symm-mem unsupported, custom-allreduce PCIe,
   MBT-vs-spec warning, Triton JIT `jit_monitor` spikes at first new shapes — all
   benign). **Anything OUTSIDE this inventory is a finding.**
4. **No other traffic against the server** — the soak and restore tests are
   LRU-sensitive and queue-sensitive (see Contention section). Production sessions
   on the same server WILL contaminate timings and eviction counters.
5. Host sanity (for cross-machine consistency): CPU governor `performance`,
   GPU clocks locked or known-band, `free -g` headroom ≥ 40 GB available.

## Test matrix

| # | Script | Class | Duration | Purpose |
|---|---|---|---|---|
| 1 | `test-gen-speed.sh` | perf probe | ~4 min | Decode throughput sanity; seeded (variance ≤3%) |
| 2 | `test-prefill-sweep.sh` | perf probe | ~4 min | Prefill tok/s at growing context; sentinel rows written to `logs/prefill-sweep.tsv` |
| 3 | `test_preempt_flush.py` | **crash regression** | ~2–10 min | Oversubscription storm: N concurrent oversized sessions + tier restore, reproduces the 2026-09-16 flush-storm crash |
| 4 | `test_cycle.sh` | **data regression** | ~8–10 min | Tier archive persistence across pressure interleaves |
| 5 | `test_churn.sh` | **data regression** | ~2 min | Sequential churn + grow-back on same session |
| 6 | `soak_watch.sh` | **soak / watch** | 4.5 h default | Loop: marker restore (perf gate) + fresh churn (capacity pressure); alerts on M1 >10 s |

Suggested order on a healthy engine: 1 → 2 → 3 → 4 → 5 → 6.
Tests 3-5 may run interleaved if doing combined stress, but then per-test
timings are contaminated — never draw perf conclusions from interleaved runs.

### 3. test_preempt_flush.py — oversubscription storm (the must-pass test)

`python3 test_preempt_flush.py` (from kit root). Deterministic in-file payloads.

- **Phase 0**: single fresh ~110k-token prompt (control).
- **Phase 1**: 3 concurrent ~166k prompts + a staggered 4th ⇒ sustained total demand
  ~665k tokens vs ~334k pool ⇒ guaranteed mid-decode preemption + multi-thousand-
  block tier flush + restore storms. This is what crashed 4 boots on 2026-09-16
  (`engine-20260916-0{45138,54426,62647,72727}.log`, `cuMemcpyBatchAsync` error 1 in
  `swap_blocks_batch`; patched by `_SWAP_CAP=32` in `vllm/v1/kv_offload/cpu/gpu_worker.py`).
- **Phase 2**: re-send one Phase-1 prompt ⇒ tier restore path (~0.9–2.5 s expected).
- Pass: every request HTTP 200, phase-2 wall < 5 s, engine health 200 after,
  OOM baseline unchanged, no `cuMemcpyBatchAsync` errors anywhere in the log,
  no `/tmp/vllm_swap_diag/recovery_*.json` created.
- Sizing per kit: pool_tokens // payload ≈ how many concurrent sessions you want
  preempting; keep ≥ 2× (2 concurrent + stagger) for the same coverage.

### 4. test_cycle.sh — tier persistence across pressure

Sequence: fresh P1 (165k) → fresh P2 (165k) → P1+AIP growth → 10× fresh pressure
fills (C0..C9) → **P1 growth re-send (disease test A: expect ≈2 s tier restore)** →
P2 growth re-send (disease test B). Gauges before/after; asserts the tier root
hash bookkeeping survives interleave (the 2026-09-13/15 "silent uncounted removal"
class of bugs). Pass = restore walls ≈2–3 s; eviction gauges explain any miss.
**Bare-metal footnote**: script's final `docker logs` step is docker-launcher-era
and exits 1 with empty output on bare-metal boots — fixture artifact, not failure;
bare-metal variant should grep `logs/engine-*.log` (see KV-LOOKUP note) instead.

### 5. test_churn.sh — sequential churn + grow-back

P1 → P2 → P1G1 (grow stored) → P3 (currently OVERSIZED: 780k chars ≈ 202k tokens >
max_model_len 200000 ⇒ server-correct 400; trim to `build(760_000, …)` when porting)
→ P1G2. Pass: restores hit tier unless intentionally LRU'd; grow-back reuses prefix.

### 6. soak_watch.sh — long unattended stability watch

`nohup bash soak_watch.sh > /tmp/soak-stdout.log 2>&1 & echo $! > /tmp/soak.pid`
(stop: `kill $(cat /tmp/soak.pid); pkill -f soak_watch.sh`)
90 iterations × ~3 min; each iter: **M1** marker (~165k, seed 900000) restore —
PASS gate ≤10 s; **X_i** fresh 165k churner (fills GPU pool → offloads → tier);
**M1G** marker+growth. Logs `/tmp/soak.log`. Alert semantics: `M1 >10 s at i>0`
= tier restore missed (either legit LRU eviction of M1's blocks — check
`evicted_total` delta — or a disease regression). GPU pool legit-fills at ~iter 14
(334k/165k); tier legit-fills later. First-15-iterations `evt>0` = something else
(usually concurrent traffic) is churning the tier — investigate before trusting.

## Metric semantics (learned the hard way 2026-09-17 — read before interpreting)

`/metrics` offload gauges (`kv_offload_cpu_*`), emitted by `cpu/manager.py::get_stats`:

| Gauge | Meaning | Trap |
|---|---|---|
| `allocation_size` histogram | **Number of CPU blocks requested per `prepare_store` call** (units = blocks) | `_sum` is a block-request sum, NOT bytes; do not divide by anything to get GB |
| `allocated` | `_num_allocated_blocks`: fresh handouts ever (never shrinks except reset). Occupied = `allocated` − `free_list_len` | `allocated == num_blocks` = every pool block in use, incl. recycled lives since |
| `free_list_len` | recycled-but-empty blocks | 0 with `allocated==num_blocks` = pool fully committed (LRU active) |
| `evictable_len` | idle stored blocks (ref_cnt 0) | pinned (in-flight/pending) = occupied − evictable |
| `evicted_total` | cumulative LRU evictions since boot | monotone; *rate* is the health signal |
| `cache_usage_perc` | pinned-only fraction (reads ~0% on an idle-but-full tier) | never use as tier occupancy — use occupied formula above |

Timeline sanity for the GLM kit: block count 12,412 × chunk ≈ 69 MiB = 800 GiB tier
(`num_blocks = cpu_bytes_to_use // aligned_kv_bytes_per_chunk` in `cpu/spec.py`;
`kv_bytes_per_chunk = worker_kv_bytes_per_block × world_size(×blocks_per_chunk)`).
One ~165k-token archive ≈ 30 GB ≈ 440 blocks ⇒ a full-ish tier holds ~28 archives;
LRU churn of ≥2k blocks/iteration under sustained churn is normal once full.

Engine-side diagnostics lines: `CPU-TIER-EVICT n_evicted=… requested=… free_before=…`
(one line per evicting `prepare_store`) — `free_before=0` with small `requested`
= ordinary tier-full LRU, benign when `evicted_total` advances in lockstep.
MTP health: periodic `SpecDecoding metrics:` lines — `Mean acceptance length` and
`Drafted throughput` — decode tok/s = steps/s × mean-acceptance (steps/s is
config-invariant ~61 on GLM kit; acceptance is content+seed dependent 1.7–3.0).

## Server process identification (bare-metal launchers)

- GPU workers show as process names `VLLM::EngineCore`, `VLLM::Worker_TP0..7`.
- `pgrep -f 'vllm.entrypoints'` matches NOTHING on this kit (cmdline is
  `vllm-bin/venv/bin/python vllm-bin/bin/vllm serve …`) — use
  `pgrep -fa 'VLLM::'` or the port: `ss -ltnp | grep :1025`, then `/proc/<pid>/cmdline`.

## Concurrency / contention rules (2026-09-17 lesson)

- The soak's M1 gate assumes it is the only writer besides itself. A single
  concurrent long-context session will: (a) stream 10s-of-GB archives into the
  tier (accelerating LRU eviction of M1), and (b) queue behind/around the marker
  restore in the scheduler (MBT budget shared) — M1 walls then read 25–70 s
  without any bug. Symptoms of contamination: `Waiting: ≥2` / `Deferred: ≥1` in
  engine ticks during "clean" runs; `evicted_total` advancing between EVERY soak
  iteration from the start; M1 times non-monotonic bursty.
- Inference quality is never at risk from this — queueing only affects latency.
  Treat soak timings under concurrent load as *capacity/queue* data, not disease.
- To re-baseline soak numbers: pause other traffic, note tier occupancy start
  (`occupied = allocated − free_list_len`), then compare iter-to-iter.

## Recording results

Append-only TSVs in `logs/` (rows self-stamped ts + config):
`gen-speed-results.tsv` (13 cols), `prefill-sweep.tsv`. Test outputs are ephemeral
(stdout); if you need them logged, tee to `/tmp/<test>-run.log`.

## Porting checklist (per new kit)

1. Update CONFIG block + `BASE`/`MODEL`/payload sizes in each script head.
2. Confirm `max_model_len` > largest payload (+completion margin).
3. Confirm both served aliases or fix scripts' `model` fields.
4. Re-derive expected restore walls (~2 s tier restore at 165k reference) and
   pool-fill iteration count (`GPU_pool_tokens / marker_tokens`).
5. Check whether the script tail assumptions hold on your launcher
   (docker present? bare-metal log path?).
