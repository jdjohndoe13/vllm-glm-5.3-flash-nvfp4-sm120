#!/usr/bin/env bash
# tune-host-for-tier.sh — host preparation for the KV offload tier.
# Invoked automatically by the launchers right before `docker run`; also
# safe to run standalone. Every step is best-effort (skipped, not fatal,
# when passwordless sudo is unavailable).
#
# What it does:
#  1. vm.compact_memory: synchronously consolidates free RAM before the
#     engine boots (best-effort pre-boot hygiene).
#  2. shmem_enabled=advise lets tmpfs files that request huge pages
#     (MADV_HUGEPAGE in the patched shared_offload_region.py) get 2 MiB
#     folios. On current kernels this is inert — shmem refuses 2 MiB
#     folios in every shmem_enabled mode (verified 2026-09-13 on
#     6.17.0-20-generic with a 7-variant test matrix), so the tier runs
#     4 KiB pages there; the setting is kept so a future kernel with
#     shmem-THP support yields 2 MiB tier pages automatically.
#
# Tier-size note: the NVIDIA driver rejects pinning above its per-context
# page-table budget (512 GiB works; 576 GiB and above fail with
# "NVRM: failed to allocate page table" on all ranks, independent of RAM
# state — bisected 2026-09-13). Host tuning does not change that ceiling.
#
# Optional: TUNE_DROP_CACHES=1 also drops page cache before compaction
# (frees more RAM, but the next boot re-reads the ~198 GB model from SSD).
set -u

run_priv() { # run_priv <command...> — root direct, else passwordless sudo
  if [ "$(id -u)" = 0 ]; then
    "$@"
  elif sudo -n true 2>/dev/null; then
    sudo -n "$@"
  else
    echo "tune-host: no root and no passwordless sudo — skipping: $*" >&2
    return 1
  fi
}

# (a) Allow huge pages for tmpfs files that ask for them (MADV_HUGEPAGE).
#     Inert on current kernels (shmem refuses 2 MiB folios — see header);
#     kept future-proof. Other tmpfs users are unaffected by "advise" mode.
SHMEM_THP=/sys/kernel/mm/transparent_hugepage/shmem_enabled
if [ -e "$SHMEM_THP" ]; then
  cur=$(cat "$SHMEM_THP")
  case "$cur" in
    *"[advise]"*|*"[always]"*|*"[within_size]"*) : ;; # already huge-enabled
    *)
      run_priv sh -c "echo advise > '$SHMEM_THP'" || true
      ;;
  esac
  echo "tune-host: shmem_enabled = $(cat "$SHMEM_THP")"
else
  echo "tune-host: shmem THP sysctl not present (old kernel?) — tier stays 4 KiB." >&2
fi

# (b) Defragment host RAM: consolidate free memory into contiguous runs.
if [ "${TUNE_DROP_CACHES:-0}" = 1 ]; then
  echo "tune-host: dropping page caches (TUNE_DROP_CACHES=1)..."
  run_priv sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
fi
echo "tune-host: compacting host memory (vm.compact_memory)..."
run_priv sh -c 'echo 1 > /proc/sys/vm/compact_memory' || true
free -g | sed -n '1,2p'
