#!/usr/bin/env bash
# tune-host-for-tier.sh — host preparation for the KV offload tier.
# Invoked automatically by the launchers right before `docker run`; also
# safe to run standalone. Every step is best-effort (skipped, not fatal,
# when passwordless sudo is unavailable).
#
# Why (measured 2026-09-13 on the failed 800-GiB tier attempt):
#  1. cudaHostRegister pins the whole tier by walking EVERY page. At 4 KiB
#     that is ~210M pages for 800 GiB (init crawled at ~140 GB/min). With
#     2 MiB tier pages it is ~409600 folios: init collapses to seconds.
#     The 2 MiB pages come from the patched shared_offload_region.py
#     (MADV_HUGEPAGE before the populate pre-fault) and only work when the
#     host's shmem_enabled includes "advise" — set below.
#  2. A large 4 KiB tier physically fragments host RAM (its pages scatter
#     across every zone) and the NVIDIA driver then fails to allocate its
#     DMA page tables: dmesg "NVRM: failed to allocate page table",
#     cudaHostRegister -> code=2 on all 8 ranks, tier left UNPINNED.
#     2 MiB pages shrink that page-table footprint ~64x, and synchronous
#     memory compaction (vm.compact_memory) before launch consolidates the
#     free memory into the contiguous runs the driver's allocator needs.
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
#     Other tmpfs users are unaffected by "advise" mode.
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
