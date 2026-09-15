#!/usr/bin/env bash
# tune-host-for-hugepages.sh — host preparation for the hugetlbfs-backed KV
# offload CPU tier (engine env VLLM_KV_OFFLOAD_TIER_HUGETLB=1, opt-in via the
# launchers' CPU_TIER_HUGETLB=1). Companion to tune-host-for-tier.sh.
#
# With the tier file on hugetlbfs, its mapping is backed by REAL 2 MiB pages:
# the NVIDIA driver's per-rank pinned page-table budget (~537-600 MB/rank at
# 4 KiB pages — the binding ceiling at >= 576 GiB tier) shrinks ~64x, which
# lets CPU_TIER_GB=800 pin successfully.
#
# What it does (script ONLY — this file changes host state when RUN):
#  1. reads the REAL huge page size from /proc/meminfo Hugepagesize
#     (2048 kB = 2 MiB on x86_64; parsed, never assumed);
#  2. pages = ceil(GiB * 1024 * 1024 kB / Hugepagesize kB);
#  3. applies `sysctl -w vm.nr_hugepages=N` (allocates + charges N huge
#     pages up front — NOT persistent across reboots) with live progress:
#     HugePages_Total is streamed while the kernel works; a silent stretch
#     is page compaction/release, not a hang;
#  4. verifies /proc/meminfo HugePages_Free == N, warning on any deficit
#     (partial allocation happens when free RAM is insufficient; pages
#     already held by a live tier file also count against it);
#  5. checks the hugetlbfs mount (VLLM_KV_OFFLOAD_HUGETLB_DIR, default
#     /dev/hugepages) and mounts it when missing:
#     mount -t hugetlbfs nodev /dev/hugepages;
#  6. reset mode (`--reset`, `reset`, or any second arg `reset`) sets
#     vm.nr_hugepages=0 — the kernel keeps pages that are still in use by a
#     live tier file and warns about the residual.
#
# Usage:
#   bash tune-host-for-hugepages.sh 800         # reserve 800 GiB of huge pages
#   bash tune-host-for-hugepages.sh 800 reset   # reset mode: vm.nr_hugepages=0
#   bash tune-host-for-hugepages.sh --reset     # same
#
# The launcher calls this ONLY when CPU_TIER_HUGETLB=1 (pre-flight, best-
# effort: on failure the engine tier falls back to /dev/shm with a warning).
# Persistence across reboots is opt-in; add to /etc/sysctl.d/99-vllm-hugepages.conf:
#   vm.nr_hugepages = <N>
# (sysctl state set here survives only until the next reboot.)
set -u

run_priv() { # run_priv <command...> — root direct, else passwordless sudo
  if [ "$(id -u)" = 0 ]; then
    "$@"
  elif sudo -n true 2>/dev/null; then
    sudo -n "$@"
  else
    echo "tune-hugepages: no root and no passwordless sudo — cannot run: $*" >&2
    return 1
  fi
}

# meminfo_field <Field> — first value of "<Field>: <value>" in /proc/meminfo.
meminfo_field() {
  awk -v key="$1:" '$1 == key {print $2; exit}' /proc/meminfo 2>/dev/null
}

# sysctl_with_progress <target> — `sysctl -w vm.nr_hugepages=<target>` with live
# progress. The kernel grows/shrinks HugePages_Total incrementally while the
# sysctl write runs, so polling /proc/meminfo in the background shows real
# progress; a long stretch without growth means the kernel is compacting or
# releasing memory — slow but working, not a hang. Returns the sysctl rc.
sysctl_with_progress() {
  _target="$1"
  run_priv sysctl -w "vm.nr_hugepages=$_target" &
  _syspid=$!
  _t0=$SECONDS
  _last=0 _lastchg=0 _beat=0
  while kill -0 "$_syspid" 2>/dev/null; do
    sleep 2
    _cur=$(meminfo_field HugePages_Total); _cur=${_cur:-0}
    _now=$((SECONDS - _t0))
    if [ "$_cur" != "$_last" ]; then
      echo "tune-hugepages:   ... HugePages_Total=$_cur / $_target (${_now}s)"
      _last=$_cur; _lastchg=$_now; _beat=$_now
    elif [ $((_now - _beat)) -ge 10 ]; then
      echo "tune-hugepages:   ... no growth for $((_now - _lastchg))s — kernel still working (compacting/releasing); elapsed ${_now}s"
      _beat=$_now
    fi
  done
  wait "$_syspid"; _rc=$?
  _cur=$(meminfo_field HugePages_Total); _cur=${_cur:-0}
  echo "tune-hugepages:   sysctl finished (rc=$_rc, $((SECONDS - _t0))s): HugePages_Total=$_cur (target was $_target)"
  return "$_rc"
}

RESET=0
case "${1-}" in
  --reset|-r|reset) RESET=1 ;;
esac
case "${2-}" in
  reset|--reset|-r) RESET=1 ;;
esac

# ---------------------------------------------------------------- reset mode
if [ "$RESET" = 1 ]; then
  echo "tune-hugepages: resetting vm.nr_hugepages to 0 (pages still mapped by a live tier file are kept by the kernel; progress below)..."
  if sysctl_with_progress 0; then
    sleep 1
    _total=$(meminfo_field HugePages_Total)
    _total=${_total:-unknown}
    if [ "$_total" = "0" ]; then
      echo "tune-hugepages: vm.nr_hugepages=0 — hugepage pool released (HugePages_Total=0)."
    else
      echo "tune-hugepages: WARNING: HugePages_Total=$_total after reset — residual pages are still mapped/reserved." >&2
      echo "               Free them by wiping stale tier files: sudo rm -f '${VLLM_KV_OFFLOAD_HUGETLB_DIR:-/dev/hugepages}'/vllm_offload_*.mmap, or reboot." >&2
    fi
  else
    echo "tune-hugepages: could not run sysctl (no root and no passwordless sudo) — pool NOT reset." >&2
    exit 1
  fi
  exit 0
fi

# ---------------------------------------------------------------- alloc mode
GIB="${1:-}"
if [ -z "$GIB" ] || ! [ "$GIB" -gt 0 ] 2>/dev/null; then
  echo "usage: bash $0 <GiB> [reset]   |   bash $0 --reset" >&2
  echo "example: sudo bash tune-host-for-hugepages.sh 800   (for CPU_TIER_GB=800 + CPU_TIER_HUGETLB=1)" >&2
  exit 2
fi

PAGE_KB=$(meminfo_field Hugepagesize)
if [ -z "$PAGE_KB" ] || ! [ "$PAGE_KB" -gt 0 ] 2>/dev/null; then
  echo "tune-hugepages: WARNING: Hugepagesize missing from /proc/meminfo — assuming 2048 kB." >&2
  PAGE_KB=2048
fi

# pages = ceil(GiB * 1024 MiB * 1024 kB/MiB / Hugepagesize kB)
PAGES=$(( (GIB * 1024 * 1024 + PAGE_KB - 1) / PAGE_KB ))

echo "tune-hugepages: reserving $PAGES hugepages of $PAGE_KB kB each (${GIB} GiB) — progress below; a few minutes is normal for large sizes."
if ! sysctl_with_progress "$PAGES"; then
  echo "tune-hugepages: WARNING: sysctl vm.nr_hugepages failed — the tier will fall back to /dev/shm (4 KiB pages, ~576 GiB pin ceiling)." >&2
  exit 1
fi

# Verify: the whole pool must exist AND be free (nothing reserved it yet).
_TOTAL=$(meminfo_field HugePages_Total)
_FREE=$(meminfo_field HugePages_Free)
if [ "$_TOTAL" != "$PAGES" ]; then
  echo "tune-hugepages: WARNING: HugePages_Total=$_TOTAL < requested $PAGES — partial allocation (not enough free/contiguous RAM)." >&2
  echo "               Free RAM first (stop other engines; vm.compact_memory=1; TUNE_DROP_CACHES=1) and re-run, or lower the tier size." >&2
elif [ "$_FREE" != "$PAGES" ]; then
  echo "tune-hugepages: WARNING: HugePages_Free=$_FREE != $_TOTAL — $((_TOTAL - _FREE)) pages are already reserved/in-use (stale hugetlbfs tier file? another engine running?)." >&2
  echo "               The engine tier will fall back to /dev/shm unless these are released." >&2
else
  echo "tune-hugepages: verified HugePages_Free=$_FREE == $PAGES pages = ${GIB} GiB of $((PAGE_KB / 1024)) MiB pages."
fi

# ------------------------------------------------------------- mount check
HDIR="${VLLM_KV_OFFLOAD_HUGETLB_DIR:-/dev/hugepages}"
if grep -qs " ${HDIR} hugetlbfs " /proc/mounts; then
  echo "tune-hugepages: hugetlbfs mount present: ${HDIR} ($(awk '$3 == "hugetlbfs" && $2 == "'"$HDIR"'" {print $4; exit}' /proc/mounts))"
else
  echo "tune-hugepages: no hugetlbfs mount at ${HDIR} — mounting (mount -t hugetlbfs nodev ${HDIR})..."
  if run_priv mkdir -p "$HDIR" && run_priv mount -t hugetlbfs -o mode=1777 nodev "$HDIR"; then
    echo "tune-hugepages: hugetlbfs mounted at $HDIR (mode 1777)"
  else
    echo "tune-hugepages: WARNING: could not mount hugetlbfs at $HDIR — the tier will fall back to /dev/shm (4 KiB pages, 576 GiB pin ceiling)." >&2
  fi
fi

# The engine (bare metal: not root) must be able to create its tier file on
# the mount — hugetlbfs mounts default to mode 0755 (root-only), which would
# make the engine's O_CREAT fail with EACCES and fall back to /dev/shm.
# mode=1777 lets any user create the file; the tier file itself stays 0600
# owned by its creator. Also fixes pre-existing systemd mounts (mode 0755).
if run_priv chmod 1777 "$HDIR" 2>/dev/null; then
  echo "tune-hugepages: ${HDIR} mode set 1777 (engine tier file creatable by the engine user)"
else
  echo "tune-hugepages: WARNING: could not chmod 1777 ${HDIR} — a non-root engine may not be able to create its tier file there (fall back to /dev/shm)." >&2
fi

# ------------------------------------------------------------ persistence hint
PERSIST_FILE=/etc/sysctl.d/99-vllm-hugepages.conf
if ! run_priv test -e "$PERSIST_FILE" 2>/dev/null; then
  echo "tune-hugepages: NOTE: vm.nr_hugepages is not persistent across reboots —"
  echo "               to make ${GIB} GiB survive reboots: echo 'vm.nr_hugepages = $PAGES' | sudo tee /etc/sysctl.d/99-vllm-hugepages.conf"
fi
