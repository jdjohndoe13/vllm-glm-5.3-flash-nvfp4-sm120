# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import errno
import mmap
import os
import time
from collections.abc import Callable

import numpy as np
import torch

from vllm.distributed.device_communicators.shm_broadcast import (
    check_shm_free_space,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)

# MADV_HUGEPAGE (Linux THP, value 14): opts this mapping in to transparent
# huge pages. Combined with the host's shmem_enabled=advise setting (set
# by the launcher's tune-host-for-tier.sh), pages faulted afterwards
# materialize as 2 MiB folios; see the MADV_HUGEPAGE block in
# SharedOffloadRegion.__init__.
_MADV_HUGEPAGE = getattr(mmap, "MADV_HUGEPAGE", 14)

# ---------------------------------------------------------------------------
# Opt-in hugetlbfs tier backing (2026-09-15).
#
# VLLM_KV_OFFLOAD_TIER_HUGETLB (truthy int) moves the tier file from
# /dev/shm to a hugetlbfs mount (default /dev/hugepages, override with
# VLLM_KV_OFFLOAD_HUGETLB_DIR), so the mapping is backed by REAL huge
# pages instead of 4 KiB pages. Why: the NVIDIA driver charges per-rank
# pinned page-table space per PAGE (~537-600 MB/rank at 4 KiB pages),
# which fails ("NVRM: failed to allocate page table" / cudaHostRegister
# code=2) at tier sizes >= 576 GiB; with 2 MiB pages that budget shrinks
# ~64x and an 800 GiB tier stops binding. Transparent huge pages cannot
# provide this on /dev/shm (shmem refuses 2 MiB folios, verified
# 2026-09-13), hence the dedicated hugetlbfs mount with pages reserved
# via vm.nr_hugepages (tune-host-for-hugepages.sh).
#
# The flag is parsed per SharedOffloadRegion instance (and once at import
# for BLOCK_SIZE_ALIGNMENT); it defaults OFF, and every failure on the
# hugetlb path falls back to the stock /dev/shm flow with a loud warning.
# ---------------------------------------------------------------------------
_HUGETLB_ENV = "VLLM_KV_OFFLOAD_TIER_HUGETLB"
_HUGETLB_DIR_ENV = "VLLM_KV_OFFLOAD_HUGETLB_DIR"
_HUGETLB_DEFAULT_DIR = "/dev/hugepages"

# Suffix multipliers for /proc/meminfo size fields (Hugepagesize prints
# e.g. "2048 kB").
_MEMINFO_UNIT_MULTIPLIERS = {
    "b": 1,
    "kb": 1024,
    "k": 1024,
    "mb": 1024**2,
    "m": 1024**2,
    "gb": 1024**3,
    "g": 1024**3,
}


class _HugeTlbFallback(Exception):
    """Internal: the hugetlbfs tier path failed validation — fall back."""


def _hugetlb_tier_requested() -> bool:
    """Whether VLLM_KV_OFFLOAD_TIER_HUGETLB (truthy int) opts into hugetlbfs."""
    raw = os.environ.get(_HUGETLB_ENV)
    if raw is None or not raw.strip():
        return False
    raw = raw.strip()
    try:
        return int(raw) != 0
    except ValueError:
        return raw.lower() in ("true", "yes", "on")


def _hugetlb_dir() -> str:
    """Tier mount point: VLLM_KV_OFFLOAD_HUGETLB_DIR or /dev/hugepages."""
    return os.environ.get(_HUGETLB_DIR_ENV, "").strip() or _HUGETLB_DEFAULT_DIR


def _is_hugetlbfs_mount(mount_dir: str) -> bool:
    """True when /proc/mounts shows mount_dir as a hugetlbfs mount."""
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            mounts = f.read()
    except OSError:
        return False
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == mount_dir and fields[2] == "hugetlbfs":
            return True
    return False


def _read_meminfo_hugepages() -> tuple[int, int] | None:
    """(huge_page_size_bytes, HugePages_Free) parsed from /proc/meminfo.

    The page size is parsed from the Hugepagesize line (usually
    "2048 kB"), never assumed. Returns None when either field is missing
    or unparseable.
    """
    page_size = None
    free = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        if page_size is None and line.startswith("Hugepagesize:"):
            try:
                value, unit = line.split(":", 1)[1].split()
            except ValueError:
                continue
            multiplier = _MEMINFO_UNIT_MULTIPLIERS.get(unit.lower())
            if multiplier is None:
                continue
            try:
                page_size = int(value) * multiplier
            except ValueError:
                continue
        elif free is None and line.startswith("HugePages_Free:"):
            try:
                free = int(line.split(":", 1)[1])
            except ValueError:
                return None
    if page_size is None or free is None or page_size <= 0:
        return None
    return page_size, free


def _hugetlb_page_size(mount_dir: str) -> int | None:
    """REAL hugetlbfs page size (os.statvfs f_bsize) for the mount, else None."""
    if not _is_hugetlbfs_mount(mount_dir):
        return None
    try:
        f_bsize = os.statvfs(mount_dir).f_bsize
    except OSError:
        return None
    if f_bsize <= 0 or f_bsize % mmap.PAGESIZE != 0:
        return None
    return f_bsize


def _hugetlb_block_alignment() -> int:
    """BLOCK_SIZE_ALIGNMENT: the alignment CPUOffloadingSpec rounds every
    offload block chunk up to (kv_bytes_per_block), before the region is
    built. Stock /dev/shm tier: 4 KiB (mmap.PAGESIZE). With the opt-in
    hugetlbfs backing: the REAL hugetlbfs page size, so block rows — and
    therefore total_size_bytes — are huge-page multiples by construction
    (the creator-side BLOCK_SIZE_ALIGNMENT sanity below holds without
    shrinking the tier). When the env is off, or the mount is not (yet)
    a hugetlbfs mount, this stays mmap.PAGESIZE (stock behavior).
    """
    if not _hugetlb_tier_requested():
        return mmap.PAGESIZE
    f_bsize = _hugetlb_page_size(_hugetlb_dir())
    if f_bsize is None:
        return mmap.PAGESIZE
    return f_bsize


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


def _madvise_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    mmap_obj.madvise(_MADV_POPULATE_WRITE, offset, length)


def _fallback_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    # Touch one byte per page via a read-modify-write so existing bytes are
    # preserved — a peer worker may have already written KV data into this
    # shared mmap by the time we run on a kernel without MADV_POPULATE_WRITE.
    arr = np.frombuffer(mmap_obj, dtype=np.uint8)
    arr[offset : offset + length : mmap.PAGESIZE] |= 0


def _get_populate_write_fn(
    mmap_obj: mmap.mmap,
) -> Callable[[mmap.mmap, int, int], None]:
    """Select the pre-faulting method once for this mmap."""
    try:
        _madvise_populate_write(mmap_obj, 0, mmap.PAGESIZE)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        logger.warning(
            "MADV_POPULATE_WRITE is not supported; falling back to per-page "
            "writes for mmap pre-population. Startup may be slower."
        )
        return _fallback_populate_write
    return _madvise_populate_write


class SharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap — or
    {VLLM_KV_OFFLOAD_HUGETLB_DIR}/vllm_offload_{engine_id}.mmap when
    VLLM_KV_OFFLOAD_TIER_HUGETLB opts the tier into hugetlbfs backing
    (real 2 MiB pages; see the hugetlbfs block near the top of this file).
    """

    BLOCK_SIZE_ALIGNMENT: int = _hugetlb_block_alignment()

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.huge_tlb = False  # True when the tier file is backed by hugetlbfs
        self.hugetlb_page_size: int = 0  # REAL hugetlbfs page size when huge_tlb
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each block row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        # Opt-in hugetlbfs backing (default OFF): when VLLM_KV_OFFLOAD_TIER_HUGETLB
        # is truthy, try to create/join the tier file on the hugetlbfs mount
        # (creator validates mount/page-size/hugepage budget and ftruncates;
        # joiners follow the file). On ANY failure up to the file being live
        # it unlinks its own stub, logs a loud warning and returns False —
        # the stock /dev/shm flow below then runs with identical semantics.
        _hugetlb_ready = (
            _hugetlb_tier_requested() and self._open_hugetlb_region(engine_id)
        )
        if not _hugetlb_ready:
            try:
                self.fd: int | None = os.open(
                    self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
            except FileExistsError:
                # Joiner path — another worker won O_EXCL. Reopen and wait
                # for the file to reach expected size.
                self.fd = os.open(self.mmap_path, os.O_RDWR)
                try:
                    _wait_for_file_size(self.fd, self.total_size_bytes)
                except (TimeoutError, OSError):
                    os.close(self.fd)
                    raise
                logger.info("Opened existing mmap file %s", self.mmap_path)
            else:
                # Creator path. We won O_EXCL, so we own the file: any
                # failure here must clean up so concurrent joiners don't
                # land on a 0-byte stub and spin in _wait_for_file_size
                # for the full 30 s timeout.
                try:
                    check_shm_free_space(self.total_size_bytes)
                    os.ftruncate(self.fd, self.total_size_bytes)
                except (RuntimeError, OSError):
                    os.unlink(self.mmap_path)
                    os.close(self.fd)
                    raise
                self._creator = True
                logger.info(
                    "Created mmap file %s (%.2f GB)",
                    self.mmap_path,
                    self.total_size_bytes / 1e9,
                )

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # Prefer 2 MiB huge pages for the tier. With shmem THP enabled on
        # the host (shmem_enabled=advise, set by tune-host-for-tier.sh),
        # pages faulted after this MADV_HUGEPAGE materialize as PMD
        # (2 MiB) folios instead of 4 KiB pages, which speeds up tier
        # init. Verified tier-size ceiling: cudaHostRegister failure is
        # NOT RAM/IOMMU/BAR1 bound — the RM's nvos_create_alloc()
        # kvzallocs a 16 B-per-4-KiB-page table per call, capped at
        # INT_MAX (~512 GiB of pages per call; drivers 580-610 affected,
        # <=575 unaffected, >=615.71 fixed) — so pin_mmap_region now
        # registers in chunks (default 64 GiB) to lift it. Best-effort:
        # silently stays on 4 KiB pages when huge pages are unavailable.
        # MUST come before the MADV_POPULATE_WRITE pre-fault below so the
        # faults allocate huge folios.
        if self.huge_tlb:
            # hugetlbfs tier: the pages are huge by construction, so
            # madvise(MADV_HUGEPAGE) is at best a no-op on a hugetlbfs
            # mapping — skip it entirely. The MADV_POPULATE_WRITE
            # pre-fault below still runs (it pre-faults the real huge
            # pages that back this file).
            logger.debug(
                "hugetlbfs tier: MADV_HUGEPAGE skipped (%d-byte pages by "
                "construction)",
                self.hugetlb_page_size,
            )
        else:
            _t0 = time.perf_counter()
            try:
                self.mmap_obj.madvise(_MADV_HUGEPAGE, 0, self.total_size_bytes)
            except (OSError, ValueError):
                logger.warning(
                    "MADV_HUGEPAGE rejected; tier stays on 4 KiB pages "
                    "(host shmem_enabled must include 'advise' for 2 MiB "
                    "pages)."
                )
            logger.debug(
                "MADV_HUGEPAGE over %d bytes: %.3f s",
                self.total_size_bytes,
                time.perf_counter() - _t0,
            )

        populate_write_fn = _get_populate_write_fn(self.mmap_obj)

        if rank is not None:
            # Populate only this worker's pages (one slot per block row).
            worker_offset = rank * cpu_page_size
            _t0 = time.perf_counter()
            page_size = self.page_size
            for block in range(num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                populate_write_fn(self.mmap_obj, aligned_offset, aligned_length)
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
                num_blocks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            populate_write_fn(self.mmap_obj, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self._canonical_offset = 0
        self.is_pinned: bool = False
        # (ptr, size) tuples set by gpu_worker.pin_mmap_region chunked registration; unwound by cleanup().
        self.registered_chunks: list = []

    def _open_hugetlb_region(self, engine_id: str) -> bool:
        """Try to back the tier with a hugetlbfs file (opt-in, default OFF).

        Returns True when self.fd is open on
        {hugetlb_dir}/vllm_offload_{engine_id}.mmap and the file already
        holds self.total_size_bytes (creator truncated it / joiner waited
        for it); self.huge_tlb is then True and self.mmap_path points at
        the hugetlbfs file. Returns False to run the stock /dev/shm flow
        (loud warning logged; the caller keeps self.mmap_path on /dev/shm).

        Same creator/joiner protocol as the stock path, with the stock
        cleanup contract: whoever wins O_EXCL owns the stub and must unlink
        it on failure so concurrent joiners don't spin 30 s on it.

        Only the CREATOR validates the backing (mount type, real page size
        via os.statvfs(f_bsize), HugePages_Free budget, BLOCK_SIZE_ALIGNMENT
        sanity). Joiners never re-check the budget: after the creator's
        ftruncate, HugePages_Free has already dropped by the tier's page
        count, so a joiner-side check would spuriously fail and split ranks
        across two backing files — the file itself is the joiner's proof
        that the creator validated everything.
        """
        hugetlb_dir = _hugetlb_dir()
        hugetlb_path = os.path.join(hugetlb_dir, f"vllm_offload_{engine_id}.mmap")
        try:
            self.fd = os.open(hugetlb_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            # Joiner path — another worker won O_EXCL on the hugetlbfs
            # file. Same protocol as the stock joiner: reopen + wait.
            try:
                self.fd = os.open(hugetlb_path, os.O_RDWR)
            except OSError as e:
                # The file vanished between the O_EXCL race and this
                # reopen: its creator failed validation and unlinked the
                # stub (and fell back itself), so no live hugetlb region
                # exists — safe to fall back too.
                return self._hugetlb_fallback(
                    f"tier file vanished while joining: {e}"
                )
            try:
                _wait_for_file_size(self.fd, self.total_size_bytes)
            except (TimeoutError, OSError):
                os.close(self.fd)
                if not os.path.exists(hugetlb_path):
                    # The stub was abandoned (its creator fell back) — no
                    # live hugetlb region exists; safe to fall back.
                    return self._hugetlb_fallback(
                        "creator abandoned the hugetlbfs tier stub "
                        "(file gone after the 30 s size wait)"
                    )
                # A live creator may still exist behind this file; falling
                # back would split ranks across two backing files. Keep the
                # stock joiner semantics instead: the wait failure is fatal.
                raise
            self.huge_tlb = True
            self.mmap_path = hugetlb_path
            self.hugetlb_page_size = _hugetlb_page_size(hugetlb_dir) or 0
            logger.info("Opened existing mmap file %s", self.mmap_path)
            return True
        else:
            # Creator path. We won O_EXCL, so we own the 0-byte stub: any
            # failure here must clean up (unlink + close) so concurrent
            # joiners don't spin in _wait_for_file_size for the full 30 s
            # timeout, then fall back to /dev/shm.
            try:
                self.hugetlb_page_size = self._validate_hugetlb_backing(hugetlb_dir)
            except _HugeTlbFallback as e:
                self._discard_hugetlb_stub(hugetlb_path)
                return self._hugetlb_fallback(str(e))
            # Round the ftruncate target UP to a huge-page multiple:
            # hugetlbfs i_size rounds to huge units (unaligned sizes can
            # EINVAL on ftruncate) and the hugepages charged must cover the
            # whole tier. With the BLOCK_SIZE_ALIGNMENT sanity above this
            # is already aligned; the rounding stays defensive. The mmap
            # below keeps mapping length = self.total_size_bytes.
            ftruncate_size = (
                (self.total_size_bytes + self.hugetlb_page_size - 1)
                // self.hugetlb_page_size
                * self.hugetlb_page_size
            )
            try:
                # check_shm_free_space is intentionally NOT called: it
                # checks /dev/shm, which does not back this file. The
                # tier budget was verified against /proc/meminfo
                # (HugePages_Free) instead.
                os.ftruncate(self.fd, ftruncate_size)
            except OSError as e:
                self._discard_hugetlb_stub(hugetlb_path)
                return self._hugetlb_fallback(
                    f"ftruncate({hugetlb_path}, {ftruncate_size}) failed: {e}"
                )
            self._creator = True
            self.huge_tlb = True
            self.mmap_path = hugetlb_path
            logger.info(
                "Created hugetlbfs tier file %s (%.2f GB, %d MiB pages)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
                self.hugetlb_page_size // (1024 * 1024),
            )
            return True

    def _validate_hugetlb_backing(self, hugetlb_dir: str) -> int:
        """Creator-side validation of the hugetlbfs backing.

        Returns the REAL hugetlbfs page size (os.statvfs(f_bsize) on the
        mount). Raises _HugeTlbFallback with the reason on any failure.
        """
        if not _is_hugetlbfs_mount(hugetlb_dir):
            raise _HugeTlbFallback(
                f"{hugetlb_dir} is not a hugetlbfs mount (see /proc/mounts; "
                "tune-host-for-hugepages.sh mounts it)"
            )
        try:
            f_bsize = os.statvfs(hugetlb_dir).f_bsize
        except OSError as e:
            raise _HugeTlbFallback(f"statvfs({hugetlb_dir}) failed: {e}") from e
        if f_bsize <= 0 or f_bsize % mmap.PAGESIZE != 0:
            raise _HugeTlbFallback(
                f"unexpected hugetlbfs f_bsize {f_bsize} on {hugetlb_dir}"
            )
        meminfo = _read_meminfo_hugepages()
        if meminfo is None:
            raise _HugeTlbFallback(
                "could not parse Hugepagesize/HugePages_Free from /proc/meminfo"
            )
        meminfo_page_size, hugepages_free = meminfo
        if meminfo_page_size != f_bsize:
            raise _HugeTlbFallback(
                f"hugetlbfs page size {f_bsize} B != /proc/meminfo Hugepagesize "
                f"{meminfo_page_size} B — the sysctl vm.nr_hugepages pool "
                "would not back this mount"
            )
        if self._row_stride % f_bsize != 0:
            raise _HugeTlbFallback(
                f"kv_bytes_per_block {self._row_stride} is not a multiple of "
                f"the hugetlbfs page size {f_bsize} (BLOCK_SIZE_ALIGNMENT "
                "sanity)"
            )
        needed_pages = (self.total_size_bytes + f_bsize - 1) // f_bsize
        if hugepages_free < needed_pages:
            raise _HugeTlbFallback(
                f"HugePages_Free {hugepages_free} < {needed_pages} needed for "
                f"{self.total_size_bytes} bytes at {f_bsize}-byte pages — "
                "run (kit dir): sudo bash tune-host-for-hugepages.sh "
                f"{(self.total_size_bytes + (1 << 30) - 1) >> 30}"
            )
        return f_bsize

    def _discard_hugetlb_stub(self, hugetlb_path: str) -> None:
        """Unlink the just-created 0-byte stub and close the fd (creator
        failure contract: concurrent joiners must not spin 30 s on it)."""
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            os.unlink(hugetlb_path)
        except OSError as e:
            logger.warning(
                "Failed to unlink hugetlbfs tier stub %s: %s", hugetlb_path, e
            )

    def _hugetlb_fallback(self, reason: str) -> bool:
        """Log the loud fallback warning; tell the caller to run /dev/shm."""
        logger.warning(
            "hugetlbfs tier backing unavailable — falling back to the stock "
            "/dev/shm tier file (%s)",
            reason,
        )
        return False

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_blocks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per block for this  tensor.
        """
        assert self.rank is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_next_canonical_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view shared by all workers for one
        canonical tensor (canonical layout).

        Must be called once per canonical tensor, instead of
        create_next_worker_view. The full mmap layout is:

            |<-------- canonical area ------->|<-------- unused ------->|
            |  all workers share this area    |                         |
            |                                 |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            ^                ^
            _canonical_offset=0, then advances by each tensor's size

        Each canonical_t{i} cell is that tensor's canonical page for the
        block. Canonical areas are carved consecutively from the start of
        each block row; consecutive rows are separated by row_stride. Every
        worker gets the identical byte ranges and writes only its disjoint
        bytes within them, as described by its canonical mappings — unlike
        create_next_worker_view, which gives each worker a private
        cpu_page_size slot per row.

        The trailing unused bytes exist only when the canonical pages sum to
        less than row_stride: page-alignment padding of the row, or
        deduplication of KV replicated across workers (e.g. the MLA latent),
        where one canonical copy replaces world_size worker copies.

        Args:
            tensor_page_size: Canonical bytes per block for this tensor.
        """
        new_offset = self._canonical_offset + tensor_page_size
        assert new_offset <= self._row_stride
        view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._canonical_offset,
        )
        self._canonical_offset = new_offset
        self._views.append(view)
        return view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            if current_platform.is_cuda_alike():
                if self.registered_chunks:
                    # Unwind the chunked registration in reverse order.
                    num_chunks = len(self.registered_chunks)
                    for i in range(num_chunks - 1, -1, -1):
                        ptr, _ = self.registered_chunks[i]
                        result = torch.cuda.cudart().cudaHostUnregister(ptr)
                        if result.value != 0:
                            logger.warning(
                                "cudaHostUnregister failed for rank=%d "
                                "chunk=%d/%d (code=%d)",
                                self.rank,
                                i + 1,
                                num_chunks,
                                result,
                            )
                else:
                    # Legacy defensive path (chunks list empty): unregister
                    # the whole region with a single call.
                    base_ptr = self._base.data_ptr()
                    result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
                    if result.value != 0:
                        logger.warning(
                            "cudaHostUnregister failed for rank=%d (code=%d)",
                            self.rank,
                            result,
                        )
            self.registered_chunks = []
            self.is_pinned = False
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
