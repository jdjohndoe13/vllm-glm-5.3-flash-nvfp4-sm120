# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import functools
import json
import os
import re
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalPageMapping,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# NOTE (kit patch, 2026-09-16 — crash-3/#4 instrumented swap path): every
# swap_blocks_batch submission is capped to _SWAP_CAP descriptors (env
# VLLM_KV_OFFLOAD_SWAP_BATCH_CAP, legacy VLLM_KV_OFFLOAD_MAX_BATCH_
# DESCRIPTORS, default 32; 1 = per-entry everywhere). Rationale: on RTX 5090
# + driver 590.48.01 the deployed _C_stable_libtorch.abi3.so batch op dies
# with "cuMemcpyBatchAsync failed at index N with error 1"
# (CUDA_ERROR_INVALID_VALUE) on some GPU->CPU preemption/eviction-flush
# batches (crash #3: index 34 of a 560-entry batch; crash #4: index 7 INSIDE
# a <=32 chunk — batch size alone is not the discriminator; small <=7-entry
# evictions and 560-entry CPU->GPU loads always passed). Upstream added an
# in-op env knob (same env name) in a LATER build than ours, so we own the
# chunk loop here. On any driver rejection we: probe the failed chunk
# item-by-item with the same op at n=1 to identify the first defective
# descriptor, log everything to /tmp/vllm_swap_diag/log_swap_T<pid>.jsonl
# (env VLLM_KV_OFFLOAD_DIAG_DIR), then complete the failed span per-entry
# via libcuda.so.1 cuMemcpyAsync on the CURRENT stream (the driver infers
# HtoD/DtoH from pointer kinds — the same fallback class the op itself uses
# and what vllm#49276 verified bit-exact). After the first recovery this
# handler goes sticky-per-entry: every later call is submitted descriptor by
# descriptor (~3 ms extra per 560-item flush, bandwidth still PCIe-bound) so
# the engine SURVIVES flushes while diagnostics accumulate. If a per-entry
# copy itself fails, the defect is item-level data: the exact
# pointers/size/stream are logged and the exception re-raised.
# NOTE (kit patch, 2026-09-16 #2): crash #5 reproduced IDENTICALLY on
# driver 615.71.09 (same n=78 flush, first chunk, item 6, probe-at-n=1 and
# per-entry cuMemcpyAsync all rejected with error 1) — confirms the defect
# is item-level data, not the driver/batch path. Stock (unpatched) -orig
# died the same way. The poison descriptor MOVES with the job (item 56/78
# boot 062647, item 6/78 boot 072727; size 542720 vs 671744). /tmp is
# tmpfs and was wiped by reboots twice, so the default diag dir moved to
# persistent storage; handler geometry is now also mirrored to the engine
# log, and the failing item is classified at failure time (in-tier?
# in-range? pointer mem-type?) so the next boot yields the root cause
# without relying on JSONL surviving.
_SWAP_CAP = 32
for _cap_env in (
    os.environ.get("VLLM_KV_OFFLOAD_SWAP_BATCH_CAP"),
    os.environ.get("VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS"),
):
    if _cap_env:
        try:
            _cap_parsed = int(_cap_env)
        except ValueError:
            _cap_parsed = 0
        if 1 <= _cap_parsed <= 1024:
            _SWAP_CAP = _cap_parsed
            break
_RECOVER_RE = re.compile(r"failed at index (\d+) with error (\d+)")
# NOTE: /tmp is tmpfs on this host — wiped by reboots, which destroyed the
# decode data twice. Default to persistent storage; env override still wins.
_DIAG_DIR = os.environ.get(
    "VLLM_KV_OFFLOAD_DIAG_DIR",
    "/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/swap_diag",
)
_PROBE_MAX = 48
_CU_LIB = None


def _get_cu_lib():
    """Lazy libcuda handle with all signatures the diag path needs."""
    global _CU_LIB
    if _CU_LIB is None:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuMemcpyAsync.restype = ctypes.c_int
        lib.cuMemcpyAsync.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.cuMemcpyDtoHAsync_v2.restype = ctypes.c_int
        lib.cuMemcpyDtoHAsync_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.cuMemcpyHtoDAsync_v2.restype = ctypes.c_int
        lib.cuMemcpyHtoDAsync_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.cuPointerGetAttribute.restype = ctypes.c_int
        lib.cuPointerGetAttribute.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.cuStreamSynchronize.restype = ctypes.c_int
        lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        _CU_LIB = lib
    return _CU_LIB


def _extent_bytes(t) -> int:
    """True addressable span of a tier view: row-interleaved slot views are
    STRIDED (stride(0) = material-row pitch), so numel*element_size understates
    them by ~8x. 1-D/0-D tensors degrade to numel exactly."""
    if t.numel() == 0:
        return 0
    if t.dim() <= 1:
        return t.numel() * t.element_size()
    stride0_b = int(t.stride(0)) * t.element_size()
    row_b = int(t[0].numel()) * t.element_size()
    return (int(t.shape[0]) - 1) * stride0_b + row_b


class _SwapDiag:
    """Cap + probe + per-entry-fallback instrumentation around the batch op.

    Never lets its own diagnostics crash the worker; never trusts the
    partially-enqueued contents of a failed batch (the failed span is
    rewritten idempotently per-entry).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sticky = False
        self._handlers: list[dict] = []

    # -- logging (best effort) -------------------------------------------
    def _log(self, event: str, **kw) -> None:
        try:
            line = json.dumps(
                {"ev": event, "t": round(time.time(), 6), **kw}) + "\n"
            with self._lock:
                os.makedirs(_DIAG_DIR, exist_ok=True)
                with open(os.path.join(
                        _DIAG_DIR,
                        "log_swap_T%s.jsonl" % os.getpid()), "a") as fh:
                    fh.write(line)
        except Exception:
            pass
        try:
            if event in ("batch_rejected", "probe_result",
                         "per_entry_failed"):
                logger.warning("swap_diag %s %s", event, kw)
        except Exception:
            pass

    def register(self, handler) -> None:
        """Log handler geometry once so failing items can be decoded
        offline (event ptr - base -> byte offset -> block ids)."""
        try:
            info = {
                "gpu_to_cpu": handler.gpu_to_cpu,
                "blocks_per_chunk": handler.dst_blocks_per_chunk,
                "src": [int(t.data_ptr()) for t in handler.src_tensors],
                "dst": [int(t.data_ptr()) for t in handler.dst_tensors],
                "src_nbytes": [t.numel() * t.element_size()
                               for t in handler.src_tensors],
                "dst_nbytes": [t.numel() * t.element_size()
                               for t in handler.dst_tensors],
                # Stride-aware classification fields: the tier layer views
                # are row-interleaved STRIDED views whose true addressable
                # span is (rows-1)*stride0 + row_bytes, NOT numel.
                "src_extent": [_extent_bytes(t) for t in
                               handler.src_tensors],
                "dst_extent": [_extent_bytes(t) for t in
                               handler.dst_tensors],
                "src_stride0": [int(t.stride(0)) * t.element_size()
                                for t in handler.src_tensors],
                "dst_stride0": [int(t.stride(0)) * t.element_size()
                                for t in handler.dst_tensors],
                "rows": int(handler.src_tensors[0].shape[0]),
                "row_bytes": int(handler.src_tensors[0].stride(0)),
            }
            self._handlers.append(info)
            self._log("handler_init", **info)
            # Mirror the geometry to the engine log so the decode survives
            # even if the JSONL write path is unavailable (tmpfs, disk, etc).
            try:
                logger.info(
                    "swap_diag handler_init gpu_to_cpu=%s bpc=%s rows=%s "
                    "row_bytes=%s dst_bases=%s src_bases=%s",
                    info["gpu_to_cpu"], info["blocks_per_chunk"],
                    info["rows"], info["row_bytes"],
                    [hex(b) for b in info["dst"]],
                    [hex(b) for b in info["src"]])
            except Exception:
                pass
        except Exception:
            # NEVER swallow: a broken register() silently disables
            # classification (in_tier=False false negative) — crash loudly
            # at handler init instead.
            logger.exception("swap_diag register failed")
            raise

    @staticmethod
    def _mem_type(ptr: int) -> int:
        """CU_MEMORYTYPE_* via cuPointerGetAttribute; 0 on any failure."""
        try:
            lib = _get_cu_lib()
            val = ctypes.c_uint64(0)
            # CU_POINTER_ATTRIBUTE_MEMORY_TYPE == 2
            rc = lib.cuPointerGetAttribute(
                ctypes.byref(val), 2, ctypes.c_uint64(ptr))
            return int(val.value) if rc == 0 else 0
        except Exception:
            return 0

    # CU_POINTER_ATTRIBUTE_RANGE_START_ADDR / _RANGE_SIZE (registered-range
    # witnesses for a single pointer).
    _ATTR_RANGE_START = 11
    _ATTR_RANGE_SIZE = 12

    def _probe_registration(self, cls: dict, sp: int, dp: int, sz: int,
                            item: int, tag: str) -> dict:
        """Witness which cudaHostRegister'd range(s) host the failing span.

        If the first and the last byte of the HOST-side span report
        DIFFERENT range start addresses, the descriptor straddles an
        internal registration seam — which is exactly what the driver
        rejects with CUDA_ERROR_INVALID_VALUE (crash-#7 root cause). The
        verdict fields: split_ranges, boundary, bytes_before/bytes_past.
        """
        out: dict = {}
        try:
            host_is_dst = cls.get("dst_mt") == 1
            hp = dp if host_is_dst else sp
            lib = _get_cu_lib()
            for pos, ptr in (("first", hp), ("last", hp + sz - 1)):
                v = ctypes.c_uint64(0)
                rc = lib.cuPointerGetAttribute(
                    ctypes.byref(v), self._ATTR_RANGE_START,
                    ctypes.c_uint64(ptr))
                out[f"{pos}_rc"] = rc
                if rc == 0:
                    out[f"{pos}_start"] = int(v.value)
            if ("first_start" in out and "last_start" in out):
                out["split_ranges"] = out["first_start"] != out["last_start"]
                if out["split_ranges"]:
                    b = out["last_start"]
                    out["boundary"] = b
                    out["bytes_before_boundary"] = b - hp
                    out["bytes_past_boundary"] = (hp + sz) - b
        except Exception as exc:
            out["probe_error"] = str(exc)[:200]
        self._log("probe_registration", tag=tag, item=item,
                  src_ptr=sp, dst_ptr=dp, size=sz, **out)
        return out

    def _split_retry(self, cls: dict, sp: int, dp: int, sz: int,
                     boundary: int, stream: int, item: int,
                     tag: str) -> bool:
        """Split the rejected descriptor AT the registration seam and copy
        both halves. Success proves the straddle cause behaviorally AND
        completes the transfer that the batch op could not."""
        host_is_dst = cls.get("dst_mt") == 1
        s1 = boundary - (dp if host_is_dst else sp)
        s2 = sz - s1
        if s1 <= 0 or s2 <= 0:
            self._log("split_retry", tag=tag, item=item, ok=False,
                      reason="bad_split", s1=s1, s2=s2)
            return False
        self._log("split_retry_begin", tag=tag, item=item, boundary=boundary,
                  part_a=s1, part_b=s2)
        if host_is_dst:
            rc1 = self._cu_memcpy(dp, sp, s1, stream)
            rc2 = self._cu_memcpy(boundary, sp + s1, s2, stream)
        else:
            rc1 = self._cu_memcpy(dp, sp, s1, stream)
            rc2 = self._cu_memcpy(dp + s1, boundary, s2, stream)
        ok = rc1 == 0 and rc2 == 0
        if ok:
            self._stream_sync(stream)
        self._log("split_retry", tag=tag, item=item, ok=ok, rc_a=rc1,
                  rc_b=rc2, boundary=boundary)
        return ok

    def _stream_sync(self, stream: int) -> None:
        """Make a recovered (async, hand-issued) copy durable before the
        caller's success path proceeds."""
        try:
            lib = _get_cu_lib()
            lib.cuStreamSynchronize(ctypes.c_void_p(stream))
        except Exception:
            pass

    def _classify(self, src_ptr: int, dst_ptr: int, size: int) -> dict:
        """Decode a poison descriptor against known handler geometry.

        In-tier (store): dst inside a pinned tier view, not past its
        stride-aware end. In-tier (load): src likewise. A descriptor that is
        in-tier yet still rejected rules out producer index math and points
        at the driver/registration; one that is out-of-tier is a producer
        bug. For strided row-interleaved views we additionally decode
        material row / layer cell / intra-row offset.
        """
        out = {"src_mt": self._mem_type(src_ptr),
               "dst_mt": self._mem_type(dst_ptr)}
        for h in self._handlers:
            if h["gpu_to_cpu"]:
                tier_bases = h["dst"]
                tier_extents = h.get("dst_extent") or h["dst_nbytes"]
                tier_strides = h.get("dst_stride0") or []
                tier_ptr = dst_ptr
            else:
                tier_bases = h["src"]
                tier_extents = h.get("src_extent") or h["src_nbytes"]
                tier_strides = h.get("src_stride0") or []
                tier_ptr = src_ptr
            for k, (base, nb) in enumerate(zip(tier_bases, tier_extents)):
                if base <= tier_ptr < base + nb:
                    out["in_tier"] = True
                    out["tier_base"] = base
                    out["tier_off"] = tier_ptr - base
                    out["tier_tail"] = base + nb - tier_ptr  # bytes to end
                    out["fits"] = (tier_ptr - base) + size <= nb
                    if k < len(tier_strides) and tier_strides[k] > 0:
                        pitch = tier_strides[k]
                        out["tier_row"] = (tier_ptr - base) // pitch
                        out["tier_cell"] = ((tier_ptr - base) % pitch)
                        # material rows are slot-interleaved; record both
                        # the cell offset and the absolute row index so
                        # seams (k*registration-chunk) can be correlated.
                        out["tier_row_pitch"] = pitch
                    break
            if "in_tier" in out:
                break
        out.setdefault("in_tier", False)
        return out

    @staticmethod
    def _stream_handle() -> int:
        try:
            return int(torch.cuda.current_stream().cuda_stream)
        except Exception:
            return 0

    def _cu_memcpy(self, dst_ptr: int, src_ptr: int, nbytes: int,
                   stream: int) -> int:
        """libcuda cuMemcpyAsync; direction inferred from pointer kinds
        (device src + registered host dst => DtoH), mirroring the op's own
        per-copy cudaMemcpyDefault fallback."""
        global _CU_LIB
        if _CU_LIB is None:
            _CU_LIB = ctypes.CDLL("libcuda.so.1")
            _CU_LIB.cuMemcpyAsync.restype = ctypes.c_int
            _CU_LIB.cuMemcpyAsync.argtypes = [
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t,
                ctypes.c_void_p,
            ]
        return int(_CU_LIB.cuMemcpyAsync(
            ctypes.c_uint64(dst_ptr), ctypes.c_uint64(src_ptr),
            ctypes.c_size_t(nbytes), ctypes.c_void_p(stream)))

    def _run_per_entry(self, src, dst, sizes, cnt: int, off: int,
                       tag: str) -> None:
        s_v, d_v, z_v = src.numpy(), dst.numpy(), sizes.numpy()
        stream = self._stream_handle()
        for i in range(off, cnt + off):
            sp, dp, sz = int(s_v[i]), int(d_v[i]), int(z_v[i])
            rc = self._cu_memcpy(dp, sp, sz, stream)
            if rc != 0:
                # Retry with the EXPLICIT direction API: if the driver's
                # default-direction inference is what's broken on this
                # pointer, an explicit DtoH/HtoD call succeeds and we log
                # the escape hatch; if it fails too, the pointer itself is
                # invalid to the driver (classification below pins whether
                # it is inside the registered tier).
                cls = self._classify(sp, dp, sz)
                rc2 = None
                try:
                    lib = _get_cu_lib()
                    if cls.get("src_mt") == 2 and cls.get("dst_mt") == 1:
                        # device src -> host dst: explicit DtoH
                        rc2 = int(lib.cuMemcpyDtoHAsync_v2(
                            ctypes.c_void_p(dp), ctypes.c_uint64(sp),
                            ctypes.c_size_t(sz), ctypes.c_void_p(stream)))
                    elif cls.get("src_mt") == 1 and cls.get("dst_mt") == 2:
                        # host src -> device dst: explicit HtoD
                        rc2 = int(lib.cuMemcpyHtoDAsync_v2(
                            ctypes.c_void_p(dp), ctypes.c_uint64(sp),
                            ctypes.c_size_t(sz), ctypes.c_void_p(stream)))
                except Exception:
                    rc2 = -1
                if rc2 == 0:
                    self._log("per_entry_explicit_ok", tag=tag, item=i,
                              src_ptr=sp, dst_ptr=dp, size=sz,
                              default_rc=rc, **cls)
                    logger.warning(
                        "swap_diag per_entry_explicit_ok item %d "
                        "(default-inference rc=%s, explicit rc=0) %s",
                        i, rc, cls)
                    continue
                # Registration-seam witness: does the failing span straddle
                # an internal cudaHostRegister chunk boundary? If yes, split
                # the copy at the seam — the root cause of crash #7 was a
                # LEGIT descriptor crossing two registered ranges, which the
                # driver rejects as a single copy (both directions, both
                # driver families). Splitting is the behavioral proof AND
                # completes the transfer. Disable only for A/B evidence
                # runs with VLLM_KV_OFFLOAD_DIAG_SPLIT_PROBE=0.
                pr = self._probe_registration(cls, sp, dp, sz, i, tag)
                _split_enabled = os.environ.get(
                    "VLLM_KV_OFFLOAD_DIAG_SPLIT_PROBE", "1") not in (
                    "0", "false", "False")
                if (pr.get("split_ranges") and pr.get("boundary")
                        and _split_enabled):
                    if self._split_retry(cls, sp, dp, sz, int(
                            pr["boundary"]), stream, i, tag):
                        self._log("per_entry_split_recovered", tag=tag,
                                  item=i, src_ptr=sp, dst_ptr=dp, size=sz,
                                  boundary=int(pr["boundary"]))
                        logger.warning(
                            "swap_diag per_entry_split_recovered item %d "
                            "(straddled registration seam at %s, default "
                            "rc=%d) ptrs=%s", i, hex(int(pr["boundary"])),
                            rc,
                            {k: v for k, v in cls.items()
                             if k not in ("src_mt", "dst_mt")})
                        continue
                self._log("per_entry_failed", tag=tag, item=i,
                          src_ptr=sp, dst_ptr=dp, size=sz, cu_rc=rc,
                          cu_rc_explicit=rc2, stream=stream, **cls)
                raise RuntimeError(
                    "swap_diag: per-entry cuMemcpyAsync failed at item %d"
                    " of %d (cu_rc=%d, explicit_rc=%s, class=%s, probe=%s)"
                    % (i, cnt, rc, rc2, cls,
                       {k: v for k, v in pr.items()
                        if k in ("split_ranges", "boundary",
                                 "bytes_before_boundary",
                                 "bytes_past_boundary")}))
        self._log("per_entry_done", tag=tag, off=off, cnt=cnt,
                  n=int(s_v.shape[0]))

    # -- the dispatcher the swap-selection sites return --------------------
    def dispatcher(self, swapper):
        def wrapped(src_ptrs, dst_ptrs, sizes, is_src_access_order_any):
            n = int(src_ptrs.shape[0])
            if self.sticky:
                self._run_per_entry(src_ptrs, dst_ptrs, sizes, n, 0,
                                    "sticky")
                return
            for off in range(0, n, _SWAP_CAP):
                end = min(off + _SWAP_CAP, n)
                try:
                    swapper(
                        src_ptrs[off:end], dst_ptrs[off:end],
                        sizes[off:end],
                        is_src_access_order_any=is_src_access_order_any)
                except Exception as exc:
                    self._recover(
                        swapper, src_ptrs, dst_ptrs, sizes,
                        off, end - off, is_src_access_order_any, exc)
        return wrapped

    def _dump_span(self, src, dst, sizes, off: int, cnt: int,
                   found: int | None) -> None:
        """Classify EVERY descriptor in the rejected chunk against the tier
        geometry. The producer's pattern (which items are out-of-tier, how
        the pointers/strides step) is the root-cause evidence — the poison
        item alone is not enough."""
        if not self._handlers:
            self._log("span_dump", note="no_geometry", off=off, cnt=cnt)
            return
        s_v, d_v, z_v = src.numpy(), dst.numpy(), sizes.numpy()
        items = []
        for i in range(off, off + cnt):
            sp, dp, sz = int(s_v[i]), int(d_v[i]), int(z_v[i])
            c = self._classify(sp, dp, sz)
            items.append({
                "i": i, "src": hex(sp), "dst": hex(dp), "size": sz,
                "in_tier": bool(c.get("in_tier")),
                "dst_off": c.get("tier_off"),
                "fits": c.get("fits"),
                "poison": i == found,
            })
        self._log("span_dump", off=off, cnt=cnt, n=int(s_v.shape[0]),
                  found=found, items=items)
        try:
            bad = [it["i"] for it in items if not it["in_tier"]]
            logger.warning(
                "swap_diag span_dump off=%d cnt=%d poison=%s "
                "out_of_tier=%s", off, cnt, found, bad)
        except Exception:
            pass

    def _recover(self, swapper, src, dst, sizes, off, cnt, any_attr,
                 exc) -> None:
        m = _RECOVER_RE.search(str(exc))
        fail_rel = int(m.group(1)) if m else None
        driver_err = int(m.group(2)) if m else None
        s_v, d_v, z_v = src.numpy(), dst.numpy(), sizes.numpy()
        self._log(
            "batch_rejected", off=off, cnt=cnt, n=int(s_v.shape[0]),
            cap=_SWAP_CAP, any_attr=any_attr, fail_idx_rel=fail_rel,
            fail_idx_abs=(off + fail_rel) if fail_rel is not None else None,
            driver_err=driver_err, error=str(exc)[:300])
        # Pristine registration witness for the known-failing item BEFORE
        # the per-entry rewrite (later copies may already have fixed the
        # pages, and attribute reads are pure queries, so this cheap probe
        # adds zero mutation risk).
        if fail_rel is not None and self._handlers:
            hit = off + fail_rel
            try:
                self._probe_registration(
                    self._classify(int(s_v[hit]), int(d_v[hit]),
                                   int(z_v[hit])),
                    int(s_v[hit]), int(d_v[hit]), int(z_v[hit]),
                    hit, tag="recover_pristine")
            except Exception as exc2:
                self._log("probe_registration", tag="recover_pristine",
                          item=hit, probe_error=str(exc2)[:200])
        found = None
        probe_note = None
        for rel in range(min(cnt, _PROBE_MAX)):
            i = off + rel
            try:
                swapper(src[i:i + 1], dst[i:i + 1], sizes[i:i + 1],
                        is_src_access_order_any=any_attr)
            except Exception as pe:
                found = i
                self._log("probe_result", found=i, src_ptr=int(s_v[i]),
                          dst_ptr=int(d_v[i]), size=int(z_v[i]),
                          error=str(pe)[:300])
                break
        if found is None and cnt > _PROBE_MAX:
            probe_note = "probed first %d only" % _PROBE_MAX
            self._log("probe_result", found=None, note=probe_note)
        # Root-cause evidence: classify the whole rejected span (which
        # items are out-of-tier, and the pointer/size pattern) BEFORE the
        # per-entry rewrite mutates anything.
        self._dump_span(src, dst, sizes, off, cnt, found)
        # Failed span may be partially enqueued (batch failure is not
        # atomic); rewrite the FULL span per-entry to be safe either way.
        self._run_per_entry(src, dst, sizes, cnt, off, "recovered")
        self.sticky = True
        self._log("recovered_ok", off=off, cnt=cnt,
                  found=found, note=probe_note,
                  sticky_enabled=True)


_SWAP_DIAG = _SwapDiag()


def _capped_swap_blocks_batch(swapper):
    """(crash-fix, 2026-09-16) cap + probe + per-entry-fallback dispatch."""
    return _SWAP_DIAG.dispatcher(swapper)


def _select_swap_blocks_fn(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    if gpu_to_cpu:
        return _capped_swap_blocks_batch(ops.swap_blocks_batch)
    # Fall back to the C++ DMA path on platforms where Triton isn't usable
    # (e.g. ROCm host mappings) or where GPU kernels cannot directly
    # dereference CPU pointers (XPU lacks CUDA's unified virtual address space,
    # so the Triton kernel's tl.load(cpu_ptr) is invalid on XPU).
    if not HAS_TRITON or current_platform.is_xpu() or current_platform.is_rocm():
        return _capped_swap_blocks_batch(ops.swap_blocks_batch)
    page_sizes = [r.page_size_bytes for g in layer_refs_per_group for r in g]
    # Triton wins only on small, 8-byte-aligned payloads.
    if (
        not page_sizes
        or max(page_sizes) >= THRESHOLD_BYTES
        or any(s % 8 for s in page_sizes)
    ):
        return _capped_swap_blocks_batch(ops.swap_blocks_batch)
    chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
    return functools.partial(swap_blocks_batch, bytes_per_chunk=chunk)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """
    Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains blocks_per_chunk sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * block_page_size

    where block_page_size = tensor.shape[1] // blocks_per_chunk (gpu page size).

    This handles tensors where row_stride != blocks_per_chunk * block_page_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        blocks_per_chunk: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.
    """
    assert skip_count < blocks_per_chunk

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if blocks_per_chunk == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for blocks_per_chunk > 1.
    assert tensor.shape[1] % blocks_per_chunk == 0
    block_page_size = tensor.shape[1] // blocks_per_chunk
    sub_offsets = np.arange(blocks_per_chunk, dtype=np.uint64) * block_page_size
    # (num_blocks, 1) + (1, blocks_per_chunk) -> (num_blocks, blocks_per_chunk)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


class CopyPlan(NamedTuple):
    """Precomputed fragment-copy template for one data ref under the canonical
    CPU layout, unrolled from the ref's mapped runs. Offsets are relative to
    the per-block base pointers on each side."""

    frag_offsets_src: np.ndarray
    frag_offsets_dst: np.ndarray
    frag_sizes: np.ndarray
    total_bytes: int

    @property
    def num_frags(self) -> int:
        return len(self.frag_sizes)


def _build_copy_plan(ref: CanonicalKVCacheRef, gpu_to_cpu: bool) -> CopyPlan:
    """Unroll one data ref's mapped runs into a per-fragment CopyPlan."""
    mapping = ref.mapping
    assert mapping is not None
    local: list[int] = []
    canonical: list[int] = []
    sizes: list[int] = []
    for run in mapping.runs:
        for i in range(run.num_fragments):
            local.append(run.local_offset + i * run.local_stride)
            canonical.append(run.canonical_offset + i * run.canonical_stride)
            sizes.append(run.fragment_size)
    src, dst = (local, canonical) if gpu_to_cpu else (canonical, local)
    return CopyPlan(
        frag_offsets_src=np.asarray(src, dtype=np.uint64),
        frag_offsets_dst=np.asarray(dst, dtype=np.uint64),
        frag_sizes=np.asarray(sizes, dtype=np.int64),
        total_bytes=sum(sizes),
    )


def _canonical_page_ids(
    block_ids: np.ndarray, blocks_per_chunk: int, count: int, skip_count: int
) -> np.ndarray:
    """Global canonical page ids matching compute_sub_block_ptrs' enumeration.
    These identify canonical pages consistently across ranks, so they key
    CanonicalPageMapping.is_writer rotation."""
    if blocks_per_chunk == 1:
        return block_ids[:count]
    flat = (
        block_ids[:, np.newaxis] * blocks_per_chunk + np.arange(blocks_per_chunk)
    ).ravel()
    return flat[skip_count : skip_count + count]


def _canonical_block_sizes(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]], num_tensors: int
) -> list[int]:
    """Canonical CPU bytes per GPU block for each tensor, taken from the refs'
    mappings. Requires every ref to carry a mapping."""
    canonical_bytes_per_block = [0] * num_tensors
    for layer_refs in layer_refs_per_group:
        for ref in layer_refs:
            assert ref.mapping is not None
            canonical_bytes_per_block[ref.tensor_idx] = max(
                canonical_bytes_per_block[ref.tensor_idx],
                ref.mapping.canonical_page_size_bytes,
            )
    assert all(size > 0 for size in canonical_bytes_per_block)
    return canonical_bytes_per_block


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register the mmap as CUDA pinned memory via cudaHostRegister, in chunks.

    The NVIDIA RM builds a 16 B-per-4-KiB-page bookkeeping table with
    kvzalloc() per registration call; kvzalloc silently rejects sizes above
    INT_MAX (2 GiB), capping one call at ~512 GiB of pages ("NVRM: failed
    to allocate page table", cudaHostRegister code=2 on drivers 580-610;
    fixed upstream ~615.71). Chunked registration raises the ceiling
    (vLLM PR #51081, sglang PR #36798; 750 GiB across 4 chunks verified in
    maru PR #64). Chunk size: VLLM_KV_OFFLOAD_REGISTER_CHUNK_GB (default 64).

    Chunk boundaries are additionally aligned to the region's material-row
    pitch (row stride = kv_bytes_per_block): an offload descriptor is always
    contained in one row, so seams that land on row starts can never be
    crossed by a descriptor. This fixes the crash-#7 class (2026-09-16):
    cuMemcpy(Batch)Async returns CUDA_ERROR_INVALID_VALUE for a span that
    straddles two cudaHostRegister'd ranges (verified on drivers 590.48 and
    615.71; the ~615.71 upstream fix only lifted the page-table cap, not the
    cross-range rejection). The decode proved the poison write started
    376,832 bytes below the 64 GiB chunk-1/chunk-2 seam and ran 294,912
    bytes past it.
    """
    if not current_platform.is_cuda_alike():
        logger.info(
            "Skipping mmap host registration on %s; cudaHostRegister is only "
            "available on CUDA/ROCm.",
            current_platform.device_name,
        )
        return

    rank = region.rank

    base_ptr = region._base.data_ptr()
    total = region.total_size_bytes

    # Chunk size in bytes: VLLM_KV_OFFLOAD_REGISTER_CHUNK_GB (GiB), default
    # 64. Clamp to [64 MiB, 512 GiB], then round DOWN to a multiple of
    # 2 MiB (2097152) so every chunk stays huge-page aligned.
    chunk_gb = 64
    raw = os.environ.get("VLLM_KV_OFFLOAD_REGISTER_CHUNK_GB")
    if raw is not None:
        try:
            chunk_gb = int(raw)
        except ValueError:
            logger.warning(
                "Invalid VLLM_KV_OFFLOAD_REGISTER_CHUNK_GB=%r; using default 64.",
                raw,
            )
            chunk_gb = 64
    chunk = max(64 * 1024**2, min(chunk_gb * 1024**3, 512 * 1024**3))
    chunk = (chunk // 2097152) * 2097152

    # Row-align the chunk: no descriptor can then straddle a registration
    # seam (they live inside one material row of _row_stride bytes). Only
    # applies when the row pitch is itself a 2 MiB multiple (hugetlbfs tier
    # geometry) and produces a sane (> 64 MiB) chunk; otherwise keep the
    # plain 2 MiB-aligned chunk.
    row_stride = int(getattr(region, "_row_stride", 0) or 0)
    if row_stride > 0 and chunk > row_stride and row_stride % 2097152 == 0:
        aligned = (chunk // row_stride) * row_stride
        if aligned >= 64 * 1024**2:
            logger.info(
                "pin_mmap_region: chunk row-aligned %d -> %d bytes "
                "(row stride %d); registration seams cannot be crossed by "
                "offload descriptors",
                chunk,
                aligned,
                row_stride,
            )
            chunk = aligned
        else:
            logger.warning(
                "pin_mmap_region: row-aligned chunk degenerated to %d bytes "
                "(< 64 MiB); keeping 2 MiB-aligned chunk %d",
                aligned,
                chunk,
            )

    # One cudaHostRegister call per chunk (python-int pointer math is fine).
    n = cdiv(total, chunk)
    registered: list[tuple[int, int]] = []
    for i in range(n):
        offset = i * chunk
        size = min(chunk, total - offset)
        result = torch.cuda.cudart().cudaHostRegister(base_ptr + offset, size, 0)
        if result.value != 0:
            # Drain the sticky CUDA error once before unwinding (maru PR #65).
            torch.cuda.cudart().cudaGetLastError()
            # Roll back previously-registered chunks in reverse order.
            for ptr, _ in reversed(registered):
                unreg_result = torch.cuda.cudart().cudaHostUnregister(ptr)
                if unreg_result.value != 0:
                    logger.warning(
                        "cudaHostUnregister rollback failed for rank=%d (code=%d)",
                        rank,
                        unreg_result,
                    )
            logger.warning(
                "cudaHostRegister failed for rank=%d (code=%d) — "
                "transfers will still work but may be slower (unpinned DMA), "
                "chunk=%d/%d, registered %.1f GB before rollback",
                rank,
                result,
                i + 1,
                n,
                i * chunk / 1e9,
            )
            return
        registered.append((base_ptr + offset, size))

    region.registered_chunks = registered
    region.is_pinned = True
    logger.info(
        "cudaHostRegister rank=%d %.2f GB pinned in %d chunk(s) of %.1f GB",
        rank,
        total / 1e9,
        n,
        chunk / 1e9,
    )


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin = PIN_MEMORY
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
    )


class SingleDirectionOffloadingHandler:
    """
    Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        canonical_layout: bool = False,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        canonical_bytes_per_block = (
            _canonical_block_sizes(layer_refs_per_group, len(gpu_tensors))
            if canonical_layout
            else None
        )

        # assert input tensors are as expected
        for t_idx, (gpu_tensor, cpu_tensor) in enumerate(zip(gpu_tensors, cpu_tensors)):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda or gpu_tensor.is_xpu
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            if canonical_bytes_per_block is not None:
                assert (
                    cpu_page_size == canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            else:
                assert cpu_page_size == gpu_page_size * blocks_per_chunk

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.layer_refs_per_group = layer_refs_per_group
        self._swap_blocks_batch = _select_swap_blocks_fn(
            layer_refs_per_group, gpu_to_cpu
        )

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * blocks_per_chunk.
        self.src_blocks_per_chunk = 1 if self.gpu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if self.gpu_to_cpu else 1
        # NOTE: must run AFTER the *_blocks_per_chunk assignments above —
        # register() reads them; calling it earlier raised AttributeError
        # that the old bare-except swallowed, so geometry never landed and
        # _classify() reported in_tier=False for every poison item.
        _SWAP_DIAG.register(self)

        # Per (group, ref) static copy plans for the canonical layout
        self._canonical_copy_plans: list[list[CopyPlan]] | None = (
            [
                [_build_copy_plan(ref, gpu_to_cpu) for ref in layer_refs]
                for layer_refs in layer_refs_per_group
            ]
            if canonical_layout
            else None
        )
        self._fill_group_ops = (
            self._fill_canonical_ops if canonical_layout else self._fill_direct_ops
        )
        # Reusable per-block base-pointer scratch for the canonical fill,
        # sized to the largest possible group (grown on demand)
        num_scratch_blocks = gpu_tensors[0].shape[0] if canonical_layout else 0
        self._scratch_bases_src = np.empty(num_scratch_blocks, dtype=np.uint64)
        self._scratch_bases_dst = np.empty(num_scratch_blocks, dtype=np.uint64)

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []
        # list of pinned descriptor buffer sets available for re-use
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def _estimate_max_copy_ops(self, group_sizes: Sequence[int]) -> int:
        """Upper bound on the number of copy descriptors for a transfer.

        Exact for the direct layout. The canonical path may fill fewer:
        writer rotation later drops the blocks this rank does not write."""
        num_copy_ops = 0
        for g_idx, (group_size, layer_refs) in enumerate(
            zip(group_sizes, self.layer_refs_per_group)
        ):
            if self._canonical_copy_plans is None:
                num_copy_ops += group_size * len(layer_refs)
            else:
                num_copy_ops += group_size * sum(
                    plan.num_frags for plan in self._canonical_copy_plans[g_idx]
                )
        return num_copy_ops

    def _fill_direct_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the direct (worker-private)
        layout: one whole-page copy per (block, ref).

        Returns (op_idx past the filled descriptors, bytes added)."""
        num_bytes = 0
        for data_ref in self.layer_refs_per_group[g_idx]:
            t_idx = data_ref.tensor_idx
            end_idx = op_idx + group_size

            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                all_src[op_idx:end_idx],
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                all_dst[op_idx:end_idx],
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
            num_bytes += group_size * data_ref.page_size_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _fill_canonical_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the canonical layout:
        scatter each block through the ref's precomputed CopyPlan, keeping
        only the blocks this rank writes.

        Returns (op_idx past the filled descriptors, bytes added)."""
        assert self._canonical_copy_plans is not None
        # Zero-copy reinterpretation for pointer arithmetic: uint64 and the
        # buffers' int64 are bit-equivalent for addresses
        all_src_u64 = all_src.view(np.uint64)
        all_dst_u64 = all_dst.view(np.uint64)
        if group_size > len(self._scratch_bases_src):
            self._scratch_bases_src = np.empty(group_size, dtype=np.uint64)
            self._scratch_bases_dst = np.empty(group_size, dtype=np.uint64)

        num_bytes = 0
        for plan, data_ref in zip(
            self._canonical_copy_plans[g_idx], self.layer_refs_per_group[g_idx]
        ):
            if plan.num_frags == 0:
                continue
            t_idx = data_ref.tensor_idx

            # 1. Base byte pointer of every block on each side
            block_bases_src = self._scratch_bases_src[:group_size]
            block_bases_dst = self._scratch_bases_dst[:group_size]
            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                block_bases_src,
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                block_bases_dst,
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            # 2. On store, keep only the blocks this rank is elected to write
            mapping = data_ref.mapping
            assert mapping is not None
            if self.gpu_to_cpu and mapping.num_writers > 1:
                block_bases_src, block_bases_dst = self._filter_writer_blocks(
                    block_bases_src,
                    block_bases_dst,
                    mapping,
                    group_dst,
                    group_size,
                    dst_skip_count,
                )
            num_active_blocks = len(block_bases_src)

            # 3. Expand (block base + fragment offset) into one descriptor
            #    per (block, fragment), writing straight into the descriptor
            #    buffers: reshaping a contiguous 1D slice is a view, so the
            #    broadcasts below allocate nothing
            end_idx = op_idx + num_active_blocks * plan.num_frags
            np.add(
                block_bases_src[:, None],
                plan.frag_offsets_src[None, :],
                out=all_src_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            np.add(
                block_bases_dst[:, None],
                plan.frag_offsets_dst[None, :],
                out=all_dst_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            all_sizes[op_idx:end_idx].reshape(num_active_blocks, plan.num_frags)[:] = (
                plan.frag_sizes
            )
            num_bytes += num_active_blocks * plan.total_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _filter_writer_blocks(
        self,
        block_bases_src: np.ndarray,
        block_bases_dst: np.ndarray,
        mapping: CanonicalPageMapping,
        group_dst: np.ndarray,
        group_size: int,
        dst_skip_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only the blocks this rank writes: replicated ranks take turns
        writing shared canonical pages, keyed by the rank-consistent CPU-side
        canonical page id."""
        cpu_page_ids = _canonical_page_ids(
            group_dst,
            self.dst_blocks_per_chunk,
            group_size,
            dst_skip_count,
        )
        writer_mask = cpu_page_ids % mapping.num_writers == mapping.writer_index
        return block_bases_src[writer_mask], block_bases_dst[writer_mask]

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        # There are 2 types of transfers:
        # 1. GPU -> CPU
        # 2. CPU -> GPU
        #
        # transfers are also to CPU blocks, EXCEPT MAYBE for the first and last block.
        # i.e. the first and last CPU blocks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU block.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU block per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU block.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.layer_refs_per_group)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.layer_refs_per_group)

        num_copy_ops = self._estimate_max_copy_ops(group_sizes)

        # reuse a pooled buffer set, growing it if this transfer needs more room
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
        all_src = src.numpy()
        all_dst = dst.numpy()
        all_sizes = sizes.numpy()

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        # count total number of bytes copied
        num_transfer_bytes = 0
        for g_idx, (group_size, block_idx) in enumerate(
            zip(group_sizes, block_indices)
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_blocks_per_chunk
            dst_logical_blocks_to_skip = block_idx % self.dst_blocks_per_chunk
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(dst_logical_blocks_count, self.dst_blocks_per_chunk)
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(src_logical_blocks_count, self.src_blocks_per_chunk)
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            op_idx, group_bytes = self._fill_group_ops(
                g_idx,
                group_src=src_blocks[src_offset:src_end_offset],
                group_dst=dst_blocks[dst_offset:dst_end_offset],
                group_size=group_size,
                src_skip_count=src_logical_blocks_to_skip,
                dst_skip_count=dst_logical_blocks_to_skip,
                all_src=all_src,
                all_dst=all_dst,
                all_sizes=all_sizes,
                op_idx=op_idx,
            )
            num_transfer_bytes += group_bytes

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        # Writer rotation may skip non-writer blocks, leaving op_idx below
        # the sized upper bound
        assert op_idx <= num_copy_ops
        src = src[:op_idx]
        dst = dst[:op_idx]
        sizes = sizes[:op_idx]

        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        # Stores must wait for the model to finish writing the KV they read.
        # Loads must wait for pending writes (including zeroing) to their
        # destination blocks; otherwise an earlier transfer can be overwritten
        # by compute-stream work that was already queued when the load began.
        stream.wait_stream(current_platform.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        # CPU->GPU reads from host pinned memory, which is never written
        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is
        # safe and lets the driver pipeline source reads. GPU->CPU reads
        # from the live GPU KV cache, which the compute stream keeps
        # writing; we must keep STREAM ordering so source reads are gated
        # by the transfer stream's wait_stream(compute) barrier.
        is_src_access_order_any = not self.gpu_to_cpu
        with current_platform.stream(stream):
            start_event.record(stream)
            if op_idx > 0:
                self._swap_blocks_batch(
                    src,
                    dst,
                    sizes,
                    is_src_access_order_any=is_src_access_order_any,
                )
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_sizes=batch_sizes,
            )
        )

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
            )

            results.append(result)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            self._buffer_pool.append(
                (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
            )
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        """Drain this direction and release its transfer-side resources."""
        sync_error: Exception | None = None
        while self._transfers:
            transfer = self._transfers[0]
            try:
                transfer.end_event.synchronize()
            except Exception as e:
                logger.exception(
                    "Failed to synchronize transfer end event; "
                    "skipping %d remaining transfers",
                    len(self._transfers) - 1,
                )
                self._transfers.clear()
                sync_error = e
                break
            self._transfers.popleft()

        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if sync_error is not None:
            raise sync_error


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
    ):
        assert not canonical_layout or mmap_region is not None
        # The caller owns mmap_region until this constructor returns. After a
        # successful construction, the worker is the sole owner and releases
        # it after both transfer directions have stopped.
        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

            if canonical_bytes_per_block is not None:
                assert mmap_region is not None
                cpu_tensor = mmap_region.create_next_canonical_view(
                    canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            elif mmap_region is not None:
                cpu_tensor = mmap_region.create_next_worker_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU -> CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async CPU -> GPU."""
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        handler_failed = False
        try:
            self._store_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down store offloading handler")
            handler_failed = True

        try:
            self._load_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down load offloading handler")
            handler_failed = True

        if self._mmap_region is not None:
            if handler_failed:
                try:
                    torch.accelerator.synchronize()
                except Exception:
                    logger.warning(
                        "Device sync before mmap cleanup failed; "
                        "proceeding with cleanup anyway",
                        exc_info=True,
                    )
            self._mmap_region.cleanup()
            self._mmap_region = None
