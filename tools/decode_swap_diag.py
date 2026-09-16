#!/usr/bin/env python3
"""Stride-aware decode of the swap_diag JSONL.

Key insight: the 22 dst tensors of the store handler are strided views of
the shared pinned region (consecutive data_ptrs 671,744 apart = one page),
NOT contiguous tensors. Bounds checks must use the STRIDED span:
    stride(0) != numel/rows  =>  span = stride*(rows-1) + row_bytes
"""
import glob
import json
import os
import sys

DIAG = sys.argv[1] if len(sys.argv) > 1 else (
    "/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4/swap_diag")

PAGE = 671744           # mamba/GDN page size (store handler row_bytes)
MIB2 = 2097152          # registration chunk alignment
TIB = 858993459200      # nominal shared region size (cpu_bytes_to_use)

evs = []
files = sorted(glob.glob(os.path.join(DIAG, "log_swap_T*.jsonl")))
for p in files:
    for line in open(p):
        try:
            evs.append((os.path.basename(p), json.loads(line)))
        except Exception:
            pass

by_rank = {}
for rank, e in evs:
    by_rank.setdefault(rank, []).append(e)

for rank in sorted(by_rank):
    evs_r = by_rank[rank]
    hi = [e for e in evs_r if e.get("ev") == "handler_init"]
    sd = [e for e in evs_r if e.get("ev") == "span_dump"]
    pf = [e for e in evs_r if e.get("ev") == "per_entry_failed"]
    if not (sd or pf):
        continue
    store = None
    load = None
    for h in hi:
        if h.get("gpu_to_cpu"):
            store = h
        else:
            load = h
    print("=" * 76)
    print("RANK %s spans=%d fails=%d" % (rank, len(sd), len(pf)))
    if not store or not store.get("dst") or not store.get("dst_nbytes"):
        continue
    b0 = min(store["dst"])
    print("region-view: b0=%s dst sizes first=%s" % (hex(b0),
          store["dst_nbytes"][:3]))
    rows = store["rows"]
    rb = store["row_bytes"]
    # The dst tensors are strided views; interpret region span from the
    # FLAT load handler view if present (rows*row_bytes = whole tier).
    if load and load.get("rows") and load.get("row_bytes"):
        span = load["rows"] * load["row_bytes"]
        lb0 = min(load["src"]) if load.get("src") else None
        print("flat load view: base=%s rows*row_bytes=%s" % (
            hex(lb0) if lb0 else "?", span))
    else:
        span = TIB
        lb0 = None

    for s0 in sd[:1]:
        items = s0.get("items") or []
        print("span off=%s cnt=%s found=%s n=%s sizes=%s" % (
            s0.get("off"), s0.get("cnt"), s0.get("found"), s0.get("n"),
            sorted(set(it["size"] for it in items))))
        # dst analysis per item
        print("  %-4s %-16s %-8s %-10s %-7s %s" % (
            "i", "dst", "off-b0", "mod page", "mod 2MiB", "2MiB-cross"))
        for it in items:
            d = int(it["dst"], 16)
            sz = it["size"]
            offb0 = d - b0
            q, r = divmod(offb0, PAGE)
            if lb0 is not None:
                reg_off = d - lb0
            else:
                reg_off = offb0
            m2 = d % MIB2
            cross2 = (d & ~(MIB2 - 1)) != ((d + sz - 1) & ~(MIB2 - 1))
            flag = "POISON" if it["i"] == s0.get("found") else ""
            print("  %-4d %-16s %-10d %-10d %-10d %s %s" % (
                it["i"], it["dst"], offb0, r, m2, cross2, flag))
        # group the sizes: dsts of same-size items should step uniformly
        for it in items[:2]:
            pass
    for e in pf:
        d = e["dst_ptr"]
        sz = e["size"]
        offb0 = d - b0
        q, r = divmod(offb0, PAGE)
        print("FAIL item=%s: dst-b0=%d page_idx(abs)=%d intra=%d "
              "in_region=%s reg_off=%d" % (
                  e.get("item"), offb0, q, r,
                  (0 <= (d - (lb0 or b0)) < span) if lb0 else "?",
                  (d - lb0) if lb0 else offb0))
