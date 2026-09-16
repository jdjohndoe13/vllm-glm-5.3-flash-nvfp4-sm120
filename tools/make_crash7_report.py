#!/usr/bin/env python3
"""Build a crash-#7 analysis report for offline review (oracle handoff)."""
import glob
import json
import subprocess

OUT = "/tmp/crash7-report.txt"
KIT = "/mnt/data/shared/models/vllm-glm-5.3-flash-nvfp4"
DIAG = KIT + "/swap_diag"

parts = []

def add(head, lines):
    parts.append("=== %s ===" % head)
    parts.extend(lines)
    parts.append("")

try:
    add("storm3.log tail", open(KIT + "/logs/storm3.log").read().splitlines()[-15:])
except Exception as exc:
    add("storm3.log tail", ["unavailable: %s" % exc])

log = KIT + "/logs/latest.log"
try:
    hits = [ln for ln in open(log) if ("swap_diag" in ln or "ERROR" in ln
            or "Traceback" in ln)]
    add("engine log signals (%d lines)" % len(hits), hits[-25:])
except Exception as exc:
    add("engine log signals", ["unavailable: %s" % exc])

for name in ("HugePages_Total:", "HugePages_Free:", "Hugepagesize",
             "FileHugePages", "AnonHugePages", "MemFree:", "MemAvailable:"):
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith(name):
                parts.append(ln.rstrip())
    except Exception:
        pass
parts.append("")

try:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10)
    add("gpu memory", q.stdout.splitlines())
except Exception as exc:
    add("gpu memory", ["unavailable: %s" % exc])

for p in sorted(glob.glob(DIAG + "/log_swap_T13739.jsonl")):
    evs = [json.loads(ln) for ln in open(p)]
    for e in evs:
        if e.get("ev") in ("batch_rejected", "per_entry_failed",
                           "handler_init"):
            parts.append("=== %s %s ===" % (p, e.get("ev")))
            parts.append(json.dumps(e))
    for e in evs:
        if e.get("ev") == "span_dump":
            parts.append("=== %s span_dump ===" % p)
            parts.append(json.dumps(e))

with open(OUT, "w") as fh:
    fh.write("\n".join(parts))
print("WROTE", OUT, "lines:", len(parts))
