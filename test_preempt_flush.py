"""Preempt-flush storm test vs. crash #3 + crash #1 signature.

Phase 0: fresh ~110k-token single prompt (control; crashed boots 1-2 here).
Phase 1: 3 CONCURRENT fresh ~150k-token sessions + 1 staggered 4th — total
KV demand ~600k tokens vs 3.3e9-byte GPU pool (~411k tokens) => guaranteed
mid-decode preemption + a BIG flush store job (the 560-entry batch that died
inside cuMemcpyBatchAsync at index 34 pre-fix, engine-20260916-054426.log's
predecessors). With the <=32-descriptor cap patch riding gpu_worker.py, the
same flush must complete as 18 ordered batch calls instead.
Phase 2: repeat ONE phase-1 session prompt (tier restore/APC hit; exercises
the load path back through the pinned tier).
"""
import json
import random
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:1025"
MODEL = "glm-5.3-flash"

results = {}


def para(seed):
    rng = random.Random(seed)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
             "oscar", "papa"]
    return " ".join(rng.choice(words) for _ in range(48))


def build(n_chars, seed0=0):
    text = ""
    i = seed0
    while len(text) < n_chars:
        text += para(i) + "\n"
        i += 1
    return text[:n_chars]


def post(label, prompt, max_tokens, timeout=900):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt + "\n\nJust reply OK."}],
            "max_tokens": max_tokens, "temperature": 0}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read())
            u = j.get("usage", {})
            results[label] = ("OK", r.status, u.get("prompt_tokens"),
                              u.get("completion_tokens"), time.time() - t0)
    except Exception as e:  # noqa: BLE001
        results[label] = ("FAIL", repr(e)[:200], None, None, time.time() - t0)


# --- Phase 0: fresh ~110k control
marker = build(470_000, 500_000)
post("PH0-fresh-110k", marker, 32)
print("phase0:", results.pop("PH0-fresh-110k"))

# --- Phase 1: preempt-flush storm (3 concurrent + 1 staggered)
storms = [(f"PH1-S{i}", build(640_000, 600_000 + i * 7_000)) for i in range(3)]
ths = []
for name, p in storms:
    t = threading.Thread(target=post, args=(name, p, 64))
    t.start()
    ths.append(t)
    time.sleep(0.3)
time.sleep(2)
t4 = threading.Thread(target=post, args=("PH1-S3-late", build(640_000, 621_000), 64))
t4.start()
ths.append(t4)
for t in ths:
    t.join()

# --- Phase 2: repeat one storm prompt (tier restore / APC hit)
post("PH2-restore", storms[0][1], 32)

for label in ("PH1-S0", "PH1-S1", "PH1-S2", "PH1-S3-late", "PH2-restore"):
    r = results.get(label, ("MISSING",))
    print(label, "->", r)
print("ALL DONE")
