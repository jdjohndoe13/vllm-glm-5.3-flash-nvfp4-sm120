#!/usr/bin/env python3
# ============================================================================
# test_load_burst.py - fire N unique, think-heavy requests at an
# OpenAI-compatible chat endpoint ALL AT ONCE (chat/completions).
#
# Load shape for engine diagnosis: many requests with SMALL prompts
# (~PROMPT_CHARS UTF-8 chars each) but a LONG output budget (chain of
# thought + answer), to reproduce low co-run throughput with a waiting
# queue (Running: 1..N, Waiting: rest) on a small max_num_seqs engine.
#
# Usage:
#   py test_load_burst.py [N] [BASE_URL] [PROMPT_CHARS] [MAX_TOKENS] [MODEL]
#     N            request count                  (default 4)
#     BASE_URL     OpenAI base, e.g. http://192.168.2.208:19999/v1
#     PROMPT_CHARS target prompt size, ~chars     (default 1000)
#     MAX_TOKENS   output cap incl. reasoning     (default 10000)
#     MODEL        model id override              (default: first of GET /models)
# Flags:
#   --dry-run       generate + measure prompts only, zero network traffic
#   --timeout SEC   per-request read timeout       (default 3600)
#   --stream        SSE streaming; prints first-token latency + live tok/s
#   --seed INT      dataset generator base seed    (default 1)
# Exit: 0 when every request succeeded, 1 otherwise.
# ============================================================================
import sys
import json
import time
import random
import http.client
import urllib.parse
import threading
from concurrent.futures import ThreadPoolExecutor

PRINT_LOCK = threading.Lock()

TASK_SPEC = (
    "You are given a dataset of integers below. This is a long-reasoning "
    "benchmark: think exhaustively and show ALL intermediate arithmetic in "
    "full - raw numbers, no summaries, no 'and so on', no skipping. "
    "Perform these tasks in order:\n"
    "1) For every m in 2..12 compute S_m = the sum of every m-th value "
    "(values[m-1::m], 0-based). Show each partial accumulation.\n"
    "2) For every odd prime p in 3..31 count how many values are divisible "
    "by p. Test each value explicitly.\n"
    "3) Compute the total sum twice: once left-to-right, once right-to-left, "
    "and confirm both totals match digit by digit.\n"
    "4) Sort all values ascending. List the first 40 and the last 40 with "
    "their 0-based positions.\n"
    "5) Finish with EXACTLY one line of JSON:\n"
    '{"req": %REQID%, "sum": <sum>, "s_m": {"2": .., ..}, "div": {"3": .., ..}}\n'
    "Reason thoroughly first; the reasoning may be long. "
    "Dataset (do not restate it):\n"
)


def build_prompt(i, seed, target_chars):
    """One unique, self-contained think-heavy prompt of ~target_chars length."""
    rng = random.Random(seed * 1000003 + i * 7919 + 17)
    nonce = "%08x" % rng.getrandbits(32)
    core = (TASK_SPEC.replace("%REQID%", str(i + 1))
            + "Request #" + str(i + 1) + ", nonce " + nonce + ".\n")
    budget = max(40, target_chars - len(core.encode("utf-8")) - 2)
    nums = []
    used = 0
    k = 0
    while True:
        v = rng.randrange(-999999999, 999999999)
        piece = str(v)
        piece_len = len(piece) + (2 if nums else 0)
        if used + piece_len > budget:
            break
        nums.append(piece)
        used += piece_len
        k += 1
        if k > 5000:
            break
    return core + ", ".join(nums)


def split_base(base_url):
    base = base_url.strip().rstrip("/")
    full_endpoint = base.endswith("/chat/completions")
    return base, full_endpoint


def autodetect_model(base_url, timeout):
    models_url = (base_url.rstrip("/")) + "/models"
    p = urllib.parse.urlsplit(models_url)
    conn = http.client.HTTPConnection(p.hostname, p.port or 80, timeout=timeout)
    try:
        conn.request("GET", (p.path or "/"), headers={"Accept": "application/json"})
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise RuntimeError("GET /models HTTP %d: %r" % (resp.status, body[:200]))
        data = json.loads(body.decode("utf-8"))
        return data["data"][0]["id"]
    finally:
        conn.close()


class Counters:
    """Shared live aggregate for --stream mode."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tokens = 0
        self.first_at = None
        self.t0 = None

    def add(self, n):
        with self.lock:
            self.tokens += n

    def note_first(self):
        with self.lock:
            if self.first_at is None:
                self.first_at = time.perf_counter()

    def slogan(self):
        with self.lock:
            el = (time.perf_counter() - self.t0) if self.t0 else 0.0
            rate = (self.tokens / el) if el > 0 else 0.0
            return "live: %6.1fs  deltas=%d  rate=%.1f tok/s" % (el, self.tokens, rate)


def run_one(base, raw_body, use_stream, timeout, ctr):
    conn = http.client.HTTPConnection(base.hostname, base.port or 80, timeout=timeout)
    t0 = time.perf_counter()
    usage = None
    try:
        conn.request(
            "POST", (base.path or "") + "/chat/completions",
            body=raw_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if use_stream else "application/json",
            },
        )
        resp = conn.getresponse()
        if resp.status != 200:
            return False, "HTTP %d: %r" % (resp.status, resp.read()[:300]), None, time.perf_counter() - t0
        if use_stream:
            deltas = 0
            buf = b""
            first_seen = False
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                if not first_seen:
                    first_seen = True
                    if ctr is not None:
                        with ctr.lock:
                            if ctr.t0 is None:
                                ctr.t0 = time.perf_counter()
                        ctr.note_first()
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    tail = line[5:].strip()
                    if tail == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(tail.decode("utf-8"))
                    except Exception:
                        continue
                    ch = obj.get("choices") or []
                    if ch and (ch[0].get("delta") or {}).get("content"):
                        deltas += 1
                        if ctr is not None:
                            ctr.add(1)
                    if obj.get("usage"):
                        usage = obj["usage"]
            return True, {"deltas": deltas, "usage_optional": usage is not None}, usage, time.perf_counter() - t0
        raw = resp.read()
        obj = json.loads(raw.decode("utf-8"))
        usage = obj.get("usage") or {}
        text = ""
        ch = obj.get("choices") or []
        if ch:
            comp = ch[0].get("message") or {}
            text = (comp.get("content") or "") + ""
        return True, {"text_chars": len(text), "words": len(text.split())}, usage, time.perf_counter() - t0
    finally:
        conn.close()


def main():
    argv = sys.argv[1:]
    DRY_RUN = "--dry-run" in argv
    STREAM = "--stream" in argv
    TIMEOUT = 3600
    BASE_SEED = 1
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--dry-run", "--stream"):
            i += 1
            continue
        if a == "--timeout" and i + 1 < len(argv):
            TIMEOUT = int(argv[i + 1])
            i += 2
            continue
        if a == "--seed" and i + 1 < len(argv):
            BASE_SEED = int(argv[i + 1])
            i += 2
            continue
        positional.append(a)
        i += 1

    N = int(positional[0]) if len(positional) > 0 else 4
    BASE_URL = positional[1] if len(positional) > 1 else "http://192.168.2.208:19999/v1"
    PROMPT_CHARS = int(positional[2]) if len(positional) > 2 else 1000
    MAX_TOKENS = int(positional[3]) if len(positional) > 3 else 10000
    MODEL = positional[4] if len(positional) > 4 else None

    payloads = []
    for idx in range(N):
        prompt = build_prompt(idx, BASE_SEED, PROMPT_CHARS)
        payloads.append((idx, prompt, len(prompt.encode("utf-8"))))
    chars_list = [c for _, _, c in payloads]
    print("generated %d unique prompts, target ~%d chars (actual min %d max %d)"
          % (N, PROMPT_CHARS, min(chars_list), max(chars_list)))

    if DRY_RUN:
        for idx, prompt, ch in payloads[:3]:
            one_line = prompt.replace("\n", " ; ")
            print("  [%d] chars=%d preview: %s" % (idx, ch, one_line[:110]))
        print("dry-run: no HTTP request was made.")
        return 0

    if MODEL is None:
        MODEL = autodetect_model(BASE_URL, min(TIMEOUT, 30))
        print("autodetected model id: " + MODEL)

    base, already_ep = split_base(BASE_URL)
    p = urllib.parse.urlsplit(base)
    url_path = ((p.path or "") + "/chat/completions")
    print("POST x%d -> %s://%s%s%s" % (
        N, p.scheme, p.netloc, url_path, "  (base already full endpoint)" if already_ep else ""))
    print("  model=%s max_tokens=%d timeout=%ds%s"
          % (MODEL, MAX_TOKENS, TIMEOUT, " stream" if STREAM else ""))

    ctr = Counters() if STREAM else None
    results = {}

    def worker(idx):
        prompt = payloads[idx][1]
        raw_body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "stream": STREAM,
            "temperature": 0.7,
        }).encode("utf-8")
        try:
            results[idx] = run_one(p, raw_body, STREAM, TIMEOUT, ctr)
        except Exception as exc:
            results[idx] = (False, repr(exc), None, 0.0)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(worker, idx) for idx, _, _ in payloads]
        if STREAM and ctr is not None:
            my_ctr = ctr

            def printer_loop():
                while any(not f.done() for f in futures):
                    time.sleep(2.0)
                    with PRINT_LOCK:
                        sys.stdout.write("\r  " + my_ctr.slogan() + "   ")
                        sys.stdout.flush()
                print()

            pt = threading.Thread(target=printer_loop, daemon=True)
            pt.start()
        for f in futures:
            f.result()
    wall = time.perf_counter() - t_start

    ok_n = 0
    comp_total = 0
    reason_total = 0
    for idx, _, _ in payloads:
        r = results.get(idx)
        if r is None:
            print("  req[%2d] NO RESULT" % (idx + 1))
            continue
        ok, info, usage, item_wall = r
        if ok:
            ok_n += 1
            if usage:
                comp = usage.get("completion_tokens", 0)
                comp_total += comp
                reason = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                reason_total += reason
                print("  req[%2d] OK  %6.1fs  usage: prompt=%d completion=%d (reasoning=%d)"
                      % (idx + 1, item_wall, usage.get("prompt_tokens", 0), comp, reason))
            else:
                print("  req[%2d] OK  %6.1fs  %s" % (idx + 1, item_wall, info))
        else:
            print("  req[%2d] ERR %6.1fs  %s" % (idx + 1, item_wall, info))

    print("done: %d/%d ok, wall=%.1fs" % (ok_n, N, wall))
    if STREAM and ctr is not None:
        ft = (ctr.first_at - t_start) if ctr.first_at else -1
        print("shared first delta at %.1fs; streamed deltas %d (server usage has exact counts above)"
              % (ft, ctr.tokens))
    else:
        print("server-reported completion tokens: %d (reasoning inside: %d)" % (comp_total, reason_total))
        if wall > 0 and comp_total > 0:
            print("aggregate completion throughput over whole batch: %.1f tok/s" % (comp_total / wall))
    return 0 if ok_n == N else 1


if __name__ == "__main__":
    sys.exit(main())
