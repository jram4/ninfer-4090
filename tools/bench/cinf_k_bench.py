#!/usr/bin/env python3
import json, os, random, subprocess, sys, time, urllib.request

ROOT = os.environ.get("CINF_ROOT", "/home/aramirezfamily/josh/cinference-4090")
BIN = os.environ.get("CINF_BIN", f"{ROOT}/build-sm89/apps/ninfer-serve")
MODEL = f"{ROOT}/models/qwen3_8_27b.v3.ninfer"
OUT = sys.argv[1]
KS = [int(k) for k in sys.argv[2].split(",")]
PORT = 8094

random.seed(7)
words = "amber basalt cedar delta ember fjord granite harbor indigo juniper kelp lagoon meadow nickel onyx prairie quartz reef summit tundra umber valley willow xenon yarrow zephyr".split()
lines = []
markers = {}
for i in range(900):
    lines.append(f"Archive entry {i:04d}: the {random.choice(words)} ledger lists {random.randint(100,999)} crates of {random.choice(words)} stored near the {random.choice(words)} gate.")
for frac, name in zip([0.01, 0.10, 0.25, 0.50, 0.75, 0.95], "ABCDEF"):
    val = f"{name}-{random.randint(10000,99999)}"
    markers[name] = val
    lines.insert(int(len(lines) * frac), f"SECRET MARKER {name} = {val}.")
archive = "\n".join(lines)

WORKLOADS = {
    "recall_nothink": dict(
        prompt=archive + "\n\nList all six SECRET MARKER values in order, then copy archive entries 0000 through 0010 exactly as written.",
        thinking=False),
    "code_nothink": dict(
        prompt="Write a complete, well-commented Python implementation of an LRU cache class with get, put, and delete, plus unit tests using unittest.",
        thinking=False),
    "reason_think": dict(
        prompt="Write a Python function that merges overlapping integer intervals. Explain its complexity.",
        thinking=True),
}

def post(prompt, thinking, max_tokens):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())

def wait_health(proc):
    for _ in range(600):
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False

os.makedirs(OUT, exist_ok=True)
summary = []
for k in KS:
    log = f"{OUT}/k{k}.requests.jsonl"
    args = [BIN, MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--max-context", "32768", "--kv-capacity", "32768",
            "--max-concurrency", "1", "--prefill-chunk", "1024", "--kv-dtype", "k8v4", "--no-prefix-reuse",
            "--request-log-jsonl", log]
    if k > 0:
        args += ["--spec", "mtp", "--draft-tokens", str(k), "--lm-head-draft"]
    with open(f"{OUT}/k{k}.stderr.log", "w") as err:
        proc = subprocess.Popen(args, stdout=err, stderr=err)
        try:
            if not wait_health(proc):
                print(f"K{k}: server failed to start", flush=True)
                summary.append({"k": k, "error": "startup"})
                continue
            post("Say hello.", False, 16)
            for name, w in WORKLOADS.items():
                for rep in range(2):
                    resp = post(w["prompt"], w["thinking"], 256)
                    msg = resp["choices"][0]["message"]
                    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
                    summary.append({"k": k, "workload": name, "rep": rep, "text_head": text[:160]})
        finally:
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
    time.sleep(3)

rows = []
for k in KS:
    log = f"{OUT}/k{k}.requests.jsonl"
    if not os.path.exists(log):
        continue
    done = [json.loads(l) for l in open(log) if l.strip()]
    done = [r for r in done if r.get("event") == "request_done"][1:]
    names = [n for n in WORKLOADS for _ in range(2)]
    for name, r in zip(names, done):
        res, t, s = r.get("result", {}), r.get("timings_seconds", {}), r.get("speculative") or {}
        ct, d = res.get("completion_tokens", 0), t.get("decode") or 0
        rounds = s.get("rounds") or ct
        rows.append({"k": k, "workload": name, "prompt_tokens": res.get("prompt_tokens"), "completion_tokens": ct,
                     "decode_tps": ct / d if d else None,
                     "acceptance": s.get("accepted_tokens", 0) / s["drafted_tokens"] if s.get("drafted_tokens") else None,
                     "tokens_per_round": ct / rounds if rounds else None,
                     "ms_per_round": 1000 * d / rounds if rounds else None})
json.dump({"rows": rows, "samples": summary, "markers": markers}, open(f"{OUT}/summary.json", "w"), indent=1)
for r in rows:
    print("K%-2d %-15s pt=%-6s ct=%-4d tps=%6.1f acc=%s tok/round=%5.2f ms/round=%5.1f" % (
        r["k"], r["workload"], r["prompt_tokens"], r["completion_tokens"], r["decode_tps"] or 0,
        "%.3f" % r["acceptance"] if r["acceptance"] is not None else "  -  ", r["tokens_per_round"] or 0,
        r["ms_per_round"] or 0), flush=True)
