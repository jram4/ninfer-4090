#!/usr/bin/env python3
"""Serve-level K / workload sweep for Cinference on a single GPU.

Each mode runs in a fresh ninfer-serve process with prefix reuse disabled. Full
outputs are stored and compared against mode k0 for the same workload, size and
repetition.

  cinf_k_bench.py --out DIR --modes k0,k3,k5,k10 --workloads recall,code,reason \
      --recall-tokens 8192,32768 --reps 2 --max-tokens 256
"""
import argparse, hashlib, json, os, random, subprocess, sys, time, urllib.request

ROOT = os.environ.get("CINF_ROOT", "/home/aramirezfamily/josh/cinference-4090")
DEFAULT_BIN = os.environ.get("CINF_BIN", f"{ROOT}/build-sm89/apps/ninfer-serve")
DEFAULT_MODEL = f"{ROOT}/models/qwen3_8_27b.v3.ninfer"
MODEL_ID = "qwen3.8-27b"
TOKENS_PER_ARCHIVE_ENTRY = 28

WORDS = ("amber basalt cedar delta ember fjord granite harbor indigo juniper kelp lagoon meadow nickel onyx "
         "prairie quartz reef summit tundra umber valley willow xenon yarrow zephyr").split()

CODE_PROMPT = ("Write a complete, well-commented Python implementation of an LRU cache class with get, put, and "
               "delete, plus unit tests using unittest.")
REASON_PROMPT = "Write a Python function that merges overlapping integer intervals. Explain its complexity."


def build_archive(target_tokens, seed=7):
    rng = random.Random(seed)
    entries = max(16, target_tokens // TOKENS_PER_ARCHIVE_ENTRY)
    lines = [f"Archive entry {i:04d}: the {rng.choice(WORDS)} ledger lists {rng.randint(100, 999)} crates of "
             f"{rng.choice(WORDS)} stored near the {rng.choice(WORDS)} gate." for i in range(entries)]
    markers = {}
    for frac, name in zip([0.01, 0.10, 0.25, 0.50, 0.75, 0.95], "ABCDEF"):
        value = f"{name}-{rng.randint(10000, 99999)}"
        markers[name] = value
        lines.insert(int(len(lines) * frac), f"SECRET MARKER {name} = {value}.")
    prompt = ("\n".join(lines) + "\n\nList all six SECRET MARKER values in order, then copy archive entries 0000 "
              "through 0010 exactly as written.")
    return prompt, markers


def workload_cases(workloads, recall_sizes):
    cases = []
    for name in workloads:
        if name == "recall":
            for size in recall_sizes:
                prompt, markers = build_archive(size)
                cases.append({"workload": f"recall_{size}", "prompt": prompt, "thinking": False, "markers": markers})
        elif name == "code":
            cases.append({"workload": "code", "prompt": CODE_PROMPT, "thinking": False})
        elif name == "reason":
            cases.append({"workload": "reason", "prompt": REASON_PROMPT, "thinking": True})
        else:
            raise SystemExit(f"unknown workload {name}")
    return cases


def mode_args(mode):
    """k0 = no speculation, kN = fixed MTP window N, adaptiveN = adaptive window with maximum N."""
    if mode == "k0":
        return []
    if mode.startswith("adaptive"):
        return ["--spec", "mtp", "--draft-tokens", mode[len("adaptive"):] or "10", "--lm-head-draft",
                "--adaptive-draft-window"]
    if mode.startswith("k"):
        return ["--spec", "mtp", "--draft-tokens", mode[1:], "--lm-head-draft"]
    raise SystemExit(f"unknown mode {mode}")


def post(port, prompt, thinking, max_tokens):
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def wait_health(proc, port):
    for _ in range(900):
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def gpu_memory_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split()[0])
    except Exception:
        return None


def request_rows(log):
    if not os.path.exists(log):
        return []
    rows = [json.loads(line) for line in open(log) if line.strip()]
    return [r for r in rows if r.get("event") == "request_done"]


def metrics(record):
    res, t, s = record.get("result", {}), record.get("timings_seconds", {}), record.get("speculative") or {}
    ct, d = res.get("completion_tokens", 0), t.get("decode") or 0
    rounds = s.get("rounds") or ct
    return {"prompt_tokens": res.get("prompt_tokens"), "completion_tokens": ct,
            "decode_tps": ct / d if d else None,
            "acceptance": s.get("accepted_tokens", 0) / s["drafted_tokens"] if s.get("drafted_tokens") else None,
            "tokens_per_round": ct / rounds if rounds else None,
            "ms_per_round": 1000 * d / rounds if rounds else None,
            "prefill_seconds": t.get("prefill")}


def run_mode(args, mode, cases):
    log = f"{args.out}/{mode}.requests.jsonl"
    if os.path.exists(log):
        os.remove(log)
    serve = [args.bin, args.model, "--host", "127.0.0.1", "--port", str(args.port), "--max-context",
             str(args.max_context), "--kv-capacity", str(args.max_context), "--max-concurrency", "1",
             "--prefill-chunk", "1024", "--kv-dtype", "k8v4", "--no-prefix-reuse", "--request-log-jsonl", log]
    serve += mode_args(mode) + args.extra_serve_arg
    samples = []
    with open(f"{args.out}/{mode}.stderr.log", "w") as err:
        proc = subprocess.Popen(serve, stdout=err, stderr=err)
        try:
            if not wait_health(proc, args.port):
                print(f"{mode}: server failed to start (see {mode}.stderr.log)", flush=True)
                return [{"mode": mode, "error": "startup"}]
            ready_mib = gpu_memory_used_mib()
            post(args.port, "Say hello.", False, 16)
            for case in cases:
                for rep in range(args.reps):
                    resp = post(args.port, case["prompt"], case["thinking"], args.max_tokens)
                    msg = resp["choices"][0]["message"]
                    text = (msg.get("reasoning_content") or "") + "\n<content>\n" + (msg.get("content") or "")
                    sample = {"mode": mode, "workload": case["workload"], "rep": rep, "text": text,
                              "sha256": hashlib.sha256(text.encode()).hexdigest(), "gpu_mib_ready": ready_mib,
                              "gpu_mib_after": gpu_memory_used_mib()}
                    if "markers" in case:
                        sample["recall_passed"] = sum(v in text for v in case["markers"].values())
                    samples.append(sample)
        finally:
            proc.terminate()
            try:
                proc.wait(120)
            except subprocess.TimeoutExpired:
                proc.kill()
    records = request_rows(log)[1:]
    for sample, record in zip(samples, records):
        sample.update(metrics(record))
    time.sleep(3)
    return samples


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--modes", default="k0,k3,k5,k10")
    p.add_argument("--workloads", default="recall,code,reason")
    p.add_argument("--recall-tokens", default="25000")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-context", type=int, default=0)
    p.add_argument("--bin", default=DEFAULT_BIN)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--port", type=int, default=8094)
    p.add_argument("--extra-serve-arg", action="append", default=[])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recall_sizes = [int(x) for x in args.recall_tokens.split(",") if x]
    cases = workload_cases(args.workloads.split(","), recall_sizes)
    if not args.max_context:
        largest = max(recall_sizes) if "recall" in args.workloads else 0
        args.max_context = max(8192, int(largest * 1.1) + args.max_tokens + 1024)
    all_samples = []
    for mode in args.modes.split(","):
        all_samples += run_mode(args, mode, cases)
    baseline = {(s["workload"], s["rep"]): s for s in all_samples if s.get("mode") == "k0" and "text" in s}
    for s in all_samples:
        ref = baseline.get((s.get("workload"), s.get("rep")))
        if ref is not None and s.get("mode") != "k0" and "text" in s:
            s["matches_k0"] = s["sha256"] == ref["sha256"]
            s["first_divergent_char"] = first_divergence(ref["text"], s["text"])
    json.dump({"args": vars(args), "samples": all_samples}, open(f"{args.out}/summary.json", "w"), indent=1)
    for s in all_samples:
        if "error" in s:
            print(f"{s['mode']:<11} ERROR {s['error']}")
            continue
        print("%-11s %-13s rep=%d pt=%-7s ct=%-4d tps=%6.1f acc=%s tok/round=%5.2f ms/round=%5.1f%s%s" % (
            s["mode"], s["workload"], s["rep"], s.get("prompt_tokens"), s.get("completion_tokens") or 0,
            s.get("decode_tps") or 0, "%.3f" % s["acceptance"] if s.get("acceptance") is not None else "  -  ",
            s.get("tokens_per_round") or 0, s.get("ms_per_round") or 0,
            "" if "matches_k0" not in s else (" =K0" if s["matches_k0"] else f" DIFF@{s['first_divergent_char']}"),
            f" recall={s['recall_passed']}/6" if "recall_passed" in s else ""), flush=True)


if __name__ == "__main__":
    main()
