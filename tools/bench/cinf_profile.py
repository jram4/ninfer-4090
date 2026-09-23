#!/usr/bin/env python3
"""Profile one recall request under nsys and attribute decode GPU time.

usage: cinf_profile.py OUT_DIR K MODE   (MODE = graph | eager)
"""
import json, os, signal, sqlite3, subprocess, sys, time, urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cinf_k_bench import build_archive

ROOT = os.environ.get("CINF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BIN = os.environ.get("CINF_BIN", f"{ROOT}/build-sm89-perf/apps/ninfer-serve")
MODEL = f"{ROOT}/models/qwen3_8_27b.v3.ninfer"
OUT, K, MODE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PORT = 8095
MAX_TOKENS = int(os.environ.get("CINF_PROFILE_TOKENS", "128"))


def recall_prompt():
    return build_archive(int(os.environ.get("CINF_PROFILE_RECALL_TOKENS", "25000")))[0]


def post(prompt, max_tokens):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def run_profile():
    os.makedirs(OUT, exist_ok=True)
    rep = f"{OUT}/k{K}_{MODE}"
    serve = [BIN, MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--max-context", "32768", "--kv-capacity", "32768",
             "--max-concurrency", "1", "--prefill-chunk", "1024", "--kv-dtype", "k8v4", "--no-prefix-reuse",
             "--request-log-jsonl", f"{rep}.requests.jsonl"]
    if K > 0:
        serve += ["--spec", "mtp", "--draft-tokens", str(K), "--lm-head-draft"]
    if MODE == "eager":
        serve += ["--no-cuda-graph"]
    nsys = ["nsys", "profile", "-o", rep, "-f", "true", "-t", "cuda,nvtx", "--sample=none", "--cpuctxsw=none"]
    if MODE == "graph":
        nsys += ["--cuda-graph-trace=node"]
    with open(f"{rep}.log", "w") as log:
        proc = subprocess.Popen(nsys + serve, stdout=log, stderr=log)
        try:
            for _ in range(900):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                    break
                except Exception:
                    if proc.poll() is not None:
                        raise SystemExit("serve exited during startup")
                    time.sleep(1)
            post("Say hello.", 8)
            post(recall_prompt(), MAX_TOKENS)
        finally:
            proc.send_signal(signal.SIGINT)
            proc.wait(600)
    subprocess.run(["nsys", "export", "--type", "sqlite", "-f", "true", "-o", f"{rep}.sqlite", f"{rep}.nsys-rep"],
                   check=True, stdout=subprocess.DEVNULL)
    return rep


def analyze(rep):
    db = sqlite3.connect(f"{rep}.sqlite")
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    nvtx = []
    for start, end, text, tid in db.execute("SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        nvtx.append((start, end, text if text is not None else strings.get(tid, "")))
    round_names = {"decode.mtp_round", "decode.ordinary_round"}
    rounds = sorted((s, e) for s, e, n in nvtx if n in round_names)
    if not rounds:
        raise SystemExit("no decode round ranges found")
    # The profiled recall request is the last contiguous run of rounds; a prefill gap separates it from warmup.
    cluster = [rounds[-1]]
    for r in reversed(rounds[:-1]):
        if cluster[-1][0] - r[1] > 200_000_000:
            break
        cluster.append(r)
    rounds = sorted(cluster)
    lo, hi = rounds[0][0], rounds[-1][1]
    runtime = {cid: start for cid, start in db.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    ranges = [(s, e, n) for s, e, n in nvtx if s >= lo and e <= hi and n in ("decode.mtp.draft", "decode.mtp.target")]
    by_kernel = defaultdict(lambda: [0, 0])
    by_phase = defaultdict(int)
    total = 0
    kernel_rows = db.execute("SELECT start, end, demangledName, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                             "WHERE start >= ? AND end <= ?", (lo, hi + 5_000_000)).fetchall()
    for start, end, dname, sname, cid in kernel_rows:
        dur = end - start
        name = strings.get(sname, "?")
        by_kernel[name][0] += dur
        by_kernel[name][1] += 1
        total += dur
        phase = "other"
        launch = runtime.get(cid)
        if MODE == "eager" and launch is not None:
            for s, e, n in ranges:
                if s <= launch <= e:
                    phase = "draft" if n == "decode.mtp.draft" else "target"
                    break
        by_phase[phase] += dur
    n_rounds = len(rounds)
    wall = hi - lo
    result = {"k": K, "mode": MODE, "rounds": n_rounds, "wall_ms_per_round": wall / n_rounds / 1e6,
              "gpu_kernel_ms_per_round": total / n_rounds / 1e6,
              "phase_ms_per_round": {p: v / n_rounds / 1e6 for p, v in by_phase.items()},
              "top_kernels": [{"name": n, "ms_per_round": v[0] / n_rounds / 1e6, "calls_per_round": v[1] / n_rounds}
                              for n, v in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:40]]}
    json.dump(result, open(f"{rep}.analysis.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "top_kernels"}))
    for row in result["top_kernels"][:25]:
        print("%8.3f ms/round  %6.1f calls  %s" % (row["ms_per_round"], row["calls_per_round"], row["name"][:110]))


if __name__ == "__main__":
    rep = f"{OUT}/k{K}_{MODE}" if os.environ.get("CINF_ANALYZE_ONLY") else run_profile()
    analyze(rep)
