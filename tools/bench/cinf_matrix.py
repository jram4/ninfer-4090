#!/usr/bin/env python3
"""Production benchmark matrix for Cinference on one GPU.

One fresh ninfer-serve process per speculative mode, prefix reuse disabled, greedy decoding. Each
sample records TTFT/prefill, decode rate, round latency, draft acceptance, VRAM, sampled GPU
utilization, KV occupancy, a functional check and divergence from K0 on the same case.

  cinf_matrix.py --out DIR --modes k0,k3,k5,k7,k10 --max-context 196608 \
      --recall-tokens 8192,32768,65536,131072,190000 --reps 2
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cinf_k_bench import (DEFAULT_MODEL, build_archive, first_divergence, metrics, mode_args, post,
                          request_rows, wait_health)

ROOT = os.environ.get("CINF_ROOT", "/home/aramirezfamily/josh/cinference-4090")

CODE_PROMPT = ("Write a complete Python module implementing an LRU cache class with get, put and delete, "
               "followed by at most five concise unittest test methods that exercise eviction order. Put everything "
               "in one ```python block "
               "and end the block with: if __name__ == '__main__': unittest.main()")
STRUCT_PROMPT = ("Return ONLY a JSON object (no prose, no code fence) describing a fictional city with keys: "
                 "name (string), founded (integer year), population (integer), districts (array of exactly 3 "
                 "objects each with name and area_km2), climate {summer_c: number, winter_c: number}, "
                 "coastal (boolean).")
REASON_PROMPT = ("A train leaves at 3:40 PM and travels 210 km at a constant 84 km/h, then waits 25 minutes, then "
                 "travels 96 km at 64 km/h. At what time does it arrive? End with 'Final answer: HH:MM' in "
                 "24-hour time.")
PROSE_PROMPT = "Write a 500-word short story about a lighthouse keeper who receives a letter from the future."
SHORT_PROMPT = "What is the capital of Australia? Answer with one word."
LONG_PROMPT = ("Write a detailed, well-structured technical guide (about 2500 words) to building a quiet, "
               "energy-efficient home NAS: hardware selection, ZFS layout, backups, networking, monitoring and "
               "maintenance.")


def extract_python(text):
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


def check_code(text):
    code = extract_python(text)
    if code is None:
        return "no-code-block"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=60)
        return "pass" if r.returncode == 0 else "tests-failed"
    except subprocess.TimeoutExpired:
        return "timeout"
    finally:
        os.unlink(f.name)


def check_json(text):
    body = text.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
    try:
        obj = json.loads(body)
    except Exception:
        return "invalid-json"
    ok = isinstance(obj, dict) and isinstance(obj.get("districts"), list) and len(obj["districts"]) == 3
    return "pass" if ok else "schema-mismatch"


def check_reason(text):
    m = re.findall(r"Final answer:\s*\**\s*(\d{1,2}:\d{2})", text)
    if not m:
        return "no-final-answer"
    return "pass" if m[-1] == "20:05" else f"wrong:{m[-1]}"


def cases(recall_sizes):
    out = [
        {"workload": "short", "prompt": SHORT_PROMPT, "thinking": False, "max_tokens": 32,
         "check": lambda c, r: "pass" if "canberra" in c.lower() else "wrong"},
        {"workload": "structured", "prompt": STRUCT_PROMPT, "thinking": False, "max_tokens": 512,
         "check": lambda c, r: check_json(c)},
        {"workload": "code", "prompt": CODE_PROMPT, "thinking": False, "max_tokens": 3072,
         "check": lambda c, r: check_code(c)},
        {"workload": "prose", "prompt": PROSE_PROMPT, "thinking": False, "max_tokens": 768,
         "check": lambda c, r: "pass" if len(c.split()) > 250 else "short"},
        {"workload": "reason", "prompt": REASON_PROMPT, "thinking": True, "max_tokens": 4096,
         "check": lambda c, r: check_reason(c)},
        {"workload": "long_gen", "prompt": LONG_PROMPT, "thinking": False, "max_tokens": 3072,
         "check": lambda c, r: "pass" if len(c.split()) > 1200 else "short"},
    ]
    for size in recall_sizes:
        prompt, markers = build_archive(size)
        out.append({"workload": f"recall_{size}", "prompt": prompt, "thinking": False, "max_tokens": 160,
                    "markers": markers,
                    "check": lambda c, r, m=markers: "pass" if all(v in c for v in m.values())
                    else f"recall-{sum(v in c for v in m.values())}/6"})
    return out


class GpuSampler:
    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=5).stdout.split(",")
                self.samples.append((float(out[0]), float(out[1]), float(out[2])))
            except Exception:
                pass
            self._stop.wait(0.5)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self):
        if not self.samples:
            return {}
        util = [s[0] for s in self.samples]
        return {"gpu_util_mean": sum(util) / len(util), "gpu_mib_peak": max(s[1] for s in self.samples),
                "gpu_watts_mean": sum(s[2] for s in self.samples) / len(self.samples)}


def run_mode(args, mode, all_cases):
    log = f"{args.out}/{mode}.requests.jsonl"
    if os.path.exists(log):
        os.remove(log)
    serve = [args.bin, args.model, "--host", "127.0.0.1", "--port", str(args.port), "--max-context",
             str(args.max_context), "--kv-capacity", str(args.max_context), "--max-concurrency", "1",
             "--prefill-chunk", str(args.prefill_chunk), "--kv-dtype", args.kv_dtype, "--no-prefix-reuse",
             "--greedy", "--request-log-jsonl", log] + mode_args(mode) + args.extra_serve_arg
    samples, started = [], time.time()
    with open(f"{args.out}/{mode}.stderr.log", "w") as err:
        proc = subprocess.Popen(serve, stdout=err, stderr=err)
        try:
            if not wait_health(proc, args.port):
                return [{"mode": mode, "error": "startup"}], None
            startup_s = time.time() - started
            with GpuSampler() as idle:
                time.sleep(2)
            post(args.port, "Say hello.", False, 16)
            for case in all_cases:
                reps = 1 if case["workload"].startswith("recall") else args.reps
                for rep in range(reps):
                    sample = {"mode": mode, "workload": case["workload"], "rep": rep}
                    try:
                        with GpuSampler() as gpu:
                            t0 = time.time()
                            resp = post(args.port, case["prompt"], case["thinking"], case["max_tokens"])
                            sample["wall_s"] = time.time() - t0
                        msg = resp["choices"][0]["message"]
                        content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
                        text = reasoning + "\n<content>\n" + content
                        sample.update({"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                       "finish_reason": resp["choices"][0].get("finish_reason"),
                                       "check": case["check"](content, reasoning), **gpu.summary()})
                    except Exception as e:
                        sample["error"] = repr(e)[:300]
                    samples.append(sample)
                    print(f"{mode} {case['workload']} rep={rep} {sample.get('check', sample.get('error'))}",
                          flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(120)
            except subprocess.TimeoutExpired:
                proc.kill()
    records = request_rows(log)[1:]
    ok = [s for s in samples if "error" not in s]
    for sample, record in zip(ok, records):
        sample.update(metrics(record))
        t = record.get("timings_seconds", {})
        sample["ttft_s"] = t.get("ttft") or t.get("prefill")
        pt, ct = sample.get("prompt_tokens") or 0, sample.get("completion_tokens") or 0
        sample["prefill_tps"] = pt / t["prefill"] if t.get("prefill") else None
        sample["kv_fill"] = (pt + ct) / args.max_context
    capacity = next((l.strip() for l in open(f"{args.out}/{mode}.stderr.log") if "capacity |" in l), None)
    time.sleep(3)
    return samples, {"mode": mode, "startup_s": startup_s, "idle": idle.summary(), "capacity_log": capacity}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--modes", default="k0,k3,k5,k7,k10")
    p.add_argument("--recall-tokens", default="8192,32768,65536,131072")
    p.add_argument("--workloads", default="")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-context", type=int, default=196608)
    p.add_argument("--prefill-chunk", type=int, default=1024)
    p.add_argument("--kv-dtype", default="k8v4")
    p.add_argument("--bin", default=f"{ROOT}/build-sm89-perf/apps/ninfer-serve")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--extra-serve-arg", action="append", default=[])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    all_cases = cases([int(x) for x in args.recall_tokens.split(",") if x])
    if args.workloads:
        keep = set(args.workloads.split(","))
        all_cases = [c for c in all_cases if c["workload"] in keep or c["workload"].split("_")[0] in keep]
    samples, servers = [], []
    for mode in args.modes.split(","):
        s, info = run_mode(args, mode, all_cases)
        samples += s
        if info:
            servers.append(info)
        json.dump({"args": vars(args), "servers": servers, "samples": samples},
                  open(f"{args.out}/summary.json", "w"), indent=1)
    base = {(s["workload"], s["rep"]): s for s in samples if s.get("mode") == "k0" and "text" in s}
    for s in samples:
        ref = base.get((s.get("workload"), s.get("rep")))
        if ref is not None and s.get("mode") != "k0" and "text" in s:
            s["matches_k0"] = s["sha256"] == ref["sha256"]
            s["first_divergent_char"] = first_divergence(ref["text"], s["text"])
    json.dump({"args": vars(args), "servers": servers, "samples": samples},
              open(f"{args.out}/summary.json", "w"), indent=1)
    for s in samples:
        if "error" in s:
            print(f"{s['mode']:<5} {s.get('workload', '-'):<14} ERROR {s['error']}")
            continue
        print("%-5s %-14s r%d pt=%-6s ct=%-5s ttft=%6.2fs pf=%6.0f tps=%6.1f ms/rd=%5.1f tok/rd=%5.2f acc=%s "
              "util=%3.0f%% mib=%5.0f %s%s" % (
                  s["mode"], s["workload"], s["rep"], s.get("prompt_tokens"), s.get("completion_tokens"),
                  s.get("ttft_s") or 0, s.get("prefill_tps") or 0, s.get("decode_tps") or 0,
                  s.get("ms_per_round") or 0, s.get("tokens_per_round") or 0,
                  "%.3f" % s["acceptance"] if s.get("acceptance") is not None else "  -  ",
                  s.get("gpu_util_mean") or 0, s.get("gpu_mib_peak") or 0, s.get("check"),
                  "" if "matches_k0" not in s else (" =K0" if s["matches_k0"] else
                                                   f" DIFF@{s['first_divergent_char']}")), flush=True)


if __name__ == "__main__":
    main()
