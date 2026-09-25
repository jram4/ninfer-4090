# tools/bench

Maintainer orchestration for the public `ninfer_bench` throughput tool, serving corpus/concurrency
runners, and the external Serve TTFT client. Correctness is owned by the affected suites under
[`tests/`](../../tests/README.md).

## E0002-H05 public width probe

[`h05_width_aba.py`](h05_width_aba.py) runs one isolated, **NON-QUALIFYING** positive-temperature
K7-A → K3 → K7-B Serve experiment. It uses only the checked-in async Python engineering prompt
from `examples/cli/` and the public tool-call case in
[`fixtures/h05_public_engineering.json`](fixtures/h05_public_engineering.json). Each case uses three
fixed request seeds. Code and tool requests alternate in a fixed order, and the same order and
request-body SHA-256 values repeat in all three arms. The report stores hashes, token lengths,
finish reasons, tool-call counts/names, timings, and GPU state; it never stores request messages,
generated text, or tool arguments.

The checked sampler is temperature `0.6`, top-p `0.95`, top-k `20`, min-p `0`, presence penalty
`1.0`, and frequency penalty `0`. Each request explicitly sets all six values and its seed. The
client sends no stop strings. Thinking is explicitly disabled. The code case retains its public
4,096-token output budget; the fixed tool-call case has a 96-token budget. An output-limit finish,
missing public quality markers, or an incorrect/missing structured tool call invalidates the run.
The tool is described to the model but never executed.

Every arm starts a fresh Serve process with the same exact `.ninfer` path and SHA-256, request
corpus, cache, graph, and runtime settings. It uses MTP with the optimized proposal head, K8V4 KV,
188,416 context and KV capacity, prefill chunks of 1,024, concurrency one, CUDA Graph enabled,
prefix reuse disabled, and one unmeasured eight-token warmup. Only draft width changes. Request
logs must confirm each actual sampler, seed, width, root-only prefix path, artifact, and GPU; GPU
UUID, binary SHA-256, CMake cache SHA-256, and source revision are held fixed across arms.

`ninfer-serve` currently exposes request-level `request_done` aggregates but not the CLI's
`--decode-research-jsonl` option. The runner's default `--decode-events auto` checks Serve help;
when the server supports that flag, it captures and validates the content-free H03 stream and
reports actual draft extent and extent-conditioned accepted/committed survival. Otherwise it marks
the report `request_aggregate_fallback`: request logs still support coarse committed yield and
round-cost triage, while exact per-position survival is unavailable. H03-captured results remain
exploratory because observer overhead has not been qualified for this Serve route.

Analysis uses completion tokens after the first prefill token for committed decode rates, divided
by request decode seconds. Tokens per round and milliseconds per round are ratio-of-sums. It also
reports draft acceptance and the raw accepted-position counts from request aggregates. Before any
axis decision, all requests must succeed with matching configuration and prompt lengths, the
three arm output-token totals must be within 15% of K3, and no paired request may differ by more
than 35% in output length. The 35% output-length guard applies to K7 versus K3 and to the K7-A/K7-B
repeat. The paired geometric-mean K7-B/K7-A decode-rate and round-time drift must each stay within
5%, with no individual pair exceeding 20% drift on either metric. After-request GPU temperature may
move at most 8°C, and graphics clock at most 5%; every measured request needs a complete
post-request temperature/clock sample. Missing thermal samples, configuration drift, failed
requests, a failed tool/quality check, output-length drift, or a K7 control outside these limits
makes the result inconclusive.

With every gate passing, the next axis is `acceptance-axis` only if geometric-mean K7 committed
tokens per round and decode tok/s each exceed K3 by at least 5%. It is `latency-axis` only if K7
round time is at least 5% higher, K7 committed tokens per round improve by less than 5%, and K7
decode tok/s improves by no more than 5%. Every other result is inconclusive. These criteria
select only a follow-up research axis; they do not qualify a production width or performance claim.

### FNN runbook

Run only after the isolated H03 CUDA integration and its focused tests pass. Use the exact checkout
and build directory recorded by the completed H03/FNN report; the path below is the current LW10
isolated root, not the protected production checkout. Do not select a checkout, build, binary, or
artifact by glob or modification time.

```bash
H03_ROOT=/home/aramirezfamily/.local/share/ninfer-e0002-lw10-d3ccf352d55b-clean
BUILD_DIR="$H03_ROOT/build"
SERVE_BIN="$BUILD_DIR/apps/ninfer-serve"
ARTIFACT=/home/aramirezfamily/josh/cinference-4090/models/qwen3_8_27b.ablit-wxp.ninfer
ARTIFACT_SHA256=b7c6693244cf7ab08554fff41c4f3c67695b6ec388c99eed6f12e0e71046fda8
FNN_H05_ROOT=/home/aramirezfamily/.local/share/ninfer/e0002-lw11-h05-r1

install -d -m 700 "$FNN_H05_ROOT"
cmake --build "$BUILD_DIR" -j --target ninfer-serve
ctest --test-dir "$BUILD_DIR" --output-on-failure -R '^ninfer_h05_width_aba_test$'
python3.11 tools/bench/h05_width_aba.py run \
  --serve "$SERVE_BIN" \
  --artifact "$ARTIFACT" \
  --artifact-sha256 "$ARTIFACT_SHA256" \
  --cmake-cache "$BUILD_DIR/CMakeCache.txt" \
  --output "$FNN_H05_ROOT/raw" \
  --port 18082 \
  --decode-events auto
```

The artifact path and hash above are the exact values recorded by the FNN preflight; the runner
rechecks the file before and between arms. Confirm the completed H03 report's checkout/build path
before using the two `H03_ROOT`/`BUILD_DIR` assignments. The runner creates the new `raw` directory
with owner-only permissions and fails if it already contains files. It writes one body-free
`run.json`, per-arm Serve request logs, private stderr logs, request summaries, and `analysis.json`.
It stops after that single ABA run. No production service is readied, restarted, or modified.

## External Serve TTFT

[`ttft/README.md`](ttft/README.md) defines the black-box latency benchmark. The measurement runner
uses frozen text/media requests and public streaming protocols without calling Engine. A separate
controller manages the fixed Qwen3.8-27B NVFP4/FP8 Serve profiles and fresh-process isolation.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

The controller chooses the profile, starts and stops Serve for every sample, runs the external
client, stages the NVFP4 artifact once in `/dev/shm`, stores raw/progress/Serve artifacts below
`profiles/bench/ttft/`, records structured per-request Serve diagnostics, and writes Markdown,
JSON, and CSV summaries. The case catalog, exact profiles, TTFT boundary, and fixture qualification
are documented in the dedicated README.

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v15. The flattened summary and schema-v4 matrix
manifest carry native facts from the report: architecture, public name, actual formats, prefill signature, artifact,
load/read/upload/staging values, Engine memory arenas including the non-additive Vision layout
inside the unified workspace and CUDA Graph allowance, per-test planned logical and
allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

## Serving corpus benchmark

[Published coverage and model results](../../docs/performance.md) identify the recorded runs.
The [serving methodology](../../docs/performance/methodology.md) owns workload definitions,
metric boundaries, aggregation, comparison rules, and publication format. This section describes
runner usage and output files.

`run_serve_corpus.py` accepts explicit `--artifact LABEL=PATH` entries. Labels identify report groups;
the selected artifact supplies the architecture, public name and weight bindings.
Omitting `--mode` selects MTP0 and MTP3; repeat `--mode` to select a subset. Use `dflash7` for
Qwen3.6-35B-A3B DFlash K=7 and `dflash2_7` for Qwen3.8-27B DFlash2 K=7, with companion weights
in the selected artifact. `--sampling greedy` selects exact argmax; the default is stochastic.
Run commands with a selected Python 3.11 interpreter, as in the model-page reproduction entries.

The serial runner writes `run.jsonl`, `summary.csv`, `summary.md`, and per-server logs under
`server/`. JSONL contains the completed requests and responses; CSV/Markdown contain fixture and
category summaries. The output directory is supplied explicitly with `--output`.

Its schema-v7 result and flattened summaries retain the actual `prefill_signature`, request Host
exposure, and decode Host/Device-wait time per round received from the schema-v21 serving records.
Request exposure is a latency distribution value and is never summed across concurrent requests;
worker aggregation uses the serving `throughput.host_work` interval deltas. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## Concurrent serving benchmark

`run_serve_concurrency.py` selects `--suite decode-saturation` or `--suite corpus-makespan`.
Their distinct time boundaries and workload dispatch are defined in the
[serving methodology](../../docs/performance/methodology.md#workloads-and-measurement-boundaries).
Repeat `--concurrency` to select C points; each point starts a fresh server. The point report
records the actual Engine configuration, automatic KV capacity, shuffle seed where applicable,
dispatch method, and per-request positions.

Schema-v3 outputs include `points/*.json`, `server/*.jsonl`, and combined `summary.json`, `summary.csv`, and
`summary.md`. Corpus runs also write complete responses in `corpus/<point>/results.jsonl` and
per-request phase summaries in that directory; older campaigns may have only point reports and
server logs. Historical model pages identify the report directory associated with each table.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.
