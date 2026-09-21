# Cinference

Cinference is a focused fork of **[NInfer by Neroued](https://github.com/Neroued/ninfer)**,
not a new engine built from scratch. NInfer provides the C++/CUDA runtime, kernels, model
support, CLI, HTTP APIs, conversion tools, and technical documentation. This fork extends
MTP speculative decoding to **10 draft tokens** using **capture-derived CUDA Graph topology
classes**. It remains specialized for a single NVIDIA GeForce RTX 5090.

Modified by satellitedown for Cinference. The Apache-2.0 [LICENSE](LICENSE), upstream and
third-party attribution in [NOTICE](NOTICE), and exact [source provenance](upstream-provenance.json)
are included. Native names stay unchanged: `ninfer`, `ninfer-serve`, `ninfer-perplexity`,
`ninfer_bench`, the `ninfer` C++ namespace, and the v3 `.ninfer` artifact format.

For a one-menu install and long-context server profile, use the separate
**[fast-long-context-cinference recipe](https://github.com/satellitedown/fast-long-context-cinference)**.
That recipe uses the non-Swift Huihui NVFP4 checkpoint. The measurements below use Swift instead.

## What changed

Upstream's MTP draft cap was five. Cinference raises the product cap and MTP round buffers and
contracts to ten. MTP graph profiles are classified from captured node-type/kernel-function
signatures, with batch size kept distinct, rather than assuming that planned frontier buckets
share a graph topology. This lets the MTP path keep separate CUDA Graph executables when larger
draft windows cross internal execution-route boundaries. It is not a replacement sampling
algorithm, a new quantization format, or a universal speed multiplier.

DFlash/DFlash2 limits remain 1..15. Architecture, API symbols, executable names, and build
organization remain those of NInfer. The inherited source changes and original commit IDs are
listed in [upstream-provenance.json](upstream-provenance.json).

## Quick start

Requirements: 64-bit Linux, RTX 5090, a CUDA toolkit supporting `sm_120a`, CMake 3.28+,
a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries (`libavformat`,
`libavcodec`, `libavutil`, `libswscale`), and `libcurl >= 7.85`. CUDA 13.1 is the
upstream-validated development toolkit; the build rejects architectures other than `sm_120a`.

```bash
git clone https://github.com/satellitedown/cinference.git
cd cinference
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

This builds the product binaries under `build/apps/`. There is no install target or packaged
binary distribution. Tests and benchmarks are excluded by default; `cmake --preset dev` enables
them and finds Python 3. See [build configuration](docs/maintainer/build-system.md).
Machine-specific paths belong in the ignored `CMakeUserPresets.json`.

The engine requires **v3 `.ninfer` artifacts**. Obtain a supported artifact from the
[upstream model list](docs/README.md#model-artifacts), or use the one-menu recipe above for
pinned Huihui download and conversion. For example, download the upstream Qwen3.8-27B NVFP4
artifact with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir models
```

If the download is v2, [upgrade it locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact)
with Python 3.11 or newer before serving. The output must be a different file:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  models/qwen3_8_27b_nvfp4.ninfer models/qwen3_8_27b_nvfp4.v3.ninfer
```

Run a server with an explicit path to your **v3** artifact (substitute its actual path):

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.v3.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 32768 --max-concurrency 1 --kv-dtype k8v4 \
  --spec mtp --draft-tokens 10 --lm-head-draft
```

This is a small-context manual example, not the long-context recipe or a benchmark command.
Use [HTTP serving](docs/serving.md) for request formats and resource controls, [CLI](docs/cli.md)
for one-shot generation, and each binary's `--help` for options. Do not expose the server publicly
without your own access controls.

## Performance

### Swift archive recall on RTX 5090

These are supplied observations from a 32 GiB RTX 5090 using
[Dragoy/Swift-Qwen3.8-27B-abliterated-NVFP4-NInfer](https://huggingface.co/Dragoy/Swift-Qwen3.8-27B-abliterated-NVFP4-NInfer/tree/4c25aa201b412fe3850e7c0c216c42da935aba38)
at revision `4c25aa201b412fe3850e7c0c216c42da935aba38`.
The workload was deterministic **archive recall**, with six synthetic markers at 1%, 10%, 25%,
50%, 75%, and 95% depth. Each row is **one cold-prefix request**, not an average.

Settings: **K8V4 KV, MTP-10**, optimized proposal head (`--lm-head-draft`), thinking off,
temperature 0, concurrency 1, 1,024-token prefill chunks, 262,144 maximum context, and at most
256 output tokens. Full settings and metric definitions are in
[the machine-readable result](results/rtx5090-archive-recall.json).

| Prompt tokens | Prefill seconds | Prefill tok/s | Decode tok/s | Draft acceptance | Marker recall |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.90 | 9,092.7 | 450.78 | 93.1% | 6/6 |
| 32,768 | 4.39 | 7,452.6 | 432.60 | 92.3% | 6/6 |
| 131,072 | 32.40 | 4,044.2 | 364.84 | 97.7% | 6/6 |
| 260,000 | 106.72 | 2,435.9 | 299.64 | 98.5% | 6/6 |

Prompt counts are tokenizer-sized requests; the native prompt timer reports 40 fewer processed
tokens per row. Decode is the server's `predicted_per_second`, calculated as
`(generated tokens - 1) / decode seconds`. It excludes prompt processing and request overhead.
The recall score only counts recovered synthetic markers, not general model capability.

**These are not measurements of the new non-Swift Huihui recipe.** Repetitive recall yields high
MTP acceptance; ordinary prose, different prompts, and lower acceptance can be much slower.
A prior 260,000-token repeat measured 275.73 tok/s. These individual observations are not a
universal speed guarantee or evidence of an across-the-board gain over upstream NInfer.

### Upstream NInfer measurements

The retained [performance index](docs/performance.md), [benchmark guide](bench/README.md), and
per-model reports describe upstream NInfer workloads and measurements. They are not fresh
Cinference MTP-10 or Huihui benchmarks. Their own models, draft windows, concurrency, timing
boundaries, and limitations apply.

## Evaluation

Historical upstream capability scores and their conditions remain in the
[model cards](docs/README.md#model-artifacts) and [evaluation guide](eval/README.md).
They do not establish capability scores for this fork's MTP-10 profile or the Huihui recipe.

## Capabilities and limits

The inherited engine supports Qwen3.5 Dense and MoE architectures, including compatible
Qwen3.6/3.8 artifacts, text/image/video input, OpenAI- and Anthropic-compatible serving,
chunked prefill, prefix reuse, offline perplexity scoring, and BF16/INT8/FP8/NVFP4/K8V4 KV storage.
Optional Vision and speculative components must be present in the artifact and enabled at startup.

The runtime uses one GPU, one resident model, and one to eight startup-fixed active-request lanes.
It does not provide multi-GPU execution, weight offload, request preemption, or distributed
serving. Context capacity depends on the selected artifact and memory profile. Returned tool calls
are for the client to execute; the engine does not execute tools.

## Documentation

The upstream technical guides are retained and applicable fork-specific ranges are updated:

- [Documentation map and upstream model artifacts](docs/README.md)
- [CLI](docs/cli.md) and [HTTP serving](docs/serving.md)
- [Build system](docs/maintainer/build-system.md) and [engine architecture](docs/maintainer/engine-architecture.md)
- [Weight conversion](docs/weight-conversion.md) and [perplexity](docs/perplexity.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Upstream performance records](docs/performance.md) and [measurement methodology](docs/performance/methodology.md)
- [CLI examples](examples/cli/), [benchmarks](bench/README.md), and [contributing](CONTRIBUTING.md)
