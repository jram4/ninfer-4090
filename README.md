# NInfer for RTX 4090 — Qwen3.6-27B on Linux

Current runtime revision: **`0.2.0-rtx4090-v1`**.

> Qwen3.6-27B inference for one 24 GiB NVIDIA GeForce RTX 4090 on Linux.

This project is an RTX 4090 (`sm_89`) Linux port of
[Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090), which is itself derived from
[Neroued/ninfer](https://github.com/Neroued/ninfer). It preserves the inherited `.ninfer`
artifact format, C++ Engine API, CLI, MTP speculative decoding, CUDA Graph execution, and
OpenAI-/Anthropic-compatible HTTP schemas. This fork contributes the Ada architecture port,
RTX 4090 dispatch measurements, retained Q5 kernel changes, a Qwen3.6-27B-only product surface,
and the accompanying qualification evidence.

The source builds only for compute capability 8.9; CMake rejects every other CUDA architecture.
This repository does not claim authorship of NInfer, MTP, CUDA Graph decoding, the artifact
format, or the HTTP compatibility layers.

The verified runtime registers exactly one artifact:

| Model | Artifact | Supported 24 GiB route | SHA-256 |
|---|---|---|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `qwen3_6_27b.ninfer` | Text, image/video Vision, MTP, prefix reuse, CUDA Graphs | `74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127` |

The model bytes, model ID, quantization, and frontend contract are unchanged by this port.
Source for other checkpoints remains dormant and is not compiled, registered, tested, packaged,
downloaded, or part of this product contract.

## Verified platform

- NVIDIA GeForce RTX 4090, 24 GiB, compute capability 8.9;
- 128 SMs, 65,536 registers/SM, 100 KiB shared memory/SM;
- 64-bit Linux;
- CUDA Toolkit 13.0 and CUDA 13 Nsight tools;
- GCC/G++ 13, CMake 4.4, Ninja;
- Python 3.11;
- vcpkg manifest dependencies at the baseline pinned in `vcpkg.json`.

See [RTX 4090 Linux qualification](docs/rtx-4090-linux.md) for the exact device record,
architecture-sensitive scheduling evidence, tests, and measured baselines.

## What this fork changes

- ports the compile-time and runtime contract from RTX 3090 `sm_86` to RTX 4090 `sm_89`;
- retunes Gated-DeltaNet cooperative dispatch around Ada's 128 SMs;
- retains measured Q5 residual improvements while rejecting slower projection candidates;
- removes Qwen3.6-35B-A3B and sparse-MoE code from the compiled/registered product;
- qualifies Qwen3.6-27B text, Vision, MTP-3, prefix reuse, and CUDA Graph routes on Linux;
- narrows packaging, tests, examples, and active documentation to the supported product.

## Measured RTX 4090 results

The controlled end-to-end workload uses `pp512`, `tg128`, 4,096-token capacity, INT8 KV,
CUDA Graphs, two warmups, and five measured repetitions on one RTX 4090:

| Mode | Prefill | Decode |
|---|---:|---:|
| Ordinary | 2,119.577 tok/s | 51.716 tok/s |
| MTP-3 optimized | 2,083.809 tok/s | 124.574 tok/s |

MTP and its optimized proposal head are inherited engine capabilities; the comparison is a
qualification result, not a claim that this fork invented the speedup. The retained local kernel
work improved the measured ordinary decode baseline by 1.38% and MTP-3 decode by 0.12%.
Curated reports are tracked under [`benchmark-results/rtx4090`](benchmark-results/rtx4090/).

## Build from source

The commands below deliberately select the verified tools instead of relying on older system
defaults. They bootstrap vcpkg, whose manifest pins the dependency baseline, and build the public
CLI and server without requiring a model checkpoint.

```bash
git clone https://github.com/jram4/ninfer-4090.git
cd ninfer-4090

git clone https://github.com/microsoft/vcpkg.git "$HOME/.local/share/vcpkg"
git -C "$HOME/.local/share/vcpkg" checkout 4bca8fd8654e5ba76f92661db7bfe954768ad8ef
"$HOME/.local/share/vcpkg/bootstrap-vcpkg.sh" -disableMetrics

export PATH=/usr/local/cuda-13/bin:$HOME/.local/bin:$PATH
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda-13/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-13

cmake -S . -B build-sm89 -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/.local/share/vcpkg/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-linux \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=OFF

cmake --build build-sm89 --parallel 2
```

The manifest limits FFmpeg to `avcodec`, `avformat`, `swscale`, and zlib. Curl retains HTTPS
support. Linux builds can still use the normal pkg-config dependency route when no vcpkg
toolchain is selected.

Maintainers enabling the CUDA and Python suites should follow [the test guide](tests/README.md).

## Download and verify the artifacts

```bash
mkdir -p models

curl -L --fail --retry 8 --retry-all-errors --continue-at - \
  -o models/qwen3_6_27b.ninfer \
  https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/main/qwen3_6_27b.ninfer

printf '%s  %s\n' \
  74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127 \
  models/qwen3_6_27b.ninfer | sha256sum --check
```

Never execute an artifact until its checksum passes.

## Run the CLI

```bash
./build-sm89/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain speculative decoding in three sentences." \
  --max-context 4096 \
  --max-new 128 \
  --kv-dtype int8 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

Use `--messages examples/cli/messages/image_chart.json` for a committed multimodal example.

Answer content is written to stdout. Loading, timing, throughput, memory, CUDA Graph, and MTP
statistics are written to stderr.

## Run the HTTP server

```bash
./build-sm89/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --model-id qwen3.6-27b \
  --max-context 4096 \
  --kv-dtype int8 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

The server implements OpenAI Chat Completions and Anthropic Messages, including streaming,
usage accounting, multimodal input on 27B, and function-tool request/response translation. See
[HTTP serving](docs/serving.md).

## Current limits

- Execution is specialized for one RTX 4090 and one CUDA device.
- The verified baseline capacity is 4,096 tokens. Larger capacities remain subject to the
  runtime memory-budget check.
- One Engine owns one resident sequence and runs one active request at a time.
- Continuous batching, multi-GPU execution, CPU/GPU offload, and distributed serving are not
  implemented.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- Results describe one qualification machine and are not a cross-GPU performance guarantee.

## Release status

`v0.2.0-rtx4090-v1` is published as a source release. Model weights and locally built binaries
are intentionally excluded. The recorded hardware evidence predates the public-source
sanitization pass; no post-sanitization binary is presented as qualified. A future binary release
requires a clean isolated RTX 4090 rebuild and acceptance run.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for supported changes, verification expectations, and the
required hardware/toolchain details for performance work. Report security issues using the
private process in [SECURITY.md](SECURITY.md), not a public issue.

## Documentation

- [Documentation index](docs/README.md)
- [RTX 4090 Linux qualification](docs/rtx-4090-linux.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Benchmarks](bench/README.md)
- [Curated RTX 4090 evidence](benchmark-results/rtx4090/README.md)
- [Tests](tests/README.md)

The earlier RTX 3090 reports remain available as explicitly historical port evidence:

- [RTX 3090 / WSL2 report](docs/rtx-3090-wsl.md)
- [RTX 3090 native Windows report](docs/rtx-3090-windows.md)
- [RTX 3090 ordinary inference analysis](docs/rtx-3090-normal-inference.md)
- [RTX 3090 35B-A3B report](docs/rtx-3090-35b-a3b.md)

They do not describe the current product contract or current runtime dispatch.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE). Published model artifacts derive from
the Apache-2.0 Qwen3.6 checkpoints. Vendored dependencies retain their own license files under
`third_party/`.
