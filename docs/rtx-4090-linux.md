# RTX 4090 Qwen3.6-27B Linux qualification

This report records the `0.2.0-rtx4090-v1` bring-up on a dedicated RTX 4090 qualification host.
It is the active product evidence for the Linux-only `sm_89` runtime. Earlier RTX 3090 and
upstream RTX 5090 reports remain historical references; they do not define current dispatch.

## Verified system

| Component | Verified value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GiB |
| Compute capability | 8.9 (`sm_89`) |
| SMs | 128 |
| Registers | 65,536 per SM |
| Shared memory | 102,400 bytes per SM; 101,376-byte opt-in block limit |
| L2 cache | 75,497,472 bytes |
| Cooperative launch | supported |
| Driver | 580.173.02 |
| OS | Pop!_OS 22.04, Linux 7.0.11 |
| CPU / RAM | Ryzen 9 7900X / 30 GiB |
| CUDA | 13.0.88 from `/usr/local/cuda-13` |
| Host compiler | GCC/G++ 13.4 |
| Build system | CMake 4.4, Ninja |
| Python | CPython 3.11.15 installed with `uv` |
| Dependencies | pinned vcpkg manifest, x64-linux |

The configure explicitly sets `CMAKE_CUDA_ARCHITECTURES=89` and
`CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`. CMake rejects every other CUDA architecture.

## GDN cooperative residency

CUDA 13 resource output for the emitted `sm_89` kernels gives:

| Geometry / schedule | Threads | Registers/thread | Dynamic shared memory | Resident CTAs/SM | Device-wide CTAs |
|---|---:|---:|---:|---:|---:|
| 27B split8 | 256 | 68 | 40 KiB | 2 | 256 |
| 27B split4/split2 | 512 | 74 | 40 KiB | 1 | 128 |

Every legal split candidate was measured at route boundaries with ten warmups, fifty
cold-cache repetitions, and a 256 MiB cache flush. A route changed only when its median was at
least 2% faster; ties retained the simpler incumbent.

The resulting production routes are:

| Target | Token columns | Route |
|---|---:|---|
| 27B | 1 | paired-row GEMV |
| 27B | 2–8 | small-T split10 |
| 27B | 9–1,151 | cooperative split8 |
| 27B | 1,152–1,280 | cooperative split4 |
| 27B | 1,281–2,688 | cooperative split2 |
| 27B | 2,689+ | unsplit |

The corresponding public workspace high-water mark is 3,535,872 bytes.

## Model routes

- Qwen3.6-27B: 4,096-token capacity, INT8 KV for the recorded baseline, Vision enabled,
  CUDA Graphs, MTP-3 with optimized proposal head, and prefix reuse.
- Qwen3.6-35B-A3B is intentionally dormant. Its source remains for a future port, but this build
  does not compile, register, test, package, download, or advertise that target.

The verified artifact is `models/qwen3_6_27b.ninfer`, 17,495,365,888 bytes, with SHA-256:

```text
74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127
```

The first public `v0.2.0-rtx4090-v1` release is source-only. A binary bundle requires a separate
clean hardware rebuild and acceptance run.

## Functional acceptance

The final two-job Release build completed successfully. The sequential native suite passed all 58
tests, including the opt-in real-artifact integration. That integration exercised text, Vision,
MTP-3 with the optimized proposal head, prefix reuse, stop handling, and ordinary/MTP CUDA Graphs.
The active artifact and 27B Python suites passed 51 tests; five private-checkpoint fixture tests
were explicitly skipped because those external fixtures were unavailable.

Visible greedy CLI acceptance produced coherent non-empty output in both modes:

| Mode | Output tokens | Decode rate | MTP acceptance |
|---|---:|---:|---:|
| ordinary | 32 | 50.80 tok/s | disabled |
| MTP-3 optimized | 32 | 90.89 tok/s | 16/36 (44.44%) |

The committed multimodal fixture returned its expected chart facts, stopped on the model stop
token, and recorded 11/12 accepted draft tokens (91.67%). Its measured Vision and text-prefill
times were 0.024 s and 0.296 s both before and after the retained Q5 change.

The committed localhost server smoke contract passed OpenAI and Anthropic streaming, token
counting, tool translation, and multimodal requests. It reported 15 counted tokens, OpenAI
`finish_reason=stop`, two OpenAI completion tokens, 85 image prompt tokens, and Anthropic
`stop_reason=end_turn`. The server was terminated with `SIGTERM`, waited to completion, and no
persistent process remains.

## Ada optimization result

Nsight Systems showed that Q4 fused SwiGLU and three Q5 GEMV shapes dominate ordinary decode. The
same paths remain dominant with MTP-3; W8 proposal-head work is the next material MTP-only path.
Together the examined paths cover more than 90% of captured GPU kernel time.

The Q4 fused route was tested with eight warps, two pipeline stages, and `cp.async.cg`; none met the
2% cold-cache and end-to-end retention threshold. Q5 candidates covered 8/16/32 rows per block and
two/three stages. The 32-row form exceeded the legal static shared-memory limit and was rejected at
link time. The three-stage form regressed. Eight rows with two stages improved the
`[5120,17408]` fused residual operator median from 95.232 us to 93.184 us (2.15%) and was retained.

Final end-to-end results use `pp512`, `tg128`, two warmups, five measured repetitions, 4,096
context, INT8 KV, and CUDA Graphs:

| Mode / phase | Initial | Final | Change |
|---|---:|---:|---:|
| ordinary `pp512` | 2118.783 tok/s | 2119.577 tok/s | +0.04% |
| ordinary `tg128` | 51.013 tok/s | 51.716 tok/s | +1.38% |
| MTP-3 `pp512` | 2085.753 tok/s | 2083.809 tok/s | -0.09% |
| MTP-3 `tg128` | 124.430 tok/s | 124.574 tok/s | +0.12% |

The definitive MTP-3 rerun is tracked at
[`benchmark-results/rtx4090/rtx4090-27b-definitive-mtp3.json`](../benchmark-results/rtx4090/rtx4090-27b-definitive-mtp3.json).
It generated exactly 128 measured decode tokens
in every repetition and aggregated 420 accepted tokens from 645 drafts across 215 rounds: 65.12%
acceptance, with five fallback steps. The ordinary benchmark reserved 19,980,158,720 GPU bytes;
MTP-3 reserved 20,452,162,304 GPU bytes. Measured workspace peaks remained far below the existing
reservation, so no CUDA Graph reservation change was required.

Final bounded Nsight Systems captures contain 128 `cudaGraphLaunch` calls for ordinary decode and
44 for MTP-3, directly confirming graph replay. The leading ordinary kernel shares were:

| Kernel path | GPU time |
|---|---:|
| Q4 fused SwiGLU | 34.9% |
| Q5 `[5120,17408]` fused residual | 22.2% |
| Q5 `[6144,5120]` | 12.9% |
| Q5 `[5120,6144]` fused residual | 9.5% |
| Q6 K=5120 prefill launch | 5.3% |

Nsight Compute 2025.3.1 was invoked from `/usr/local/cuda-13/bin/ncu`, but the installed driver
denied hardware-counter access with `ERR_NVGPUCTRPERM`. No system security setting was changed.
CUDA 13 object resource output and Nsight Systems timing were used for the retained decision.

## Reproduction

Configure and build:

```bash
PATH=/usr/local/cuda-13/bin:$PATH \
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
$HOME/.local/bin/cmake -S . -B build-sm89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPython3_EXECUTABLE="$PWD/build-sm89/py311/bin/python" \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/.local/share/vcpkg/scripts/buildsystems/vcpkg.cmake" \
  -DBUILD_TESTING=ON \
  -DNINFER_BUILD_BENCHMARKS=ON
$HOME/.local/bin/cmake --build build-sm89 --parallel 2
```

The controlled end-to-end baseline uses `pp512`, `tg128`, two warmups, five measured
repetitions, INT8 KV, CUDA Graphs, and 4,096-token capacity. It records ordinary decode with MTP
disabled and MTP-3 with the optimized proposal head:

```bash
build-sm89/bench/ninfer_bench \
  --weights models/qwen3_6_27b.ninfer \
  -p 512 -n 128 --max-ctx 4096 --kv-dtype int8 \
  --warmup 2 --repetitions 5 --mtp-draft-tokens 0 \
  --output json --output-file profiles/bench/rtx4090-27b-final-mtp0.json

build-sm89/bench/ninfer_bench \
  --weights models/qwen3_6_27b.ninfer \
  -p 512 -n 128 --max-ctx 4096 --kv-dtype int8 \
  --warmup 2 --repetitions 5 --mtp-draft-tokens 3 --lm-head-draft \
  --output json --output-file profiles/bench/rtx4090-27b-definitive-mtp3.json
```

The corresponding bounded traces are:

```text
profiles/nsys/rtx4090-27b-final-tg128-mtp0.nsys-rep
profiles/nsys/rtx4090-27b-final-tg128-mtp3.nsys-rep
```

Curated JSON/CSV evidence is tracked under
[`benchmark-results/rtx4090/`](../benchmark-results/rtx4090/). Raw Nsight captures and local
working reports remain ignored under `profiles/` and are not distributed.

## Known limitations

- This product is Linux, RTX 4090, `sm_89`, and Qwen3.6-27B only.
- The verified memory/performance contract is 4,096 context; no larger capacity is claimed here.
- Execution is single-device and single-sequence; no batching, multi-GPU, or offload is provided.
- NCU hardware-counter evidence requires an administrator to enable profiling permissions.
- This is a measured baseline and retained local optimization, not a claim of global optimality or
  minimum throughput.
