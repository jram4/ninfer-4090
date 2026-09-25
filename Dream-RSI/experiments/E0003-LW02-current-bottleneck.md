# E0003-LW02 current bottleneck profile

**Status: incomplete; one fresh public request was measured and profiled.** The user-directed wind-down arrived after the one successful capture. No second workload, clarification capture, optimization, or H02 comparison was started. H02 is **unadjudicated** by this record.

## Scope and provenance

- Source base: `c2b1796d7de02cffa8f418c576e54ed4f6ab83dc` (exact checkout base).
- FNN workspace identity: `e0003-lw02-c2b1796d7de0-29986ddeabac`; owner-only root. The identity suffix records the original instrumentation patch SHA-256 `29986ddeabac1538ecfcf888eef7b3c6d96a047cd1c1b8d0b03dd95b283d3f62`.
- Instrumentation-only C++ additions: NVTX boundaries for MTP alignment, sampled proposal work, and the target head. No mathematical or runtime behavior was changed.
- The successful capture used an owner-only runner copy with SHA-256 `2d9df9bd23fac9ee966d5d1e7e0f4bf4da6918febc19bd844a913b0c59c758ba`. It differs from the original runner in the supported Nsight option (`--cuda-graph-trace=node:host-only`); the corrected runner is the committed `tools/research/e0003_lw02_profile.py`. The preceding invocation using `node:host-only:nvtx-live` was rejected before starting the server and issued no request.
- Model artifact: `qwen3_8_27b.ablit-wxp.ninfer`, 18,210,704,128 bytes, SHA-256 `b7c6693244cf7ab08554fff41c4f3c67695b6ec388c99eed6f12e0e71046fda8` (pinned during read-only preflight; not rehashed during workspace setup).
- No H05 corpus or private cohort was opened or used. The request body, response body, raw request logs, server logs, and Nsight trace/database remain in the owner-only FNN root and are not included in this repository record.

## Build and environment

- Host: Pop!_OS 22.04, kernel `7.0.11-76070011-generic`, AMD Ryzen 9 7900X (24 logical CPUs).
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, UUID `GPU-008f1b73-105e-b50b-7469-ddca95c233f6`, driver `580.173.02`, 24,564 MiB.
- Release build: CMake 4.4.0, Ninja 1.13.0, GCC/G++ 13.4.0, NVCC 13.0.88, CUDA runtime 13.0, `CMAKE_CUDA_ARCHITECTURES=89`, `NINFER_BUILD_APPS=ON`, tests and benchmarks off. All 438 Ninja steps completed; `ninfer-serve` linked successfully.
- `ninfer-serve`: 217,496,832 bytes, SHA-256 `950ac3b601ee52493fab56c43f74ef11705c7c83de60ee709b9a20458e6e58fa`. `CMakeCache.txt` SHA-256 `9f0772f86098917eef10846751472fc7212adf75329db85ec08624bdf2875c8f`.
- FNN’s system libcurl 7.81 and FFmpeg 4.4 headers were below the source requirements. Signed curl 8.22.0 and full shared FFmpeg 9.0.2 were built under the owner-only root; OpenSSL development files were extracted privately. No system packages were installed or changed. FFmpeg’s tarball SHA-256 is `8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e`, with a good GPG signature verified against official key fingerprint `FCF986EA15E6E293A5644F10B4322F04D67658D8`. Installed FFmpeg module versions were libavformat/libavcodec `63.1.102`, libavutil `61.1.102`, and libswscale `10.1.102`. See [FFmpeg release and verification information](https://ffmpeg.org/download.html).
- Curl 8.22.0’s release signature was verified against the official curl signing-key fingerprint `27EDEAF22F3ABCEB50DB9A125CC908FDB71E12C2`; its source tarball SHA-256 is `f7ef3ae8a22e521f289803fe93543eb64c329b58aa73a9e224dfd915a2a5f4f7`. See [curl release verification](https://curl.se/docs/verify.html).

## Workload and unprofiled reference

One fresh server process handled one synthetic public engineering request through `/v1/chat/completions`; this was not an H05 prompt. The prompt was a short acceptance-criteria task (77 prompt tokens); its body is omitted here. Model `qwen/qwen3.8-27b`; max output 128; `temperature=1.0`, seed `7362050925`, `top_p=0.8`, `top_k=20`, presence penalty 1.5, frequency penalty 0, min-p 0, thinking disabled. Startup-fixed settings matched the configured production route: max context/KV capacity 188,416, K8V4 KV, concurrency 1, prefill chunk 1,024, MTP window 7 with `lm-head-draft`, CUDA Graphs enabled, default prefix reuse enabled with zero prefix-hit tokens in this fresh server. All three production systemd units remained inactive before the request; GPU baseline use was 170 MiB.

- HTTP 200; 128 completion tokens; 77 prompt tokens; zero cached prompt tokens; request wall 1.5860 s; TTFT 197.1 ms; prefill 149.0 ms; decode 1.3881 s. The server’s aggregate decode throughput field was 79.99 tokens/s. Throughput uses this unprofiled reference only.
- Engine counters: 52 MTP rounds; 352 drafted tokens; 75 accepted tokens; zero fallback steps; accepted-token positions `[37, 15, 8, 6, 5, 3, 1]`; average 2.44 committed decode tokens per round. Request, prompt, and completion SHA-256 values matched the profiled run: `ce0cf2a16fab8b874058e32ef8eaf117f0cc97de59d983044c6edf7374813eb9`, `56e40d89adbf7c01e23962a274fc6e53650a9fc8f887c5360a03501ea9b06e24`, and `45c75ea5648966265e6f2b84f2de8e097adcfae07423a04bd3994de29c514ea0`. Bodies are not reproduced.

## Profile attribution

One current Nsight Systems capture ran the identical request (same request/output hashes). Installed Nsight Systems was 2025.3.2.474; CUDA graph nodes, CUDA, NVTX, and OS runtime were traced. The capture exported 118,694 events. The server logged 1.6254 s request wall under profiling; this is not a throughput result.

Two graph warmup rounds preceded the request. Filtering NVTX ranges at the logged request-start timestamp leaves the 52 request rounds:

| Request-round attribution | Total | Mean per round | Share of outer round wall |
|---|---:|---:|---:|
| `decode.mtp_round` wall | 1.4070 s | 27.06 ms | 100% |
| `decode.mtp.submit` | 28.89 ms | 0.556 ms | 2.05% |
| `decode.mtp.wait` | 1.3779 s | 26.50 ms | 97.93% |
| CUDA `cudaStreamSynchronize` inside wait | 1.3778 s over 52 calls | 26.50 ms/call | 97.93% |

Summed graph-kernel duration intersecting those rounds was 1.3721 s across 45,396 events; interval union was 1.3716 s. Kernel-duration shares below use the summed graph-kernel duration denominator (small overlap, about 0.04%). Categories come from demangled kernel-name families; they are not separate end-to-end speedup estimates.

| Kernel family | Share of summed graph-kernel time |
|---|---:|
| Linear/GEMM | 78.49% |
| GDN/recurrent | 12.32% |
| Vocabulary projection | 5.45% |
| Norm/position/cache/other | 1.65% |
| Attention | 1.46% |
| Sampling/MTP control | 0.62% |

The request is GPU-compute dominated, with linear/GEMM kernels the largest measured family. The trace supports that family-level attribution, but this short one-request sample does not isolate a single kernel’s removable end-to-end share, establish an optimization’s gain, or generalize across contexts. Therefore it cannot adjudicate H02 or justify a latency-only candidate against the `0.245` share bound. No further profile was run.

## Evidence location and remaining work

Owner-only raw artifacts are under the FNN identity root, relative to that root: `results/reference/`, `results/profile/`, `results/ninfer-build.log`, and `results/ffmpeg-build.log`. These contain the logs, measurements, Nsight report, and SQLite export; they remain private and are excluded from Git. This checked-in record contains only aggregate, public-safe evidence.

Completed: exact-base checkout; private dependency verification/build; Release/sm_89 application build; one positive-temperature reference request; one corrected one-request graph-node capture; profile counter and NVTX/kernel-timeline extraction; production-unit inactive checks. Not run: unit tests (not requested), any optimizer or candidate experiment, any second/longer/profile control request, context-length stratification, whole-inference benchmark campaign, or H02 adjudication. The parent should treat this return as a bounded observation, not a completed optimization decision.
