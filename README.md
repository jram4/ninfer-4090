<!-- Modified for the Cinference RTX 4090 port. Upstream attribution is retained in NOTICE. -->

# Cinference 4090: Qwen3.8-27B inference optimized for a single RTX 4090

A C++/CUDA inference engine for Qwen3.8-27B on one RTX 4090 (24 GB, Ada, sm_89). It is a port of
[Cinference](https://github.com/satellitedown/cinference) (RTX 5090), which is derived from
[NInfer](https://github.com/Neroued/ninfer). See [Attribution](#attribution).

**Production configuration:** MTP speculative decoding with a fixed 7-token draft window (K7),
K8V4 KV cache (FP8 keys, NVFP4 values), 188,416-token engine context, concurrency 1.

| Workload (greedy, thinking off unless noted) | Prompt tokens | K0 tok/s | K7 tok/s | K7 draft acceptance |
|---|---:|---:|---:|---:|
| Needle recall | 8,219 | 51.7 | 275.2 | 0.99 |
| Needle recall | 130,421 | 42.9 | 226.1 | 0.99 |
| Needle recall | 189,004 | 39.7 | 209.4 | 0.99 |
| Structured JSON | 77 | 53.0 | 228.0 | 0.72 |
| Reasoning (thinking on) | 122 | 52.9 | 179.9 | 0.54 |
| Prose | 34 | 52.7 | 87.7 | 0.19 |

K0 is plain autoregressive decode (no speculation); it is the raw decode baseline. Speculative
throughput depends on how often the target model accepts the drafted tokens, so it varies by
roughly 3x across workloads. The 200+ tok/s figures apply to predictable output (retrieval,
structured data, code), not to open-ended prose.

## Results

Measured 2026-09-23 on one RTX 4090 with
[`tools/bench/cinf_matrix.py`](tools/bench/cinf_matrix.py). The raw per-request data is in
[`results/rtx4090-20260923/`](results/rtx4090-20260923/).

Decode tok/s: completion tokens after the first token, divided by decode wall time. Values are
medians of 2 repetitions for the short workloads and 1 run for the recall workloads.

| Workload | Prompt tokens | K0 | K3 | K5 | **K7** | K10 |
|---|---:|---:|---:|---:|---:|---:|
| short answer | 24 | 78.8 | 134.1 | 115.0 | **112.2** | 97.3 |
| structured JSON | 77 | 53.0 | 156.9 | 176.2 | **228.0** | 200.1 |
| code | 69 | 52.7 | 155.2 | 177.9 | **209.1** | 206.7 |
| reasoning (thinking on) | 122 | 52.9 | 153.0 | 150.0 | **179.9** | 176.4 |
| prose | 34 | 52.7 | 98.2 | 86.5 | **87.7** | 74.6 |
| long generation (~2.5K tokens) | 55 | 52.5 | 105.7 | 95.9 | **96.6** | 90.2 |
| needle recall 8K | 8,219 | 51.7 | 170.2 | 218.2 | **275.2** | 312.5 |
| needle recall 32K | 32,686 | 49.5 | 163.8 | 209.8 | **263.1** | 295.0 |
| needle recall 64K | 65,275 | 47.2 | 154.9 | 198.0 | **247.8** | 289.3 |
| needle recall 131K | 130,421 | 42.9 | 142.1 | 181.8 | **226.1** | 241.0 |
| needle recall 189K | 189,004 | 39.7 | 132.4 | 169.1 | **209.4** | 232.5 |

Round latency (one draft plus one verify step), in ms:

| Setting | Short prompt | 189K prompt |
|---|---:|---:|
| K0 (one token per step) | 18.9 | 25.2 |
| K3 | 22.2 | 30.2 |
| K7 | 26.6 | 38.2 |
| K10 | 30.5 | 45.9 |

Draft acceptance at K7:

| Workload | Acceptance |
|---|---:|
| recall | 0.99 |
| structured | 0.72 |
| code | 0.65 |
| reasoning | 0.54 |
| long generation | 0.23 |
| prose | 0.19 |

Prefill and time to first token (TTFT) at K7:

| Prompt tokens | TTFT | Prefill tok/s |
|---:|---:|---:|
| 8,219 | 3.6 s | 2,285 |
| 32,686 | 15.2 s | 2,153 |
| 65,275 | 33.1 s | 1,971 |
| 130,421 | 77.7 s | 1,680 |
| 189,004 | 127.4 s | 1,485 |

Prefill speed does not depend on K.

### Comparison with the previous 4090 engine

The previous FNN deployment was the [NInfer RTX 4090 port](https://github.com/jram4/ninfer-4090/tree/master)
running at K3 with its rk4v4-e8 KV cache. It was measured with the same tool and workloads
([`oldprod-k3-summary.json`](results/rtx4090-20260923/oldprod-k3-summary.json)).

| Workload | Previous (K3) | This (K7) | Change |
|---|---:|---:|---:|
| structured JSON | 151.4 | 228.0 | +51% |
| code | 154.9 | 209.1 | +35% |
| reasoning | 147.6 | 179.9 | +22% |
| prose | 99.0 | 87.7 | −11% |
| long generation | 102.9 | 96.6 | −6% |
| needle recall 8K | 171.4 | 275.2 | +61% |
| needle recall 32K | 156.4 | 263.1 | +68% |
| needle recall 131K | 134.5 | 226.1 | +68% |

Interpretation: a fixed K7 window helps when acceptance is high and costs throughput when it is
low. Each K7 round is about 4.4 ms slower than a K3 round, and low-acceptance text doesn't gain
enough extra accepted tokens to pay for it. K3 remains faster for prose and long free-form
generation. K7 was chosen because it leads or ties on structured output, code and reasoning,
and loses less on prose than K10.

## Methodology

- **Hardware:** one RTX 4090 (24 GB), Linux, CUDA 13.x. One server process per mode, started
  with `--greedy --no-prefix-reuse`, so every request prefills from scratch.
- **Workloads:**
  - short factual answer
  - JSON extraction (validated)
  - Python with unit tests (extracted and executed)
  - prose
  - arithmetic reasoning with thinking on (answer checked)
  - ~2.5K-token long generation
  - synthetic needle retrieval with 6 markers at 8K–189K prompt tokens
- **Recorded:**
  - TTFT, prefill tok/s and decode tok/s
  - ms per round and draft acceptance, from the server's request log
  - GPU memory and utilization, sampled every 0.5 s from `nvidia-smi`
  - KV fill, output SHA-256, and the first divergence from K0
- **Tool:**

```bash
python3 tools/bench/cinf_matrix.py --out bench_results_matrix \
  --modes k0,k3,k5,k7,k10 --recall-tokens 8192,32768,65536,131072,190000 \
  --reps 2 --max-context 196608 --kv-dtype k8v4 \
  --bin build-sm89/apps/ninfer-serve --model models/<artifact>.ninfer
```

The matrix used a 196,608-token capacity to reach the 189K prompt. Production uses 188,416
tokens for VRAM headroom (see below).

## Correctness

Measured facts:

- **Needle recall:** all 6 markers are retrieved at every context and every K. The output is
  byte-identical to K0.
- **Short answer:** the output is byte-identical to K0 at every K.
- **Structured JSON:** passes validation at every K. It is identical to K0 at K3, K5 and K10.
  At K7 it differs by one token (170 vs 171).
- **Prose, reasoning and long generation:** outputs diverge from K0 after the first few tokens
  at every K > 0, but all still pass their checks, including the reasoning answer.
- **Code:** the generated unit tests fail at every K, including K0, so this reflects the model's
  output rather than the speculative path.
- **Determinism:** repeated runs with the same mode are byte-identical.
- **KV value decode:** the new NVFP4 decode is bit-identical to the previous implementation
  over all 256 byte values and every legal E4M3 scale code. This was checked with an
  exhaustive device test during development; the test is not committed.
- **Attention tests:** `ninfer_softmax_attention_test --k8v4-only` passes, including the new
  verify widths 9–16, CUDA Graph replay, and contexts up to 32,768.

Interpretation: divergence under speculation comes from numerical differences between the
batched verify path and one-token decode. Greedy selection amplifies these into different
continuations. Retrieval outputs, which are strongly determined by the prompt, don't change.
Perplexity was not re-measured. Prefill changed only through the bit-identical V decode.

## Engine changes in this port

On top of the Cinference RTX 4090 port (MTP-10, Ada fallbacks, K8V4 cache):

- **Faster NVFP4 V-cache decode:** E2M1 nibbles are placed directly into FP16 bits and
  scaled with two `HMUL2`s, replacing a per-nibble switch
  (`src/ops/kv_cache/nvfp4_group16_codec.cuh`). Measured during development at 25K keys:
  T1 attention went from 240 to 101 µs, and T11 from 594 to 191 µs.
- **Single-pass K8V4 attention for verify widths 9–16** at batch 1, replacing the chunked
  two-pass path (`src/ops/softmax_attention/`).
- **Ada small-T tensor-core kernels for Q4/Q5 projections** at T = 2–16: attention input,
  SwiGLU, `linear_add`, and GDN value/z. The weights are read once per verify round
  (`src/ops/linear/ada_small_t_mma.cuh`).
- **Serving and benchmark tooling:**
  - a Prometheus `/metrics` endpoint (`llamacpp:prompt_tokens_total`,
    `llamacpp:tokens_predicted_total`, decode rounds, request gauges)
  - `ninfer_qwen3_5_mtp_round_bench` fixed for K ≤ 4
  - `tools/bench/cinf_matrix.py`, `cinf_k_bench.py` and `cinf_profile.py` (nsys)

## Context and VRAM

Measured with K8V4 KV and MTP enabled:

| KV capacity (tokens) | Free VRAM after reservation |
|---:|---:|
| 131,072 | 2.42 GiB |
| 163,840 | 1.59 GiB |
| **188,416 (production)** | **997 MiB** |
| 196,608 | 937–965 MiB |
| 229,376 / 262,144 | fails to start |

At 196,608 tokens without speculation (K0), 2.00 GiB is left free. Peak process memory is about
22.0 GB at K0 and 23.1 GB with MTP. The native model context is 262K. 188K is the practical
ceiling on 24 GB with about 1 GiB of headroom.

## Limitations

- **Fixed K:** fixed K7 is 6–11% slower than the previous engine's K3 on prose and long
  generation. Adaptive per-request K is not implemented; it is future work.
- **Context:** the ceiling is 188K on 24 GB, versus 262K native.
- **Long-context TTFT:** TTFT grows superlinearly, to 78 s at 131K and 127 s at 189K.
- **Concurrency:** 1.
- **Output divergence:** outputs diverge from K0 on open-ended text (see
  [Correctness](#correctness)).
- **Remaining cost per K10 round:**
  - The Q5 17,408-wide projection kernel reaches about 520 GB/s.
  - Drafting (the MTP layer plus the proposal head, about 9 ms) is bandwidth-bound. Reducing it
    requires a smaller draft head in the artifact, not kernel work.
- **Blackwell-only paths:** native NVFP4 weight kernels and TMA prompt kernels fail closed on
  Ada. `ninfer_softmax_attention_test` in full or `--nvfp4-only` mode aborts on those stubs.
- **Model artifact:** benchmarks used a locally built NInfer v3 artifact: Q4/Q5 Text body, Q8
  embedding and output head, 18,210,749,936 bytes, SHA-256 `5f67e9e4…c274d`. It is not
  redistributed here. The public
  [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) artifact has
  the same format family but was not benchmarked here.

## Build

Requirements: Linux, CUDA 13.x, CMake ≥ 3.28, Ninja, and an RTX 4090 or another sm_89 GPU.

```bash
cmake -S . -B build-sm89 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=ON
ninja -C build-sm89 ninfer-serve ninfer_softmax_attention_test ninfer_qwen3_5_mtp_round_bench
./build-sm89/tests/ninfer_softmax_attention_test --k8v4-only
```

Leave out `-DBUILD_TESTING=ON`, `-DNINFER_BUILD_BENCHMARKS=ON` and the extra targets to build
only the server. See also [build system](docs/maintainer/build-system.md) and
[PORT-RTX4090.md](PORT-RTX4090.md).

## Serve

The production flags:

```bash
./build-sm89/apps/ninfer-serve models/<artifact>.ninfer \
  --host 127.0.0.1 --port 8080 --model-id qwen/qwen3.8-27b \
  --max-context 188416 --kv-capacity 188416 --kv-dtype k8v4 \
  --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --spec mtp --draft-tokens 7 --lm-head-draft \
  --preserve-thinking --default-max-tokens 8192
```

The server exposes the OpenAI Chat Completions and Responses APIs, the Anthropic Messages API,
`/health` and `/metrics`. See [docs/serving.md](docs/serving.md) and [docs/cli.md](docs/cli.md).

### systemd deployment

The deployment files are in [`deploy/fnn/`](deploy/fnn/):

1. Copy `cinference-qwen38-4090.env.example` to `cinference-qwen38-4090.env`, which git
   ignores, and set absolute paths.
2. Adjust `WorkingDirectory` and `EnvironmentFile` in `cinference-qwen38-4090.service`.
3. Install it as a user unit:

```bash
cp deploy/fnn/cinference-qwen38-4090.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cinference-qwen38-4090
loginctl enable-linger "$USER"   # keep running without a login session
```

The unit restarts on failure after 5 s. In testing it recovered from `kill -9` in 17 s. Point
`CINF_BIN` at a copied release binary rather than the build tree, so rebuilds don't affect the
running service.

To roll back, the unit declares `Conflicts=` with the previous `ninfer-qwen38-4090.service`:

```bash
systemctl --user disable --now cinference-qwen38-4090
systemctl --user enable --now ninfer-qwen38-4090
```

## Attribution

- [NInfer](https://github.com/Neroued/ninfer) by Neroued and contributors: the engine, the
  `.ninfer` format, the model cards and most of the source.
- [Cinference](https://github.com/satellitedown/cinference) by satellitedown: MTP-10,
  capture-derived CUDA Graph reuse, and the RTX 5090 results in
  [`results/rtx5090-archive-recall.json`](results/rtx5090-archive-recall.json).
- [ninfer-3090](https://github.com/Don-Chad/ninfer-3090) by Don-Chad: the GitHub fork parent of
  this repository.
- RTX 4090 port and the changes listed above: Joshua Ramirez.

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Vendored dependencies keep their
own licenses. No model weights are included.
