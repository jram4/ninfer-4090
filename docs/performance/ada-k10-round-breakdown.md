# Ada K10 round breakdown (RTX 4090, Qwen3.8-27B groupwise v3)

Measured 2026-09-23 on FNN with `build-sm89-perf` (branch `ada-k10-perf`, base `76b3975`).

- Serve profile: `tools/bench/cinf_profile.py`, one recall request (25,168 prompt tokens, thinking off, greedy,
  128 output tokens), `--kv-dtype k8v4`, prefill chunk 1024, prefix reuse off, CUDA graphs on,
  `nsys --cuda-graph-trace=node`. Times are GPU kernel time per decode round inside the request's round window.
- Op sweeps: the repository op benchmarks with `--execution graph` and a cold L2.

## Per-round cost

| K | Verify width | Wall ms/round | GPU kernel ms/round |
|---|---|---|---|
| 0 | 1 | 22.34 | 21.94 |
| 5 | 6 | 33.54 | 33.02 |
| 10 | 11 | 54.29 | 53.63 |

The graph replay leaves under 1 ms of host gap per round. The round is GPU-bound.

## Kernel time per round (ms)

| Kernel family | K0 | K5 | K10 | Role |
|---|---|---|---|---|
| Q5 linear_add (`q5_rowsplit_gemm_simt_split2`) | 5.64 | 7.25 | 12.50 | down, GDN-out and attention-out projections (128 calls) |
| Q4 SwiGLU (`q4_linear_swiglu_gemv_pair` at T=1, `q4_ksplit_mma` otherwise) | 6.92 | 7.50 | 11.05* | gate/up, 64 calls (*K10 also includes the Q4 part of the input projections) |
| Q5 input projections (`q5_rowsplit_gemv` / `_split4` / column-tiled `_simt`) | 2.90 | 3.43 | 7.12 | GDN value/z and attention gate/value, 64 calls |
| Q4 input projections (`q4_rowsplit_gemv` / `_simt`) | 1.15 | 2.42 | (in ksplit) | GDN q/k and attention q/k |
| Attention (`causal_attention_small_t_k8v4_tiled`) | 3.18 (16) | 4.96 (21) | 10.31 (43) | 16 target layers plus one MTP call per draft step |
| Q8 (`q8_ksplit_mma`) | 1.42 (1) | 3.99 (26) | 6.56 (51) | output head (1 call) plus MTP layer (5 per draft step) |
| Draft head (`q4_rowsplit_gemv`, 131072x5120) | - | 1.88 | 3.77 | one per draft step |
| GDN recurrent record/fold | 0.23 | 0.83 | 1.06 | |

At K10, drafting (10 steps) costs about 0.52 ms of Q8 MTP layer, 0.38 ms of draft head and about 0.2 ms of
MTP attention per step, roughly 11 ms in total. Target verification is about 42 ms against 21 ms at T=1.

## Op width scaling (cold L2, graph)

| Op | T=1 | T=6 | T=8 | T=11 | T=12 | T=13 | T=16 |
|---|---|---|---|---|---|---|---|
| Q5 linear_add 5120x17408 (us) | 92 (635 GB/s) | 114 | 131 | 178 (332 GB/s) | 183 | 185 | 267 (MMA) |
| Q5 linear_add 5120x6144 (us) | 37 | 49 | 56 | 96 | 74 | 82 | 97 (MMA) |
| Q4 SwiGLU 34816x5120 (us) | 145 (652 GB/s) | 148 | 150 | 183 (519 GB/s) | 186 | - | 202 |
| GDN input Q4/Q5 (us) | 87 | 111 | 106 | 161 | 167 | 233 (MMA) | 236 (MMA) |
| Attention input Q4/Q5 (us) | 74 | 118 | 91 | 121 | 128 | 201 (MMA) | 201 (MMA) |
| Reference: Q4 GEMV 131072x5120 T=1 | 437 us, 816 GB/s | | | | | | |
| Reference: Q8 5120x17408 T=1 | 156 us, 609 GB/s | | | | | | |

## Findings

1. Every target projection runs at about 600-650 GB/s at T=1, below the 816 GB/s the Q4 draft head reaches on the
   same card. The narrow-width kernels issue too little memory traffic in flight for Ada.
2. The Q5 SIMT split2 kernel becomes FMA/unpack-bound from T=8 (11 TFLOP/s at T=11), so its cost grows with width.
3. The input-projection small-T kernels are column-tiled (8 columns), so T=11 reads the Q5 weights twice.
4. The existing Q5, GDN-input and attention-input MMA routes (T>=13/14/17) are slower than the SIMT routes on Ada
   (about 200-250 GB/s). Rerouting widths 9-12 to them would make things worse.
5. The K8V4 small-T attention kernel is limited to 8 query tokens per pass (`TokenTile * GroupSize <= 48`, group size
   6), so width 11 reads the 25K-token KV cache twice per full-attention layer.
6. The Q4 path named in the plan (`launch_q4_simt<SimtR4C4>`) is not used by this model. The four route families in the
   plan are replaced by the Q5 linear_add, Q5 input and attention items above.

## Phase 2 route list

1. Q5 linear_add, widths 1-16: a single-pass tensor-core kernel that keeps weight reads near bandwidth for every width.
2. Q5 input projections (GDN value/z, attention gate/value), widths 2-16: the same single-pass structure.
3. K8V4 small-T attention: one KV pass for widths 9-12.
4. Q4 SwiGLU widths 9-16: width scaling from T=8 to T=11 (150 to 183 us).

Bandwidth floors at 850 GB/s: Q5 5120x17408 about 69 us, Q4 SwiGLU about 118 us.
