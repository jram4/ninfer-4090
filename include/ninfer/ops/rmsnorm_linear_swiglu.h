#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: rmsnorm_linear_swiglu
 *
 * Math / indexing:
 *   For each token t, compute
 *     inv[t] = 1 / sqrt(mean_d(x[d,t]^2) + eps)
 *     h[d,t] = BF16(x[d,t] * inv[t] * (unit_offset ? 1 + norm_weight[d]
 *                                                        : norm_weight[d]))
 *   Then, for each output row i and token t, compute
 *     gate = sum_k decode(gate_up_weight[i,k]) * h[k,t]
 *     up = sum_k decode(gate_up_weight[17408+i,k]) * h[k,t]
 *     out[i,t] = BF16((gate / (1 + exp(-gate))) * up).
 *   The BF16 cast in h is semantic and must be preserved even when h is not materialized.
 *
 * Logical shapes / supported domain:
 *   x and norm_weight are contiguous BF16, with x [5120,T] and norm_weight [5120]. Output is
 *   contiguous BF16 [17408,T]. T is in [2,8]. gate_up_weight is the registered Q4_G64_FP16
 *   RowSplit weight [34816,5120], with FP16 scales for each 64-element K group. Gate rows occupy
 *   [0,17408) and the corresponding up rows occupy [17408,34816).
 *
 * Numeric:
 *   The independent oracle evaluates RMSNorm in FP64 from represented BF16 inputs and rounds h to
 *   BF16 before exact-decoding the Q4 weight and evaluating the LinearSwiGLU formula in FP64. The
 *   final BF16 output uses the LinearSwiGLU A16 numerical criterion. The fused implementation may
 *   choose its natural reduction and accumulator precision but must retain the BF16 boundary at h.
 *
 * Effects:
 *   Writes the full output. x, norm_weight, gate_up_weight and output must not alias; inputs and
 *   weights are read-only.
 *
 * Workspace:
 *   None. There is no persistent state side effect.
 */
void rmsnorm_linear_swiglu(const Tensor& x, const Tensor& norm_weight, float eps,
                           bool unit_offset, const Weight& gate_up_weight, Tensor& out,
                           cudaStream_t stream);

} // namespace ninfer::ops
