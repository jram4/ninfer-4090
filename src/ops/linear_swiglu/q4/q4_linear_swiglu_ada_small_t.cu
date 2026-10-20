#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/ada_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGateUpRows   = 34816;
constexpr int kIntermediate = kGateUpRows / 2;
constexpr int kHidden       = 5120;

struct SwiGluEpilogue {
    static constexpr bool kPairedHalves = true;

    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void operator()(int row, int token, float gate, float up) const {
        out[static_cast<std::int64_t>(token) * out_ld + row] = __float2bfloat16_rn(silu(gate) * up);
    }
};

struct SplitKSwiGluEpilogue {
    static constexpr bool kPairedHalves = true;

    float* partial;
    std::int32_t output_rows;
    std::int32_t partial_ld;

    __device__ __forceinline__ void operator()(int row, int token, float gate, float up,
                                               int split) const {
        const std::int64_t base = static_cast<std::int64_t>(token) * partial_ld +
                                  split * 2 * output_rows + row;
        partial[base]                  = gate;
        partial[base + output_rows]    = up;
    }
};

__global__ void q4_linear_swiglu_split_k_reduce_kernel(
    const float* __restrict__ partial, __nv_bfloat16* __restrict__ out, std::int32_t output_rows,
    std::int32_t output_ld, std::int32_t partial_ld, std::int32_t tokens) {
    const std::int32_t index = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::int32_t count = output_rows * tokens;
    if (index >= count) { return; }

    const std::int32_t token = index / output_rows;
    const std::int32_t row   = index - token * output_rows;
    const std::int64_t base  = static_cast<std::int64_t>(token) * partial_ld + row;
    const float gate = partial[base] + partial[base + 2 * output_rows];
    const float up   = partial[base + output_rows] + partial[base + 3 * output_rows];
    out[static_cast<std::int64_t>(token) * output_ld + row] = __float2bfloat16_rn(silu(gate) * up);
}

template <int RowTiles, int Warps, int G, int Stages>
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const SwiGluEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data),
                                  static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16))};
    const auto* xp   = static_cast<const __nv_bfloat16*>(x.data);
    const auto x_ld  = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto* code = static_cast<const std::uint8_t*>(w.qdata);
    const auto* sc   = static_cast<const std::uint8_t*>(w.scales);
    using Rows       = AdaPairedHalfRows<kIntermediate>;
    if (x.ne[1] <= 8) {
        using S = Q4AdaSmallTSchedule<RowTiles, Warps, 1, G, Stages>;
        static_assert(kGateUpRows % S::kRowsPerCta == 0);
        ada_small_t_mma_launch<S, kHidden, Rows>(xp, x_ld, code, nullptr, sc, kGateUpRows, x.ne[1],
                                                 epilogue, stream);
    } else {
        using S = Q4AdaSmallTSchedule<RowTiles, Warps, 2, G, Stages>;
        static_assert(kGateUpRows % S::kRowsPerCta == 0);
        ada_small_t_mma_launch<S, kHidden, Rows>(xp, x_ld, code, nullptr, sc, kGateUpRows, x.ne[1],
                                                 epilogue, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_swiglu_ada_small_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    if (w.n != kGateUpRows || w.k != kHidden || w.padded_shape[1] != kHidden) {
        throw std::invalid_argument("q4 linear_swiglu ada small-T requires weight [34816,5120]");
    }
    if (x.ne[1] < 1 || x.ne[1] > 16) {
        throw std::invalid_argument("q4 linear_swiglu ada small-T requires T in [1,16]");
    }
    launch<4, 8, 8, 3>(x, w, out, stream);
}

void q4_linear_swiglu_split_k_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     Tensor& partial, cudaStream_t stream) {
    if (w.n != kGateUpRows || w.k != kHidden || w.padded_shape[1] != kHidden ||
        x.ne[0] != kHidden || x.ne[1] != 8 || out.ne[0] != kIntermediate || out.ne[1] != 8) {
        throw std::invalid_argument("q4 linear_swiglu split-K requires [34816,5120] at T=8");
    }
    if (partial.dtype != DType::FP32 || partial.ne[0] != 4 * kIntermediate || partial.ne[1] != 8) {
        throw std::invalid_argument("q4 linear_swiglu split-K requires FP32 partials [4*17408,8]");
    }

    using S = Q4AdaSmallTSchedule<4, 8, 1, 8, 3, 2>;
    using Rows = AdaPairedHalfRows<kIntermediate>;
    static_assert(kGateUpRows % S::kRowsPerCta == 0);

    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    const auto x_ld = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto* code = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* partial_data = static_cast<float*>(partial.data);
    const auto partial_ld = static_cast<std::int32_t>(partial.nb[1] / sizeof(float));
    const SplitKSwiGluEpilogue epilogue{partial_data, kIntermediate, partial_ld};
    ada_small_t_mma_launch<S, kHidden, Rows>(xp, x_ld, code, nullptr, scales, kGateUpRows,
                                              x.ne[1], epilogue, stream);
    CUDA_CHECK(cudaGetLastError());

    auto* out_data = static_cast<__nv_bfloat16*>(out.data);
    const auto output_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const auto partial_ld = static_cast<std::int32_t>(partial.nb[1] / sizeof(float));
    const int elements = kIntermediate * 8;
    constexpr int kThreads = 256;
    q4_linear_swiglu_split_k_reduce_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0,
                                              stream>>>(
        static_cast<const float*>(partial.data), out_data, kIntermediate, output_ld, partial_ld, 8);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
