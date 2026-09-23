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

} // namespace ninfer::ops::detail
