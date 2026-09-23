#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_ada_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct ResidualEpilogue {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void operator()(int row, int token, float value) const {
        __nv_bfloat16* slot = out + static_cast<std::int64_t>(token) * out_ld + row;
        *slot               = __float2bfloat16(value + __bfloat162float(*slot));
    }
};

template <int RowTiles, int Warps, int G, int Stages, int K>
void launch(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const std::int32_t rows = residual_out.ne[0];
    const ResidualEpilogue epilogue{
        static_cast<__nv_bfloat16*>(residual_out.data),
        static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16))};
    const auto* xp   = static_cast<const __nv_bfloat16*>(x.data);
    const auto x_ld  = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto* code = static_cast<const std::uint8_t*>(w.qdata);
    const auto* hi   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* sc   = static_cast<const std::uint8_t*>(w.scales);
    if (x.ne[1] <= 8) {
        using S = Q5AdaSmallTSchedule<RowTiles, Warps, 1, G, Stages>;
        if (rows % S::kRowsPerCta != 0) {
            throw std::invalid_argument("q5 linear_add ada small-T: rows must tile the CTA");
        }
        q5_ada_small_t_mma_launch<S, K>(xp, x_ld, code, hi, sc, rows, x.ne[1], epilogue, stream);
    } else {
        using S = Q5AdaSmallTSchedule<RowTiles, Warps, 2, G, Stages>;
        if (rows % S::kRowsPerCta != 0) {
            throw std::invalid_argument("q5 linear_add ada small-T: rows must tile the CTA");
        }
        q5_ada_small_t_mma_launch<S, K>(xp, x_ld, code, hi, sc, rows, x.ne[1], epilogue, stream);
    }
}

} // namespace

void q5_linear_add_ada_small_t_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > 16) {
        throw std::invalid_argument("q5 linear_add ada small-T: T must be in [1,16]");
    }
    if (w.k == 6144 && w.padded_shape[1] == 6144) {
        launch<5, 8, 8, 3, 6144>(x, w, residual_out, stream);
    } else if (w.k == 17408 && w.padded_shape[1] == 17408) {
        launch<5, 8, 16, 3, 17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add ada small-T: unsupported exact K");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
