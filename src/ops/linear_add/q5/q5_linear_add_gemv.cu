#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <bool kCandidate>
void launch_residual(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out          = static_cast<__nv_bfloat16*>(residual_out.data);

    if (w.n == 5120 && w.k == 6144 && w.padded_shape[1] == 6144) {
        q5_rowsplit_gemv_residual_launch_kernel<5120, 6144, 8, 2, !kCandidate>(
            xp, codes, high, scales, out, stream);
    } else if (w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408) {
        q5_rowsplit_gemv_residual_launch_kernel<5120, 17408, 8, 2, false>(
            xp, codes, high, scales, out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add GEMV: unsupported exact shape");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q5_linear_add_gemv_residual_control_launch(const Tensor& x, const Weight& w,
                                                Tensor& residual_out, cudaStream_t stream) {
    launch_residual<false>(x, w, residual_out, stream);
}

void q5_linear_add_gemv_residual_candidate_launch(const Tensor& x, const Weight& w,
                                                  Tensor& residual_out, cudaStream_t stream) {
    launch_residual<true>(x, w, residual_out, stream);
}

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    // On sm_89 the 12 KiB activation for [5120,6144] remains cache-resident. Reading
    // it directly removes the replicated shared-memory allocation per CTA while
    // preserving weight decode, FMA, warp reduction, residual add, and BF16 cast order.
    q5_linear_add_gemv_residual_candidate_launch(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
