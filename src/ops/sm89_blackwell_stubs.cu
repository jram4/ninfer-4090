// RTX 4090 / Ada compatibility stubs for Blackwell-only NVFP4 launch routes.
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument(
        "Cinference-4090: native NVFP4 weight/TMA execution requires Blackwell; "
        "use a groupwise Q4/Q5/Q6/Q8/BF16 .ninfer artifact on RTX 4090");
}
} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4GeometryId, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    unsupported();
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                                     std::int32_t, float, cudaStream_t) {
    unsupported();
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                               __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    unsupported();
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4GeometryId, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, __nv_bfloat16*, std::int32_t,
                                      float, cudaStream_t) {
    unsupported();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float,
                                         cudaStream_t) {
    unsupported();
}

void causal_attention_prompt_nvfp4_kernel_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&, Tensor&,
                                                 cudaStream_t) {
    unsupported();
}

void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor&, const Tensor&,
                                                       const Tensor&, const Tensor&, float,
                                                       const PagedKVBatchLayerView&, Tensor&,
                                                       cudaStream_t) {
    unsupported();
}

} // namespace ninfer::ops::detail
