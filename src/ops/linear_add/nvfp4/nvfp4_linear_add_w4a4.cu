#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using M32N64            = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

// This projection selects its own route, so the layout the quantizer writes below must be derived
// from the same predicate; the two are read together at the call site for that reason.
constexpr bool w4a4_tma_route(std::int32_t tokens) { return tokens >= 1024; }

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& residual, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    auto* output      = static_cast<__nv_bfloat16*>(residual.data);
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Nvfp4W4a4Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Geometry, M32N128>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<Geometry, M128N128Pipelined>(weight, residual, workspace, tokens, stream);
    } else {
        launch_gemm<Geometry, M128N128Resident>(weight, residual, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    (void)x; (void)weight; (void)residual; (void)workspace; (void)stream;
    throw std::invalid_argument(
        "Cinference-4090: native NVFP4 linear-add weights require Blackwell");
}

} // namespace ninfer::ops::detail
