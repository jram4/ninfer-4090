#include "ninfer/ops/rmsnorm_linear_swiglu.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool overlaps(const void* first, std::size_t first_bytes, const void* second,
              std::size_t second_bytes) {
    if (first == nullptr || second == nullptr || first_bytes == 0 || second_bytes == 0) {
        return false;
    }
    const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
    const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
    if (first_bytes > std::numeric_limits<std::uintptr_t>::max() - first_begin ||
        second_bytes > std::numeric_limits<std::uintptr_t>::max() - second_begin) {
        return true;
    }
    const auto first_end = first_begin + first_bytes;
    const auto second_end = second_begin + second_bytes;
    return first_begin < second_end && second_begin < first_end;
}

bool valid_q4_gate_up(const Weight& weight) {
    return weight.qtype == QType::Q4_G64_FP16 && weight.layout == QuantLayout::RowSplit &&
           weight.group_size == 64 && weight.group == 64 && weight.scale_dtype == DType::FP16 &&
           weight.ndim == 2 && weight.shape[0] == 34816 && weight.shape[1] == 5120 &&
           weight.padded_shape[0] == 34816 && weight.padded_shape[1] == 5120 &&
           weight.qdata != nullptr && weight.scales != nullptr && weight.qhigh == nullptr &&
           weight.high_plane_bytes == 0;
}

} // namespace

void rmsnorm_linear_swiglu(const Tensor& x, const Tensor& norm_weight, float eps, bool unit_offset,
                           const Weight& gate_up_weight, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || norm_weight.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("rmsnorm_linear_swiglu: x/norm_weight/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm_linear_swiglu: eps must be positive and finite");
    }
    const std::int32_t tokens = x.ne[1];
    if (x.ne[0] != 5120 || tokens < 2 || tokens > 8 || x.ne[2] != 1 || x.ne[3] != 1 ||
        norm_weight.ne[0] != 5120 || norm_weight.ne[1] != 1 || norm_weight.ne[2] != 1 ||
        norm_weight.ne[3] != 1 || out.ne[0] != 17408 || out.ne[1] != tokens || out.ne[2] != 1 ||
        out.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm_linear_swiglu: unsupported tensor shape");
    }
    if (!x.is_contiguous() || !norm_weight.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("rmsnorm_linear_swiglu: tensors must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(norm_weight.data, alignof(__nv_bfloat162)) ||
        !aligned_to(out.data, 16) || x.data == out.data || norm_weight.data == out.data) {
        throw std::invalid_argument("rmsnorm_linear_swiglu: invalid tensor pointer or alignment");
    }
    if (!valid_q4_gate_up(gate_up_weight) || !aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, 4)) {
        throw std::invalid_argument("rmsnorm_linear_swiglu: unsupported Q4 gate/up weight");
    }
    const std::size_t output_bytes = static_cast<std::size_t>(out.bytes());
    const std::size_t packed_bytes =
        static_cast<std::size_t>(gate_up_weight.n) * gate_up_weight.k / 2;
    const std::size_t scale_bytes =
        static_cast<std::size_t>(gate_up_weight.n) * gate_up_weight.k / 64 * sizeof(std::uint16_t);
    if (overlaps(out.data, output_bytes, x.data, static_cast<std::size_t>(x.bytes())) ||
        overlaps(out.data, output_bytes, norm_weight.data,
                 static_cast<std::size_t>(norm_weight.bytes())) ||
        overlaps(out.data, output_bytes, gate_up_weight.qdata, packed_bytes) ||
        overlaps(out.data, output_bytes, gate_up_weight.scales, scale_bytes)) {
        throw std::invalid_argument("rmsnorm_linear_swiglu: output must not overlap an input or weight");
    }

    detail::q4_rmsnorm_linear_swiglu_small_t_launch(x, norm_weight, eps, unit_offset,
                                                    gate_up_weight, out, stream);
}

} // namespace ninfer::ops
