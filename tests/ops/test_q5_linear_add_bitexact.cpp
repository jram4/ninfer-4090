#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/row_split_pack.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint16_t> copy_bf16_bits(const DBuf& buffer, std::size_t words) {
    std::vector<std::uint16_t> bits(words);
    CUDA_CHECK(cudaMemcpy(bits.data(), buffer.p, words * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    return bits;
}

int compare_bits(const std::string& label, const DBuf& control, const DBuf& candidate,
                 std::size_t words) {
    const std::vector<std::uint16_t> expected = copy_bf16_bits(control, words);
    const std::vector<std::uint16_t> actual   = copy_bf16_bits(candidate, words);
    const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (mismatch.first == expected.end()) { return 0; }
    const std::size_t index = static_cast<std::size_t>(mismatch.first - expected.begin());
    std::cerr << label << ": BF16 bit mismatch at " << index << " control=0x" << std::hex
              << *mismatch.first << " candidate=0x" << *mismatch.second << std::dec << '\n';
    return 1;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float magnitude) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -magnitude, magnitude);
    round_to_bf16(values);
    return values;
}

int q5_linear_add_case(std::int32_t k, std::uint32_t seed) {
    constexpr std::int32_t n = 5120;
    row_split::PackedWeight packed =
        row_split::make_patterned_weight(QType::Q5G64_F16S, n, k, seed + 1000u);
    DBuf device_weight(packed.payload.size());
    CUDA_CHECK(cudaMemcpy(device_weight.p, packed.payload.data(), packed.payload.size(),
                          cudaMemcpyHostToDevice));
    const Weight weight = packed.device_weight(device_weight.p);

    const std::vector<float> x = make_bf16_values(static_cast<std::size_t>(k), seed, 8.0f);
    const std::vector<float> residual =
        make_bf16_values(static_cast<std::size_t>(n), seed + 2000u, 4.0f);
    DBuf device_x = to_device_bf16(x);
    DBuf control  = to_device_bf16(residual);
    DBuf candidate = to_device_bf16(residual);
    Tensor tx(device_x.p, DType::BF16, {k, 1});
    Tensor control_out(control.p, DType::BF16, {n, 1});
    Tensor candidate_out(candidate.p, DType::BF16, {n, 1});

    ops::detail::q5_linear_add_gemv_residual_control_launch(tx, weight, control_out, nullptr);
    ops::detail::q5_linear_add_gemv_residual_candidate_launch(tx, weight, candidate_out, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return compare_bits("Q5 LinearAdd [5120," + std::to_string(k) + "]", control, candidate,
                        static_cast<std::size_t>(n));
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    const int failures = q5_linear_add_case(6144, 17u);
    std::cout << (failures ? "FAIL" : "OK") << " bit-exact Q5 LinearAdd candidate\n";
    return failures ? 1 : 0;
}
