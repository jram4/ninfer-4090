#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 3> kTokenCases{2, 7, 8};
        constexpr std::array<std::int32_t, 2> kGraphCases{7, 8};
        const int failures = run_profile(
            "RMSNorm+LinearSwiGLU Q4_A16",
            {QType::Q4_G64_FP16, 34816, 5120, 17408, 2408U, ActivationCompute::A16},
            kTokenCases, kGraphCases, true);
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " RMSNorm+LinearSwiGLU Q4_A16 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "RMSNorm+LinearSwiGLU Q4_A16 test failed: " << error.what() << '\n';
        return 1;
    }
}
