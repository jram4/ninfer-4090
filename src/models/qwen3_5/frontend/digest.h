#pragma once

#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

using Sha256Digest = std::array<std::uint8_t, 32>;

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input);
[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input,
                                  const std::function<void()>& checkpoint);
[[nodiscard]] Sha256Digest sha256(std::string_view input);
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);
// Hash token IDs as their signed int32 bit patterns serialized in little-endian order.
[[nodiscard]] std::string token_ids_sha256_hex(std::span<const TokenId> token_ids);

} // namespace ninfer::models::qwen3_5::frontend
