#pragma once

#include <cstdint>

namespace ninfer::runtime {

struct DecodeRoundCommitMetadata {
    bool available                 = false;
    std::uint32_t context_frontier = 0;
    std::uint32_t draft_extent     = 0;
    std::uint32_t accepted_prefix  = 0;
    std::uint32_t licensed_count   = 0;
    std::uint32_t committed_count  = 0;
};

[[nodiscard]] constexpr DecodeRoundCommitMetadata make_decode_round_commit_metadata(
    std::uint32_t context_frontier, std::uint32_t draft_extent,
    std::uint32_t accepted_prefix, std::uint32_t licensed_count,
    std::uint32_t committed_count) noexcept {
    return DecodeRoundCommitMetadata{
        .available        = true,
        .context_frontier = context_frontier,
        .draft_extent     = draft_extent,
        .accepted_prefix  = accepted_prefix,
        .licensed_count   = licensed_count,
        .committed_count  = committed_count,
    };
}

} // namespace ninfer::runtime
