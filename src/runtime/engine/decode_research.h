#pragma once

#include "ninfer/types.h"
#include "runtime/contract/decode_research.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::runtime {

struct DecodeResearchEventInput {
    std::uint64_t engine_session_id   = 0;
    std::uint64_t engine_round_id     = 0;
    std::uint64_t request_id          = 0;
    std::uint64_t request_round_index = 0;
    std::uint32_t batch_size          = 0;
    SpeculativeBackend backend        = SpeculativeBackend::None;
    std::uint32_t draft_window        = 0;
    bool use_cuda_graph               = true;
    std::uint32_t context_frontier    = 0;
    std::uint32_t draft_extent        = 0;
    std::uint32_t accepted_prefix     = 0;
    std::uint32_t licensed_count      = 0;
    std::uint32_t committed_count     = 0;
    std::uint64_t round_elapsed_ns    = 0;
    bool terminal                    = false;
    bool cancelled                   = false;
    FinishReason finish_reason       = FinishReason::None;
};

[[nodiscard]] inline DecodeResearchEvent
make_decode_research_event(const DecodeResearchEventInput& input) noexcept {
    return DecodeResearchEvent{
        .engine_session_id    = input.engine_session_id,
        .engine_round_id      = input.engine_round_id,
        .request_id           = input.request_id,
        .request_round_index  = input.request_round_index,
        .batch_size           = input.batch_size,
        .backend              = input.backend,
        .draft_window         = input.draft_window,
        .use_cuda_graph       = input.use_cuda_graph,
        .context_frontier     = input.context_frontier,
        .draft_extent         = input.draft_extent,
        .accepted_prefix      = input.accepted_prefix,
        .committed_draft_prefix = std::min(input.accepted_prefix, input.committed_count),
        .licensed_count       = input.licensed_count,
        .committed_count      = input.committed_count,
        .round_elapsed_ns     = input.round_elapsed_ns,
        .terminal             = input.terminal,
        .truncated            = !input.cancelled &&
                                input.committed_count < input.licensed_count,
        .cancelled            = input.cancelled,
        .finish_reason        = input.finish_reason,
    };
}

} // namespace ninfer::runtime
