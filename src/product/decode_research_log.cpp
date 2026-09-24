#include "product/decode_research_log.h"

#include <ostream>

namespace ninfer::product {
namespace {

const char* backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

const char* finish_reason_name(FinishReason reason) noexcept {
    switch (reason) {
    case FinishReason::None:
        return "none";
    case FinishReason::OutputLimit:
        return "output_limit";
    case FinishReason::ContextCapacity:
        return "context_capacity";
    case FinishReason::StopToken:
        return "stop_token";
    case FinishReason::StopString:
        return "stop_string";
    case FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

} // namespace

void write_decode_research_event(std::ostream& output, const DecodeResearchEvent& event) {
    output << "{\"schema\":\"ninfer.decode_research_round.v1\""
           << ",\"engine_session_id\":" << event.engine_session_id
           << ",\"engine_round_id\":" << event.engine_round_id
           << ",\"request_id\":" << event.request_id
           << ",\"request_round_index\":" << event.request_round_index
           << ",\"batch_size\":" << event.batch_size
           << ",\"backend\":\"" << backend_name(event.backend) << '"'
           << ",\"draft_window\":" << event.draft_window
           << ",\"cuda_graph\":" << (event.use_cuda_graph ? "true" : "false")
           << ",\"context_frontier\":" << event.context_frontier
           << ",\"draft_extent\":" << event.draft_extent
           << ",\"accepted_prefix\":" << event.accepted_prefix
           << ",\"committed_draft_prefix\":" << event.committed_draft_prefix
           << ",\"licensed_count\":" << event.licensed_count
           << ",\"committed_count\":" << event.committed_count
           << ",\"round_elapsed_ns\":" << event.round_elapsed_ns
           << ",\"terminal\":" << (event.terminal ? "true" : "false")
           << ",\"truncated\":" << (event.truncated ? "true" : "false")
           << ",\"cancelled\":" << (event.cancelled ? "true" : "false")
           << ",\"finish_reason\":\"" << finish_reason_name(event.finish_reason)
           << "\"}\n";
}

} // namespace ninfer::product
