#include "product/decode_research_log.h"
#include "runtime/contract/decode_research.h"
#include "runtime/engine/decode_research.h"

#include <iostream>
#include <sstream>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::runtime;
    int failures = 0;

    const EngineOptions defaults;
    failures += check(!defaults.decode_research_observer.callback,
                      "decode research observation is not disabled by default");

    const DecodeRoundCommitMetadata partial =
        make_decode_round_commit_metadata(4096, 3, 2, 4, 3);
    failures += check(partial.available && partial.context_frontier == 4096 &&
                          partial.draft_extent == 3 && partial.accepted_prefix == 2 &&
                          partial.licensed_count == 4 && partial.committed_count == 3,
                      "Program commit metadata did not preserve the speculative row facts");

    const DecodeResearchEvent partial_event = make_decode_research_event(DecodeResearchEventInput{
        .engine_session_id   = 101,
        .engine_round_id     = 7,
        .request_id          = 4,
        .request_round_index = 3,
        .batch_size          = 1,
        .backend             = SpeculativeBackend::Mtp,
        .draft_window        = 3,
        .use_cuda_graph      = true,
        .context_frontier    = partial.context_frontier,
        .draft_extent        = partial.draft_extent,
        .accepted_prefix     = partial.accepted_prefix,
        .licensed_count      = partial.licensed_count,
        .committed_count     = partial.committed_count,
        .round_elapsed_ns    = 1250000,
    });
    failures += check(partial_event.committed_draft_prefix == 2 && partial_event.truncated &&
                          !partial_event.terminal && !partial_event.cancelled,
                      "short Engine commitment did not bound the committed draft prefix");

    const DecodeResearchEvent rejected = make_decode_research_event(DecodeResearchEventInput{
        .backend          = SpeculativeBackend::Mtp,
        .draft_window     = 5,
        .draft_extent     = 5,
        .accepted_prefix  = 0,
        .licensed_count   = 1,
        .committed_count  = 1,
        .finish_reason    = FinishReason::None,
    });
    failures += check(rejected.accepted_prefix == 0 && rejected.committed_draft_prefix == 0 &&
                          !rejected.truncated,
                      "fully rejected draft extent was not distinguished from its correction token");

    const DecodeResearchEvent short_extent = make_decode_research_event(DecodeResearchEventInput{
        .backend             = SpeculativeBackend::DFlash2,
        .draft_window        = 7,
        .draft_extent        = 2,
        .accepted_prefix     = 2,
        .licensed_count      = 3,
        .committed_count     = 3,
        .terminal            = true,
        .finish_reason       = FinishReason::StopToken,
    });
    failures += check(short_extent.draft_extent == 2 && short_extent.accepted_prefix == 2 &&
                          short_extent.committed_draft_prefix == 2 && short_extent.terminal &&
                          short_extent.finish_reason == FinishReason::StopToken &&
                          !short_extent.truncated,
                      "short draft extent or terminal stop state was reported incorrectly");

    const DecodeResearchEvent cancelled = make_decode_research_event(DecodeResearchEventInput{
        .backend          = SpeculativeBackend::Mtp,
        .draft_window     = 3,
        .draft_extent     = 3,
        .accepted_prefix  = 2,
        .licensed_count   = 3,
        .committed_count  = 0,
        .terminal         = true,
        .cancelled        = true,
        .finish_reason    = FinishReason::Cancelled,
    });
    failures += check(cancelled.committed_draft_prefix == 0 && !cancelled.truncated &&
                          cancelled.cancelled && cancelled.terminal,
                      "cancellation was incorrectly counted as output truncation");

    const DecodeResearchEvent ordinary = make_decode_research_event(DecodeResearchEventInput{
        .backend          = SpeculativeBackend::None,
        .licensed_count   = 1,
        .committed_count  = 1,
        .finish_reason    = FinishReason::None,
    });
    std::ostringstream jsonl;
    product::write_decode_research_event(jsonl, ordinary);
    const std::string line = jsonl.str();
    failures += check(line.find("\"schema\":\"ninfer.decode_research_round.v1\"") !=
                          std::string::npos &&
                          line.find("\"draft_extent\":0") != std::string::npos &&
                          line.find("\"committed_count\":1") != std::string::npos &&
                          line.find("prompt") == std::string::npos &&
                          line.find("token_ids") == std::string::npos &&
                          line.find("\"output\"") == std::string::npos,
                      "research serialization is missing fields or contains content payloads");
    return failures == 0 ? 0 : 1;
}
