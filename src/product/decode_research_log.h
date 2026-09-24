#pragma once

#include "ninfer/types.h"

#include <iosfwd>

namespace ninfer::product {

// Write one content-free decode research record. The caller owns stream lifetime and flushing.
void write_decode_research_event(std::ostream& output, const DecodeResearchEvent& event);

} // namespace ninfer::product
