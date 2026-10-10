#pragma once
#include <cmath>
#include <string>

namespace strata::spec {
// NAVER DLoop, arXiv:2610.07659: gate the latest complete block, then verify
// all accumulated drafts together. Stock MTP weights; no loop-aware training.
struct DLoopConfig {
    bool enabled = false;
    int block_size = 3;
    int max_loops = 2;
    double gate = -0.5;
    int drafts() const { return block_size * max_loops; }
    std::string error() const {
        if (block_size < 1 || block_size > 7 || max_loops < 1 || max_loops > 7 / block_size)
            return "DLoop needs block_size * max_loops <= 7, with both positive";
        if (!std::isfinite(gate) || gate > 0)
            return "DLoop gate must be finite and <= 0 (natural log probability)";
        return {};
    }
    bool extend(const float* probabilities, int count) const {
        double score = 0;
        for (int i = 0; i < count; ++i) {
            const double q = probabilities[i];
            if (!(q > 0 && q <= 1)) return false;
            score += std::log(q);
        }
        return count == block_size && score >= gate;
    }
};
} // namespace strata::spec
