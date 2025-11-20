#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include "OlmoAttention.h"
#include "OlmoMlp.h"
#include "RMSNorm.h"

class OlmoBlock {
public:
    OlmoBlock(const std::string& folder, unsigned int index);

    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);

    // Profiling data (public for access)
    double attention_time_ms = 0.0;
    double norm1_time_ms = 0.0;
    double mlp_time_ms = 0.0;
    double norm2_time_ms = 0.0;
    unsigned int block_index = 0;

private:
    OlmoAttention m_attention;
    RMSNorm m_postAttentionNorm;
    OlmoMlp m_mlp;
    RMSNorm m_postMlpNorm;
};
