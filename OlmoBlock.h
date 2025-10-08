#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include "OlmoAttention.h"
#include "OlmoMlp.h"
#include "RMSNorm.h"

class OlmoBlock {
public:
    OlmoBlock(const std::string& folder, unsigned int index);

    xt::xtensor<float, 1> forward(const xt::xtensor<float, 1>& input);

private:
    OlmoAttention m_attention;
    RMSNorm m_postAttentionNorm;
    OlmoMlp m_mlp;
    RMSNorm m_postMlpNorm;
};
