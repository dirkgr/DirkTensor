#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include "OlmoAttention.h"
#include "OlmoMlp.h"
#include "RMSNorm.h"

class OlmoBlock {
public:
    OlmoBlock(const std::string& folder, unsigned int index);

    xt::xtensor<float, 1> forward(const auto& input) {
        const auto after_attention = m_attention.forward(input);
        const auto normed_after_attention = m_postAttentionNorm.forward(after_attention);
        const auto h = input + normed_after_attention;

        const auto after_mlp = m_mlp.forward(h);
        const auto normed_after_mlp = m_postMlpNorm.forward(after_mlp);
        return h + normed_after_mlp;
    }

private:
    OlmoAttention m_attention;
    RMSNorm m_postAttentionNorm;
    OlmoMlp m_mlp;
    RMSNorm m_postMlpNorm;
};
