#include "OlmoBlock.h"

#include <format>

OlmoBlock::OlmoBlock(const std::string& folder, const unsigned int index) :
    m_attention(folder, index),
    m_postAttentionNorm(std::format("{}/model.layers.{}.post_attention_layernorm.weight.npy", folder, index)),
    m_mlp(folder, index),
    m_postMlpNorm(std::format("{}/model.layers.{}.post_feedforward_layernorm.weight.npy", folder, index))
{
    // nothing to do
}

xt::xtensor<float, 3> OlmoBlock::forward(const xt::xtensor<float, 3>& input) {
    const auto after_attention = m_attention.forward(input);
    const auto normed_after_attention = m_postAttentionNorm.forward(after_attention);
    const auto h = input + normed_after_attention;

    const auto after_mlp = m_mlp.forward(h);
    const auto normed_after_mlp = m_postMlpNorm.forward(after_mlp);
    return h + normed_after_mlp;
}
