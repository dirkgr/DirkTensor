#include "OlmoBlock.h"

#include <format>

OlmoBlock::OlmoBlock(const std::string& folder, const unsigned int index) :
    m_attention(folder, index),
    m_postAttentionNorm(std::format("{}/model.layers.{}.post_attention_layernorm.weight.npy", folder, index)),
    m_mlp(folder, index),
    m_postMlpNorm(std::format("{}/model.layers.{}.post_feedforward_layernorm.weight.npy", folder, index))
{
}

xt::xtensor<float, 3> OlmoBlock::forward(const xt::xtensor<float, 3>& input) {
    const auto after_attention = m_attention.forward(input);
    const auto normed_after_attention = m_postAttentionNorm.forward(after_attention);
    const auto h = input + normed_after_attention;

    const auto after_mlp = m_mlp.forward(h);
    const auto normed_after_mlp = m_postMlpNorm.forward(after_mlp);
    return h + normed_after_mlp;
}

xt::xtensor<float, 3> OlmoBlock::backward(const xt::xtensor<float, 3>& d_output) {
    auto grad = m_postMlpNorm.backward(d_output);
    grad = m_mlp.backward(grad);
    xt::xtensor<float, 3> d_h = grad + d_output;  // Must evaluate, not lazy expression
    grad = m_postAttentionNorm.backward(d_h);
    grad = m_attention.backward(grad);
    return grad + d_h;
}

void OlmoBlock::step(float lr) {
    m_attention.step(lr);
    m_postAttentionNorm.step(lr);
    m_mlp.step(lr);
    m_postMlpNorm.step(lr);
}

void OlmoBlock::zero_grad() {
    m_attention.zero_grad();
    m_postAttentionNorm.zero_grad();
    m_mlp.zero_grad();
    m_postMlpNorm.zero_grad();
}