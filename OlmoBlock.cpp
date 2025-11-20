#include "OlmoBlock.h"

#include <chrono>
#include <format>
#include <iostream>

#include <xtensor/io/xio.hpp>


OlmoBlock::OlmoBlock(const std::string& folder, const unsigned int index) :
    m_attention(folder, index),
    m_postAttentionNorm(std::format("{}/model.layers.{}.post_attention_layernorm.weight.npy", folder, index)),
    m_mlp(folder, index),
    m_postMlpNorm(std::format("{}/model.layers.{}.post_feedforward_layernorm.weight.npy", folder, index)),
    block_index(index)
{
    // nothing to do
}

xt::xtensor<float, 3> OlmoBlock::forward(const xt::xtensor<float, 3>& input) {
    using Clock = std::chrono::high_resolution_clock;

    // Attention
    auto start = Clock::now();
    const auto after_attention = m_attention.forward(input);
    attention_time_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    // First normalization
    start = Clock::now();
    const auto normed_after_attention = m_postAttentionNorm.forward(after_attention);
    const auto h = input + normed_after_attention;
    norm1_time_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    // MLP
    start = Clock::now();
    const auto after_mlp = m_mlp.forward(h);
    mlp_time_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    // Second normalization
    start = Clock::now();
    const auto normed_after_mlp = m_postMlpNorm.forward(after_mlp);
    const auto result = h + normed_after_mlp;
    norm2_time_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    return result;
}
