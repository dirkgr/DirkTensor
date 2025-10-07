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
