#include "OlmoModel.h"

#include <cassert>
#include <xtensor/io/xnpy.hpp>

OlmoModel::OlmoModel(const std::string& folder) : m_norm(folder + "/model.norm.weight.npy") {
    m_embeddings = xt::load_npy<float>(folder + "/model.embed_tokens.weight.npy");
    assert(m_embeddings.shape().size() == 2);
    assert(m_embeddings.shape()[1] == d_model);

    m_lmHead = xt::transpose(xt::load_npy<float>(folder + "/lm_head.weight.npy"));

    for(size_t i = 0; i < n_layers; i++)
        m_blocks[i] = std::make_unique<OlmoBlock>(folder, i);
}

xt::xtensor<float, 1> OlmoModel::forward(const uint32_t token) {
    // Embedding
    auto x = xt::view(m_embeddings, token, xt::all());

    // Blocks
    for(size_t i = 0; i < n_layers; i++)
        x = m_blocks[i]->forward(x);

    // Norm
    x = m_norm.forward(x);

    // LM Head
    return xt::linalg::dot(x, m_lmHead);
}
