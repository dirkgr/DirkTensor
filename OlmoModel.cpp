#include "OlmoModel.h"

#include <cassert>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "xtutil.h"

OlmoModel::OlmoModel(const std::string& folder) : m_norm(folder + "/model.norm.weight.npy") {
    m_embeddings = xt::load_npy<float>(folder + "/model.embed_tokens.weight.npy");
    assert(m_embeddings.shape().size() == 2);
    assert(m_embeddings.shape()[1] == d_model);

    m_lmHead = xt::load_npy<float>(folder + "/lm_head.weight.npy");

    for(size_t i = 0; i < n_layers; i++)
        m_blocks[i] = std::make_unique<OlmoBlock>(folder, i);
}

xt::xtensor<float, 3> OlmoModel::forward(const xt::xtensor<uint32_t, 2>& batch) {
    // Embedding
    xt::xtensor<float, 3> x = xt::empty<float>({
        batch.shape(0),
        batch.shape(1),
        m_embeddings.shape(1)
    });
    for (size_t b = 0; b < batch.shape(0); b++) {
        for (size_t i = 0; i < batch.shape(1); i++) {
            xt::noalias(xt::view(x, b, i)) = xt::view(m_embeddings, batch(b, i));
        }
    }

    // Blocks
    for(size_t i = 0; i < n_layers; i++)
        x = m_blocks[i]->forward(x);

    // Norm
    x = m_norm.forward(x);

    // LM Head
    return batched_projection(x, m_lmHead);
}
