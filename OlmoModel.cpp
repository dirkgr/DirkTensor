#include "OlmoModel.h"

#include <cassert>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "xtutil.h"

OlmoModel::OlmoModel(const std::string& folder) :
    m_norm(folder + "/model.norm.weight.npy"),
    m_lmHead(folder + "/lm_head.weight.npy")
{
    m_embeddings.w = xt::load_npy<float>(folder + "/model.embed_tokens.weight.npy");
    assert(m_embeddings.shape().size() == 2);
    assert(m_embeddings.shape(1) == d_model);

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
            xt::noalias(xt::view(x, b, i)) = xt::view(m_embeddings.w, batch(b, i));
        }
    }

    // Blocks
    for(size_t i = 0; i < n_layers; i++)
        x = m_blocks[i]->forward(x);

    // Norm
    const auto normed_x = m_norm.forward(x);

    return m_lmHead.forward(normed_x);
}

void OlmoModel::backward(const xt::xtensor<float, 3>& d_output) {
    auto grad = m_lmHead.backward(d_output);
    grad = m_norm.backward(grad);

    for(int i = n_layers - 1; i >= 0; i--)
        grad = m_blocks[i]->backward(grad);

    // TODO
}
