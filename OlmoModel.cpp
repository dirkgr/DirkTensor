#include "OlmoModel.h"

#include <cassert>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "xtutil.h"

OlmoModel::OlmoModel(const std::string& folder) : m_norm(folder + "/model.norm.weight.npy") {
    m_embeddings.w = xt::load_npy<float>(folder + "/model.embed_tokens.weight.npy");
    assert(m_embeddings.shape().size() == 2);
    assert(m_embeddings.shape(1) == d_model);

    m_lmHead.w = xt::load_npy<float>(folder + "/lm_head.weight.npy");

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
    m_actBeforeLmHead = m_norm.forward(x);

    // LM Head - optimized with reshape + dot instead of tensordot
    const size_t batch_size = x.shape(0);
    const size_t seq_len = x.shape(1);
    const size_t hidden_dim = x.shape(2);
    const size_t vocab_size = m_lmHead.shape(0);

    // Reshape from [batch, seq, d_model] to [batch*seq, d_model]
    auto x_2d = xt::reshape_view(m_actBeforeLmHead, {batch_size * seq_len, hidden_dim});

    // Matrix multiply: [batch*seq, d_model] @ [d_model, vocab_size] -> [batch*seq, vocab_size]
    // m_lmHead is [vocab_size, d_model], so we need to transpose it
    auto logits_2d = xt::linalg::dot(x_2d, xt::transpose(m_lmHead.w));

    // Reshape back to [batch, seq, vocab_size]
    return xt::eval(xt::reshape_view(logits_2d, {batch_size, seq_len, vocab_size}));
}

void OlmoModel::backward(const xt::xtensor<float, 3>& grad) {
    if (m_lmHead.grad.size() == 0)
        m_lmHead.grad = xt::zeros_like(m_lmHead.w);

    const size_t batch_size = grad.shape(0);
    const size_t seq_len = grad.shape(1);
    const size_t vocab_size = grad.shape(2);
    assert(m_lmHead.shape(0) == vocab_size);
    const size_t d_model = m_lmHead.shape(1);

    const auto reshapedActBeforeLmHead = xt::reshape_view(
        m_actBeforeLmHead,
        {batch_size * seq_len, d_model});
    const auto reshapedGrad = xt::reshape_view(
        grad,
        {batch_size * seq_len, vocab_size});
    m_lmHead.grad += xt::linalg::dot(       // (vocab_size, d_model)
        xt::transpose(reshapedGrad),        // (vocab_size, batch_size * seq_len)
        reshapedActBeforeLmHead);           // (batch_size * seq_len, d_model)

    print_shape(m_lmHead.grad);
    std::cout << "m_lmHead.grad" << std::endl << m_lmHead.grad << std::endl;
}
