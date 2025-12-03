#include "OlmoLmHead.h"

#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

OlmoLmHead::OlmoLmHead(const std::string &filename) {
    m_lmHead.w = xt::load_npy<float>(filename);
}

xt::xtensor<float, 3> OlmoLmHead::forward(const xt::xtensor<float, 3>& input) {
    m_activationsBefore = input;

    const size_t batch_size = input.shape(0);
    const size_t seq_len = input.shape(1);
    const size_t hidden_dim = input.shape(2);
    const size_t vocab_size = m_lmHead.shape(0);

    // Reshape from [batch, seq, d_model] to [batch*seq, d_model]
    const auto x = xt::reshape_view(m_activationsBefore, {batch_size * seq_len, hidden_dim});

    // Matrix multiply: [batch*seq, d_model] @ [d_model, vocab_size] -> [batch*seq, vocab_size]
    // m_lmHead is [vocab_size, d_model], so we need to transpose it
    auto logits_2d = xt::linalg::dot(x, xt::transpose(m_lmHead.w));

    // Reshape back to [batch, seq, vocab_size]
    return xt::reshape_view(logits_2d, {batch_size, seq_len, vocab_size});
}

xt::xtensor<float, 3> OlmoLmHead::backward(const xt::xtensor<float, 3>& grad) {
    if (m_lmHead.grad.size() == 0)
        m_lmHead.grad = xt::zeros_like(m_lmHead.w);

    const size_t batch_size = grad.shape(0);
    const size_t seq_len = grad.shape(1);
    const size_t vocab_size = grad.shape(2);
    assert(m_lmHead.shape(0) == vocab_size);
    const size_t d_model = m_lmHead.shape(1);

    // gradient with respect to the lm head weights
    const auto reshaped_activations_before = xt::reshape_view(
        m_activationsBefore,
        {batch_size * seq_len, d_model});
    const auto reshaped_grad = xt::reshape_view(
        grad,
        {batch_size * seq_len, vocab_size});
    m_lmHead.grad += xt::linalg::dot(       // (vocab_size, d_model)
        xt::transpose(reshaped_grad),       // (vocab_size, batch_size * seq_len)
        reshaped_activations_before);       // (batch_size * seq_len, d_model)

    // gradient with respect to the input
    const auto reshaped_result = xt::linalg::dot(
        reshaped_grad,   // (batch_size * seq_len, vocab_size)
        m_lmHead.w);     // (vocab_size, d_model)
    return xt::reshape_view(reshaped_result, {batch_size, seq_len, d_model});
}
