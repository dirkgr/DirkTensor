#include "OlmoLmHead.h"

#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "cached_path.h"

OlmoLmHead::OlmoLmHead(const std::string &filename) {
    m_lmHead.w = xt::load_npy<float>(cached_path(filename));
    m_lmHead.grad = xt::zeros_like(m_lmHead.w);
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
    xt::xtensor<float, 2> reshaped_activations_before = xt::reshape_view(
        m_activationsBefore,
        {batch_size * seq_len, d_model});
    xt::xtensor<float, 2> reshaped_grad = xt::reshape_view(
        grad,
        {batch_size * seq_len, vocab_size});
    // grad += reshaped_grad^T @ reshaped_activations_before
    const int tokens = static_cast<int>(batch_size * seq_len);
    const int vs = static_cast<int>(vocab_size);
    const int dm = static_cast<int>(d_model);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        vs, dm, tokens,
        1.0f, reshaped_grad.data(), vs,
        reshaped_activations_before.data(), dm,
        1.0f, m_lmHead.grad.data(), dm);

    // gradient with respect to the input
    const auto reshaped_result = xt::linalg::dot(
        reshaped_grad,   // (batch_size * seq_len, vocab_size)
        m_lmHead.w);     // (vocab_size, d_model)
    return xt::reshape_view(reshaped_result, {batch_size, seq_len, d_model});
}

void OlmoLmHead::step(float lr) {
    m_lmHead.w -= lr * m_lmHead.grad;
}

void OlmoLmHead::zero_grad() {
    m_lmHead.grad.fill(0.0f);
}
