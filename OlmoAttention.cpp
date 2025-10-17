#include "OlmoAttention.h"

#include <cmath>
#include <format>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

OlmoAttention::OlmoAttention(const std::string& folder, const unsigned int index) :
    m_qNorm(std::format("{}/model.layers.{}.self_attn.q_norm.weight.npy", folder, index)),
    m_kNorm(std::format("{}/model.layers.{}.self_attn.k_norm.weight.npy", folder, index))
{
    m_qProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.q_proj.weight.npy", folder, index));
    m_kProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.k_proj.weight.npy", folder, index));
    m_vProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.v_proj.weight.npy", folder, index));
    m_oProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.o_proj.weight.npy", folder, index));

    // kv cache
    m_kCache = xt::empty<float>({seq_len, n_heads, head_dim});
    m_vCache = xt::empty<float>({seq_len, n_heads, head_dim});

    // Precompute RoPE sin/cos tables manually for better performance
    m_rope_sin.resize(seq_len * head_dim);
    m_rope_cos.resize(seq_len * head_dim);

    // Compute inverse frequencies: inv_freq[i] = 1.0 / (theta ^ (2i/head_dim))
    std::vector<double> inv_freq(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; ++i) {
        inv_freq[i] = 1.0 / std::pow(theta, (2.0 * i) / head_dim);
    }

    // Compute sin/cos for each position and dimension
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < head_dim / 2; ++i) {
            double angle = pos * inv_freq[i];
            float sin_val = std::sin(angle);
            float cos_val = std::cos(angle);

            // Store twice (concatenated pattern from original implementation)
            m_rope_sin[pos * head_dim + i] = sin_val;
            m_rope_sin[pos * head_dim + head_dim/2 + i] = sin_val;
            m_rope_cos[pos * head_dim + i] = cos_val;
            m_rope_cos[pos * head_dim + head_dim/2 + i] = cos_val;
        }
    }
}

xt::xtensor<float, 1> OlmoAttention::forward(const xt::xtensor<float, 1>& input) {
    const auto q = xt::reshape_view(m_qNorm.forward(xt::linalg::dot(m_qProj, input)), {n_heads, head_dim});
    const auto k = xt::reshape_view(m_kNorm.forward(xt::linalg::dot(m_kProj, input)), {n_heads, head_dim});
    const auto v = xt::reshape_view(xt::linalg::dot(m_vProj, input), {n_heads, head_dim});
    // q, k, v are all (n_heads, head_dim)

    // apply RoPE
    const auto q_with_rope = apply_rope(q, m_kvCacheEnd);
    const auto k_with_rope = apply_rope(k, m_kvCacheEnd);

    // put into cache
    xt::view(m_kCache, m_kvCacheEnd) = k_with_rope;
    xt::view(m_vCache, m_kvCacheEnd) = v;
    m_kvCacheEnd += 1;

    const auto ks = xt::view(m_kCache, xt::range(0, m_kvCacheEnd));
    const auto vs = xt::view(m_vCache, xt::range(0, m_kvCacheEnd));
    // ks and vs are (seq, n_heads, head_dim)

    // attend
    const auto logits = xt::sum(xt::view(q_with_rope, xt::newaxis(), xt::all()) * ks, {2}) / std::sqrt(head_dim);
    // logits are (seq, n_heads)

    // softmax
    const auto exp_logits = xt::eval(xt::exp(logits));
    const auto exp_logits_sum = xt::sum(exp_logits, {0});
    const auto softmax = exp_logits / exp_logits_sum;
    // softmax is (seq, n_heads)

    // apply weights to V
    const auto weighted_sums = xt::sum(vs * xt::view(softmax, xt::all(), xt::all(), xt::newaxis()), {0});
    // weighted_sums is (n_heads, head_dim)
    const auto attention_output = xt::eval(xt::reshape_view(weighted_sums, {n_heads * head_dim}));
    // attention_output is (d_model,)

    return xt::linalg::dot(m_oProj, attention_output);
}

xt::xtensor<float, 2> OlmoAttention::apply_rope(const xt::xtensor<float, 2>& input, size_t position) {
    // Manual implementation for better performance - avoids xtensor overhead and concatenate
    // Input dimensions: (n_heads, head_dim)

    xt::xtensor<float, 2> output = xt::empty<float>({n_heads, head_dim});

    const float* sin_row = &m_rope_sin[position * head_dim];
    const float* cos_row = &m_rope_cos[position * head_dim];

    // For each head
    for (size_t head = 0; head < n_heads; ++head) {
        const float* input_ptr = &input(head, 0);
        float* output_ptr = &output(head, 0);

        // Apply RoPE: output = input * cos + rotated(input) * sin
        // Rotated means: [-input[64:128], input[0:64]]
        for (size_t i = 0; i < head_dim / 2; ++i) {
            // First half: output[i] = input[i] * cos[i] + (-input[i + half]) * sin[i]
            output_ptr[i] = input_ptr[i] * cos_row[i] - input_ptr[i + head_dim/2] * sin_row[i];

            // Second half: output[i+half] = input[i+half] * cos[i+half] + input[i] * sin[i+half]
            output_ptr[i + head_dim/2] = input_ptr[i + head_dim/2] * cos_row[i + head_dim/2] + input_ptr[i] * sin_row[i + head_dim/2];
        }
    }

    return output;
}
