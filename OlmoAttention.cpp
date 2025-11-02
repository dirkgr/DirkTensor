#include "OlmoAttention.h"

#include <cmath>
#include <format>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "xtutil.h"

// Static member definitions
std::vector<float> OlmoAttention::s_rope_sin;
std::vector<float> OlmoAttention::s_rope_cos;

void OlmoAttention::init_rope_tables() {
    // Only initialize once (check if already done)
    if (!s_rope_sin.empty()) {
        return;
    }

    // Precompute RoPE sin/cos tables manually for better performance
    s_rope_sin.resize(max_seq_len * head_dim);
    s_rope_cos.resize(max_seq_len * head_dim);

    // Compute inverse frequencies: inv_freq[i] = 1.0 / (theta ^ (2i/head_dim))
    std::vector<double> inv_freq(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; ++i) {
        inv_freq[i] = 1.0 / std::pow(theta, (2.0 * i) / head_dim);
    }

    // Compute sin/cos for each position and dimension
    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        for (size_t i = 0; i < head_dim / 2; ++i) {
            double angle = pos * inv_freq[i];
            float sin_val = std::sin(angle);
            float cos_val = std::cos(angle);

            // Store twice (concatenated pattern from original implementation)
            s_rope_sin[pos * head_dim + i] = sin_val;
            s_rope_sin[pos * head_dim + head_dim/2 + i] = sin_val;
            s_rope_cos[pos * head_dim + i] = cos_val;
            s_rope_cos[pos * head_dim + head_dim/2 + i] = cos_val;
        }
    }
}

OlmoAttention::OlmoAttention(const std::string& folder, const unsigned int index) :
    m_qNorm(std::format("{}/model.layers.{}.self_attn.q_norm.weight.npy", folder, index)),
    m_kNorm(std::format("{}/model.layers.{}.self_attn.k_norm.weight.npy", folder, index))
{
    m_qProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.q_proj.weight.npy", folder, index));
    m_kProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.k_proj.weight.npy", folder, index));
    m_vProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.v_proj.weight.npy", folder, index));
    m_oProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.o_proj.weight.npy", folder, index));

    // Initialize RoPE tables once (shared across all instances)
    init_rope_tables();
}

xt::xtensor<float, 3> OlmoAttention::forward(const xt::xtensor<float, 3>& input) {
    const unsigned int batch_size = input.shape(0);
    const unsigned int seq_len = input.shape(1);

    const auto projected_qs = batched_projection(input, m_qProj);
    const auto normed_qs = m_qNorm.forward(projected_qs);
    const auto qs = xt::reshape_view(projected_qs, {batch_size, seq_len, n_heads, head_dim});

    const auto projected_ks = batched_projection(input, m_kProj);
    const auto normed_ks = m_kNorm.forward(projected_ks);
    const auto ks = xt::reshape_view(projected_ks, {batch_size, seq_len, n_heads, head_dim});

    const auto projected_vs = batched_projection(input, m_vProj);
    const auto vs = xt::eval(xt::reshape_view(projected_vs, {batch_size, seq_len, n_heads, head_dim}));

    // qs, ks, vs are all (batch_size, seq_len, n_heads, head_dim)

    // apply RoPE
    const auto qs_with_rope = xt::eval(apply_rope(qs));
    const auto ks_with_rope = xt::eval(apply_rope(ks));

    // attend
    auto attention_output = xt::zeros_like(input);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t position = 0; position < seq_len; ++position) {
            const auto q = xt::view(qs_with_rope, b, position, xt::newaxis(), xt::all());
            const auto logits = xt::sum(q * ks_with_rope, {2}) / std::sqrt(head_dim);
            const auto exp_logits = xt::eval(xt::exp(logits));
            const auto exp_logits_sum = xt::sum(exp_logits, {0});
            const auto softmax = exp_logits / exp_logits_sum;
            // softmax is (seq, n_heads)

            // apply weights to V
            const auto weighted_sums =
                xt::sum(vs * xt::view(softmax, xt::all(), xt::all(), xt::newaxis()), {0});
            // weighted_sums is (n_heads, head_dim)
            xt::view(attention_output, b, position) +=
                xt::reshape_view(weighted_sums, {n_heads * head_dim});
        }
    }

    return batched_projection(attention_output, m_oProj);
}

xt::xtensor<float, 4> OlmoAttention::apply_rope(const xt::xtensor<float, 4>& input) {
    // Manual implementation for better performance - avoids xtensor overhead and concatenate
    // Input dimensions: (batch_size, seq_len, n_heads, head_dim)
    const auto batch_size = input.shape(0);
    const auto seq_len = input.shape(1);
    const auto n_heads = input.shape(2);
    const auto head_dim = input.shape(3);

    xt::xtensor<float, 4> output = xt::empty_like(input);

    for (size_t position = 0; position < seq_len; position++) {
        const float* sin_row = &s_rope_sin[position * head_dim];
        const float* cos_row = &s_rope_cos[position * head_dim];

        for (size_t b = 0; b < batch_size; ++b) {
            // For each head
            for (size_t head = 0; head < n_heads; ++head) {
                const float* input_ptr = &input(b, position, head, 0);
                float* output_ptr = &output(b, position, head, 0);

                // Apply RoPE: output = input * cos + rotated(input) * sin
                // Rotated means: [-input[64:128], input[0:64]]
                for (size_t i = 0; i < head_dim / 2; ++i) {
                    // First half: output[i] = input[i] * cos[i] + (-input[i + half]) * sin[i]
                    output_ptr[i] = input_ptr[i] * cos_row[i] - input_ptr[i + head_dim/2] * sin_row[i];

                    // Second half: output[i+half] = input[i+half] * cos[i+half] + input[i] * sin[i+half]
                    output_ptr[i + head_dim/2] = input_ptr[i + head_dim/2] * cos_row[i + head_dim/2] + input_ptr[i] * sin_row[i + head_dim/2];
                }
            }
        }
    }

    return output;
}
