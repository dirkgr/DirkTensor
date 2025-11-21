#include "OlmoAttention.h"

#include <cmath>
#include <format>
#include <algorithm>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/views/xview.hpp>

#include "xtutil.h"
#include "model_config.h"

static constexpr float rope_theta = 500000;

static constexpr auto rope_buffers() {
    const auto inv_freq =
        1.0 / (xt::pow(rope_theta, xt::arange<double>(0, head_dim, 2) / head_dim));
    const auto seq = xt::arange<double>(0, max_seq_len);
    const auto freqs =
        xt::eval(xt::view(seq, xt::all(), xt::newaxis()) * xt::view(inv_freq, xt::newaxis(), xt::all()));
    const auto positions = xt::concatenate(std::tuple(freqs, freqs), 1);
    const auto pos_sin = xt::sin(positions);
    const auto pos_cos = xt::cos(positions);

    const xt::xtensor<float, 2> pos_sin_f = xt::cast<float>(pos_sin);
    const xt::xtensor<float, 2> pos_cos_f = xt::cast<float>(pos_cos);

    return std::pair(pos_sin_f, pos_cos_f);
}


OlmoAttention::OlmoAttention(const std::string& folder, const unsigned int index) :
    m_qNorm(std::format("{}/model.layers.{}.self_attn.q_norm.weight.npy", folder, index)),
    m_kNorm(std::format("{}/model.layers.{}.self_attn.k_norm.weight.npy", folder, index))
{
    m_qProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.q_proj.weight.npy", folder, index));
    m_kProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.k_proj.weight.npy", folder, index));
    m_vProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.v_proj.weight.npy", folder, index));
    m_oProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.o_proj.weight.npy", folder, index));
}


xt::xtensor<float, 3> OlmoAttention::forward(const xt::xtensor<float, 3>& input) {
    const unsigned int batch_size = input.shape(0);
    const unsigned int seq_len = input.shape(1);
    const unsigned int d_model = input.shape(2);

    // Optimize projections using reshape + dot (same as MLP optimization)
    // Reshape input from [batch, seq, d_model] to [batch*seq, d_model]
    auto input_2d = xt::reshape_view(input, {batch_size * seq_len, d_model});

    // Project Q, K, V using efficient BLAS operations
    auto projected_qs_2d = xt::linalg::dot(input_2d, xt::transpose(m_qProj));
    auto projected_qs = xt::reshape_view(projected_qs_2d, {batch_size, seq_len, d_model});
    const auto normed_qs = m_qNorm.forward(projected_qs);
    auto qs = xt::eval(xt::reshape_view(normed_qs, {batch_size, seq_len, n_heads, head_dim}));

    auto projected_ks_2d = xt::linalg::dot(input_2d, xt::transpose(m_kProj));
    auto projected_ks = xt::reshape_view(projected_ks_2d, {batch_size, seq_len, d_model});
    const auto normed_ks = m_kNorm.forward(projected_ks);
    auto ks = xt::eval(xt::reshape_view(normed_ks, {batch_size, seq_len, n_heads, head_dim}));

    auto projected_vs_2d = xt::linalg::dot(input_2d, xt::transpose(m_vProj));
    auto projected_vs = xt::reshape_view(projected_vs_2d, {batch_size, seq_len, d_model});
    auto vs = xt::eval(xt::reshape_view(projected_vs, {batch_size, seq_len, n_heads, head_dim}));

    // Apply RoPE
    qs = xt::eval(apply_rope(qs));
    ks = xt::eval(apply_rope(ks));

    // Batch attention computation
    // Process all heads and positions at once instead of sequential processing
    auto attention_output = xt::eval(xt::zeros<float>({batch_size, seq_len, d_model}));

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < n_heads; ++h) {
            // Get Q, K, V for this batch and head
            auto q_head = xt::view(qs, b, xt::all(), h, xt::all()); // [seq_len, head_dim]
            auto k_head = xt::view(ks, b, xt::all(), h, xt::all()); // [seq_len, head_dim]
            auto v_head = xt::view(vs, b, xt::all(), h, xt::all()); // [seq_len, head_dim]

            // Compute attention scores for all query-key pairs at once
            // scores = Q @ K^T / sqrt(head_dim)
            auto scores = xt::eval(xt::linalg::dot(q_head, xt::transpose(k_head)) / std::sqrt(static_cast<float>(head_dim)));

            // Apply causal mask (set future positions to -inf)
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = i + 1; j < seq_len; ++j) {
                    scores(i, j) = -std::numeric_limits<float>::infinity();
                }
            }

            // Compute softmax row-wise
            auto exp_scores = xt::exp(scores - xt::amax(scores, {1}, xt::keep_dims));
            auto attention_weights = exp_scores / xt::sum(exp_scores, {1}, xt::keep_dims);

            // Apply attention weights to values
            auto head_output = xt::linalg::dot(attention_weights, v_head); // [seq_len, head_dim]

            // Copy to output tensor
            for (size_t pos = 0; pos < seq_len; ++pos) {
                for (size_t d = 0; d < head_dim; ++d) {
                    attention_output(b, pos, h * head_dim + d) = head_output(pos, d);
                }
            }
        }
    }

    // Output projection using efficient BLAS
    auto output_2d = xt::reshape_view(attention_output, {batch_size * seq_len, d_model});
    auto result_2d = xt::linalg::dot(output_2d, xt::transpose(m_oProj));
    auto result = xt::reshape_view(result_2d, {batch_size, seq_len, d_model});

    return xt::eval(result);
}


xt::xtensor<float, 4> OlmoAttention::apply_rope(const xt::xtensor<float, 4>& input) {
    // Input dimensions: (batch_size, seq_len, n_heads, head_dim)
    const auto batch_size = input.shape(0);
    const auto seq_len = input.shape(1);

    static const auto [pos_sin, pos_cos] = rope_buffers();

    auto output = xt::zeros_like(input);

    // Vectorized RoPE - process all heads at once
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t position = 0; position < seq_len; ++position) {
            auto input_pos = xt::view(input, b, position); // [n_heads, head_dim]
            auto output_pos = xt::view(output, b, position); // [n_heads, head_dim]

            // Rotate first half with second half (for all heads simultaneously)
            auto input_first_half = xt::view(input_pos, xt::all(), xt::range(0, head_dim / 2));
            auto input_second_half = xt::view(input_pos, xt::all(), xt::range(head_dim / 2, head_dim));

            // Build rotated vector for sin component
            auto output_first_half = xt::view(output_pos, xt::all(), xt::range(0, head_dim / 2));
            auto output_second_half = xt::view(output_pos, xt::all(), xt::range(head_dim / 2, head_dim));

            xt::noalias(output_first_half) = -input_second_half;
            xt::noalias(output_second_half) = input_first_half;

            // Apply sin and cos components (broadcast across all heads)
            auto sin_coeff = xt::view(pos_sin, position); // [head_dim]
            auto cos_coeff = xt::view(pos_cos, position); // [head_dim]

            // RoPE formula: x * cos + rotated_x * sin
            xt::noalias(output_pos) = output_pos * sin_coeff + input_pos * cos_coeff;
        }
    }
    return output;
}