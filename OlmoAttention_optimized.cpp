#include "OlmoAttention.h"

#include <cmath>
#include <format>
#include <iostream>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xnoalias.hpp>

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
    // rope buffers are (seq_len, head_dim)
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
    // m_qProj is [d_model, d_model], we need to transpose for correct multiplication
    auto projected_qs_2d = xt::linalg::dot(input_2d, xt::transpose(m_qProj));
    auto projected_qs = xt::reshape_view(projected_qs_2d, {batch_size, seq_len, d_model});
    const auto normed_qs = m_qNorm.forward(projected_qs);
    const auto qs = xt::reshape_view(normed_qs, {batch_size, seq_len, n_heads, head_dim});

    auto projected_ks_2d = xt::linalg::dot(input_2d, xt::transpose(m_kProj));
    auto projected_ks = xt::reshape_view(projected_ks_2d, {batch_size, seq_len, d_model});
    const auto normed_ks = m_kNorm.forward(projected_ks);
    const auto ks = xt::reshape_view(normed_ks, {batch_size, seq_len, n_heads, head_dim});

    auto projected_vs_2d = xt::linalg::dot(input_2d, xt::transpose(m_vProj));
    auto projected_vs = xt::reshape_view(projected_vs_2d, {batch_size, seq_len, d_model});
    const auto vs = xt::reshape_view(projected_vs, {batch_size, seq_len, n_heads, head_dim});

    // Apply RoPE (vectorized version)
    const auto qs_with_rope = apply_rope_vectorized(qs);
    const auto ks_with_rope = apply_rope_vectorized(ks);

    // Batched attention computation
    auto attention_output = compute_attention_batched(qs_with_rope, ks_with_rope, vs, batch_size, seq_len);

    // Output projection using efficient BLAS
    auto output_2d = xt::reshape_view(attention_output, {batch_size * seq_len, d_model});
    auto result_2d = xt::linalg::dot(output_2d, xt::transpose(m_oProj));
    auto result = xt::reshape_view(result_2d, {batch_size, seq_len, d_model});

    return xt::eval(result);
}


xt::xtensor<float, 3> OlmoAttention::compute_attention_batched(
    const xt::xtensor<float, 4>& qs_with_rope,
    const xt::xtensor<float, 4>& ks_with_rope,
    const xt::xtensor<float, 4>& vs,
    size_t batch_size, size_t seq_len) {

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attention_output = xt::zeros<float>({batch_size, seq_len, n_heads * head_dim});

    // Process attention for each batch
    for (size_t b = 0; b < batch_size; ++b) {
        // Transpose K and V for more efficient computation
        // ks shape: [seq_len, n_heads, head_dim] -> [n_heads, head_dim, seq_len]
        auto k_batch = xt::view(ks_with_rope, b);
        auto k_transposed = xt::transpose(k_batch, {1, 2, 0});

        // vs shape: [seq_len, n_heads, head_dim] -> [n_heads, seq_len, head_dim]
        auto v_batch = xt::view(vs, b);
        auto v_transposed = xt::transpose(v_batch, {1, 0, 2});

        // Process all heads at once for each position
        for (size_t pos = 0; pos < seq_len; ++pos) {
            // Query for current position: [n_heads, head_dim]
            auto q = xt::view(qs_with_rope, b, pos);

            // Compute scores for all heads at once
            // q shape: [n_heads, head_dim]
            // k_transposed[:, :, :pos+1] shape: [n_heads, head_dim, pos+1]
            // Result: [n_heads, pos+1]
            auto k_slice = xt::view(k_transposed, xt::all(), xt::all(), xt::range(0, pos + 1));
            auto scores = xt::zeros<float>({n_heads, pos + 1});

            for (size_t h = 0; h < n_heads; ++h) {
                auto q_head = xt::view(q, h);
                auto k_head = xt::view(k_slice, h);
                auto score_head = xt::linalg::dot(q_head, k_head);
                xt::noalias(xt::view(scores, h)) = score_head * scale;
            }

            // Apply softmax
            auto max_scores = xt::amax(scores, {1});
            auto exp_scores = xt::exp(scores - xt::view(max_scores, xt::all(), xt::newaxis()));
            auto sum_exp = xt::sum(exp_scores, {1});
            auto attn_weights = exp_scores / xt::view(sum_exp, xt::all(), xt::newaxis());

            // Apply attention weights to values
            // attn_weights shape: [n_heads, pos+1]
            // v_transposed[:, :pos+1, :] shape: [n_heads, pos+1, head_dim]
            // Result: [n_heads, head_dim]
            auto v_slice = xt::view(v_transposed, xt::all(), xt::range(0, pos + 1), xt::all());
            auto weighted_v = xt::zeros<float>({n_heads, head_dim});

            for (size_t h = 0; h < n_heads; ++h) {
                auto weights_head = xt::view(attn_weights, h);
                auto v_head = xt::view(v_slice, h);
                auto result = xt::sum(v_head * xt::view(weights_head, xt::all(), xt::newaxis()), {0});
                xt::noalias(xt::view(weighted_v, h)) = result;
            }

            // Reshape and store
            xt::noalias(xt::view(attention_output, b, pos)) = xt::flatten(weighted_v);
        }
    }

    return attention_output;
}


xt::xtensor<float, 4> OlmoAttention::apply_rope_vectorized(const xt::xtensor<float, 4>& input) {
    // Input dimensions: (batch_size, seq_len, n_heads, head_dim)
    const auto batch_size = input.shape(0);
    const auto seq_len = input.shape(1);

    static const auto [pos_sin, pos_cos] = rope_buffers();
    // rope buffers are (seq_len, head_dim)

    auto output = xt::zeros_like(input);

    // Vectorized RoPE application
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t pos = 0; pos < seq_len; ++pos) {
            auto input_pos = xt::view(input, b, pos); // [n_heads, head_dim]
            auto output_pos = xt::view(output, b, pos); // [n_heads, head_dim]

            // Apply RoPE to all heads at once
            // First half rotation
            auto input_first_half = xt::view(input_pos, xt::all(), xt::range(0, head_dim / 2));
            auto input_second_half = xt::view(input_pos, xt::all(), xt::range(head_dim / 2, head_dim));

            // Sin component
            auto sin_part = xt::concatenate(std::tuple(-input_second_half, input_first_half), 1);
            auto sin_coeff = xt::view(pos_sin, pos);

            // Cos component
            auto cos_coeff = xt::view(pos_cos, pos);

            // Combined RoPE: x * cos + rotated_x * sin
            xt::noalias(output_pos) = input_pos * cos_coeff + sin_part * sin_coeff;
        }
    }

    return output;
}


// Keep original implementations as fallback
xt::xtensor<float, 4> OlmoAttention::apply_rope(const xt::xtensor<float, 4>& input) {
    return apply_rope_vectorized(input);
}