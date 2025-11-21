#include "OlmoAttention.h"

#include <cmath>
#include <format>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

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
    // Weight matrices are [d_model, d_model], need transpose for multiplication
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
    const auto vs = xt::eval(xt::reshape_view(projected_vs, {batch_size, seq_len, n_heads, head_dim}));

    // qs, ks, vs are all (batch_size, seq_len, n_heads, head_dim)

    // apply RoPE
    const auto qs_with_rope = xt::eval(apply_rope(qs));
    const auto ks_with_rope = xt::eval(apply_rope(ks));

    // attend - parallelized with TBB
    auto attention_output = xt::zeros_like(input);
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, batch_size, 0, seq_len),
        [&](const tbb::blocked_range2d<size_t>& r) {
            for (size_t b = r.rows().begin(); b != r.rows().end(); ++b) {
                for (size_t position = r.cols().begin(); position != r.cols().end(); ++position) {
                    const auto q = xt::view(qs_with_rope, b, position, xt::newaxis(), xt::all());
                    const auto k = xt::view(ks_with_rope, b, xt::range(0, position + 1)); // causal mask is in here
                    const auto v = xt::view(vs, b, xt::range(0, position + 1));
                    const auto logits = xt::sum(q * k, {2}) / std::sqrt(head_dim);

                    auto softmax = xt::eval(xt::exp(logits));
                    const auto exp_logits_sum = xt::eval(xt::sum(softmax, {0}));
                    xt::noalias(softmax) /= exp_logits_sum;
                    // softmax is (seq, n_heads)

                    // apply weights to V
                    const auto weighted_sums =
                        xt::sum(v * xt::view(softmax, xt::all(), xt::all(), xt::newaxis()), {0});
                    // weighted_sums is (n_heads, head_dim)

                    xt::noalias(xt::view(attention_output, b, position)) +=
                        xt::reshape_view(weighted_sums, {n_heads * head_dim});
                }
            }
        }
    );

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
    // rope buffers are (seq_len, head_dim)

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
