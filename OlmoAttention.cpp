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
    // Pre-allocate output tensor with proper dimensions
    xt::xtensor<float, 3> attention_output = xt::zeros<float>({batch_size, seq_len, d_model});

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, batch_size, 0, seq_len),
        [&](const tbb::blocked_range2d<size_t>& r) {
            // Pre-allocate thread-local buffers to avoid repeated allocations
            xt::xtensor<float, 1> logits_buffer = xt::zeros<float>({static_cast<size_t>(n_heads)});
            xt::xtensor<float, 1> output_buffer = xt::zeros<float>({static_cast<size_t>(n_heads * head_dim)});

            for (size_t b = r.rows().begin(); b != r.rows().end(); ++b) {
                for (size_t position = r.cols().begin(); position != r.cols().end(); ++position) {
                    // Clear output buffer
                    std::fill(output_buffer.begin(), output_buffer.end(), 0.0f);

                    // For each head, compute attention weights and accumulate
                    for (size_t h = 0; h < n_heads; ++h) {
                        // Compute logits for this head: q @ k^T / sqrt(head_dim)
                        float max_logit = -std::numeric_limits<float>::infinity();

                        // First pass: compute logits and find max for numerical stability
                        for (size_t k_pos = 0; k_pos <= position; ++k_pos) {
                            float dot = 0.0f;
                            for (size_t d = 0; d < head_dim; ++d) {
                                dot += qs_with_rope(b, position, h, d) * ks_with_rope(b, k_pos, h, d);
                            }
                            logits_buffer(h) = dot * scale;  // Reusing buffer temporarily
                            max_logit = std::max(max_logit, logits_buffer(h));
                        }

                        // Second pass: compute softmax and weighted sum
                        float sum_exp = 0.0f;
                        for (size_t k_pos = 0; k_pos <= position; ++k_pos) {
                            float dot = 0.0f;
                            for (size_t d = 0; d < head_dim; ++d) {
                                dot += qs_with_rope(b, position, h, d) * ks_with_rope(b, k_pos, h, d);
                            }
                            float logit = dot * scale;
                            float weight = std::exp(logit - max_logit);
                            sum_exp += weight;

                            // Accumulate weighted values
                            for (size_t d = 0; d < head_dim; ++d) {
                                output_buffer(h * head_dim + d) += weight * vs(b, k_pos, h, d);
                            }
                        }

                        // Normalize by sum of exp
                        for (size_t d = 0; d < head_dim; ++d) {
                            output_buffer(h * head_dim + d) /= sum_exp;
                        }
                    }

                    // Copy to output
                    for (size_t i = 0; i < n_heads * head_dim; ++i) {
                        attention_output(b, position, i) = output_buffer(i);
                    }
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

    // Pre-allocate output with explicit type
    xt::xtensor<float, 4> output = xt::zeros<float>({batch_size, seq_len, static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});

    // Scalar RoPE implementation - avoid xtensor expression template overhead
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t position = 0; position < seq_len; ++position) {
            for (size_t h = 0; h < n_heads; ++h) {
                // RoPE formula: output = input * cos + rotated_input * sin
                // where rotated_input = [-second_half, first_half]
                const size_t half = head_dim / 2;

                // First half of output: input[first_half] * cos - input[second_half] * sin
                for (size_t d = 0; d < half; ++d) {
                    const float cos_val = pos_cos(position, d);
                    const float sin_val = pos_sin(position, d);
                    output(b, position, h, d) =
                        input(b, position, h, d) * cos_val - input(b, position, h, d + half) * sin_val;
                }

                // Second half of output: input[second_half] * cos + input[first_half] * sin
                for (size_t d = 0; d < half; ++d) {
                    const float cos_val = pos_cos(position, d + half);
                    const float sin_val = pos_sin(position, d + half);
                    output(b, position, h, d + half) =
                        input(b, position, h, d + half) * cos_val + input(b, position, h, d) * sin_val;
                }
            }
        }
    }
    return output;
}
