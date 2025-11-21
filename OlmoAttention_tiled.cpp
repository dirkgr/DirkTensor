#include "OlmoAttention.h"

#include <cmath>
#include <format>
#include <algorithm>
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
    const auto qs = xt::reshape_view(normed_qs, {batch_size, seq_len, n_heads, head_dim});

    auto projected_ks_2d = xt::linalg::dot(input_2d, xt::transpose(m_kProj));
    auto projected_ks = xt::reshape_view(projected_ks_2d, {batch_size, seq_len, d_model});
    const auto normed_ks = m_kNorm.forward(projected_ks);
    const auto ks = xt::reshape_view(normed_ks, {batch_size, seq_len, n_heads, head_dim});

    auto projected_vs_2d = xt::linalg::dot(input_2d, xt::transpose(m_vProj));
    auto projected_vs = xt::reshape_view(projected_vs_2d, {batch_size, seq_len, d_model});
    const auto vs = xt::eval(xt::reshape_view(projected_vs, {batch_size, seq_len, n_heads, head_dim}));

    // Apply RoPE
    const auto qs_with_rope = xt::eval(apply_rope(qs));
    const auto ks_with_rope = xt::eval(apply_rope(ks));

    // Tiled attention computation
    auto attention_output = compute_tiled_attention(qs_with_rope, ks_with_rope, vs, batch_size, seq_len);

    // Output projection using efficient BLAS
    auto output_2d = xt::reshape_view(attention_output, {batch_size * seq_len, d_model});
    auto result_2d = xt::linalg::dot(output_2d, xt::transpose(m_oProj));
    auto result = xt::reshape_view(result_2d, {batch_size, seq_len, d_model});

    return xt::eval(result);
}


// Tiled attention implementation inspired by Flash Attention but for CPU
xt::xtensor<float, 3> OlmoAttention::compute_tiled_attention(
    const xt::xtensor<float, 4>& qs_with_rope,
    const xt::xtensor<float, 4>& ks_with_rope,
    const xt::xtensor<float, 4>& vs,
    size_t batch_size, size_t seq_len) {

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attention_output = xt::eval(xt::zeros<float>({batch_size, seq_len, static_cast<size_t>(n_heads * head_dim)}));

    // Tile sizes optimized for cache (L1 cache is typically 32KB-64KB)
    // Each tile should fit Q, K, V blocks in L1/L2 cache
    const size_t Q_TILE = 16;  // Query tile size
    const size_t KV_TILE = 32; // Key/Value tile size

    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, batch_size, 1, 0, seq_len, Q_TILE),
        [&](const tbb::blocked_range2d<size_t>& r) {
            for (size_t b = r.rows().begin(); b != r.rows().end(); ++b) {
                // Process queries in tiles
                for (size_t q_start = r.cols().begin(); q_start != r.cols().end(); q_start += Q_TILE) {
                    size_t q_end = std::min(q_start + Q_TILE, seq_len);

                    // For each query tile, we need to compute attention with all previous keys
                    // Process keys/values in tiles for better cache usage
                    for (size_t q_pos = q_start; q_pos < q_end; ++q_pos) {
                        // Online softmax - compute max and sum incrementally
                        auto max_scores = xt::eval(xt::zeros<float>({static_cast<size_t>(n_heads)}) - std::numeric_limits<float>::infinity());
                        auto sum_exp = xt::eval(xt::zeros<float>({static_cast<size_t>(n_heads)}));
                        auto output_accum = xt::eval(xt::zeros<float>({static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)}));

                        // Process K,V in tiles
                        for (size_t kv_start = 0; kv_start <= q_pos; kv_start += KV_TILE) {
                            size_t kv_end = std::min(kv_start + KV_TILE, q_pos + 1);
                            size_t tile_size = kv_end - kv_start;

                            // Load query for this position (reuse across KV tiles)
                            auto q = xt::view(qs_with_rope, b, q_pos); // [n_heads, head_dim]

                            // Compute scores for this KV tile
                            auto tile_scores = xt::eval(xt::zeros<float>({static_cast<size_t>(n_heads), tile_size}));

                            // Vectorized score computation for all heads
                            for (size_t h = 0; h < n_heads; ++h) {
                                auto q_head = xt::view(q, h); // [head_dim]
                                for (size_t kv_idx = 0; kv_idx < tile_size; ++kv_idx) {
                                    auto k_head = xt::view(ks_with_rope, b, kv_start + kv_idx, h);
                                    float score = 0.0f;
                                    // Manual dot product for better control
                                    for (size_t d = 0; d < head_dim; ++d) {
                                        score += q_head(d) * k_head(d);
                                    }
                                    tile_scores(h, kv_idx) = score * scale;
                                }
                            }

                            // Online softmax update
                            for (size_t h = 0; h < n_heads; ++h) {
                                // Find max in this tile
                                float tile_max = -std::numeric_limits<float>::infinity();
                                for (size_t i = 0; i < tile_size; ++i) {
                                    tile_max = std::max(tile_max, tile_scores(h, i));
                                }

                                // Update running max and adjust previous sum
                                float old_max = max_scores(h);
                                float new_max = std::max(old_max, tile_max);
                                float adjustment = std::exp(old_max - new_max);

                                // Update sum with adjustment
                                sum_exp(h) *= adjustment;
                                // Scale all accumulated output values for this head
                                for (size_t d = 0; d < head_dim; ++d) {
                                    output_accum(h, d) *= adjustment;
                                }

                                // Add this tile's contribution
                                for (size_t i = 0; i < tile_size; ++i) {
                                    float exp_score = std::exp(tile_scores(h, i) - new_max);
                                    sum_exp(h) += exp_score;

                                    // Accumulate weighted values
                                    auto v_head = xt::view(vs, b, kv_start + i, h); // [head_dim]
                                    for (size_t d = 0; d < head_dim; ++d) {
                                        output_accum(h, d) += exp_score * v_head(d);
                                    }
                                }

                                max_scores(h) = new_max;
                            }
                        }

                        // Final normalization
                        for (size_t h = 0; h < n_heads; ++h) {
                            for (size_t d = 0; d < head_dim; ++d) {
                                output_accum(h, d) /= sum_exp(h);
                            }
                        }

                        // Store result
                        xt::noalias(xt::view(attention_output, b, q_pos)) = xt::flatten(output_accum);
                    }
                }
            }
        }
    );

    return attention_output;
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