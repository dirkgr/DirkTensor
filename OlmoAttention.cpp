#include "OlmoAttention.h"

#include <cmath>
#include <format>
#include <vector>
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
    m_qProj.w = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.q_proj.weight.npy", folder, index));
    m_kProj.w = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.k_proj.weight.npy", folder, index));
    m_vProj.w = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.v_proj.weight.npy", folder, index));
    m_oProj.w = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.o_proj.weight.npy", folder, index));

    // Pre-allocate gradients to avoid allocation during backward pass
    m_qProj.grad = xt::zeros_like(m_qProj.w);
    m_kProj.grad = xt::zeros_like(m_kProj.w);
    m_vProj.grad = xt::zeros_like(m_vProj.w);
    m_oProj.grad = xt::zeros_like(m_oProj.w);
}


xt::xtensor<float, 3> OlmoAttention::forward(const xt::xtensor<float, 3>& input) {
    const unsigned int batch_size = input.shape(0);
    const unsigned int seq_len = input.shape(1);
    const unsigned int d_model = input.shape(2);

    // Optimize projections using reshape + dot (same as MLP optimization)
    // Reshape input from [batch, seq, d_model] to [batch*seq, d_model]
    auto input_2d = xt::reshape_view(input, {batch_size * seq_len, d_model});
    m_act_input_2d = input_2d;  // Save for backward

    // Project Q, K, V using efficient BLAS operations
    // Weight matrices are [d_model, d_model], need transpose for multiplication
    auto projected_qs_2d = xt::linalg::dot(input_2d, xt::transpose(m_qProj.w));
    auto projected_qs = xt::reshape_view(projected_qs_2d, {batch_size, seq_len, d_model});
    const auto normed_qs = m_qNorm.forward(projected_qs);
    const auto qs = xt::reshape_view(normed_qs, {batch_size, seq_len, n_heads, head_dim});
    m_act_qs_before_rope = qs;  // Save for backward (after norm, before RoPE)

    auto projected_ks_2d = xt::linalg::dot(input_2d, xt::transpose(m_kProj.w));
    auto projected_ks = xt::reshape_view(projected_ks_2d, {batch_size, seq_len, d_model});
    const auto normed_ks = m_kNorm.forward(projected_ks);
    const auto ks = xt::reshape_view(normed_ks, {batch_size, seq_len, n_heads, head_dim});
    m_act_ks_before_rope = ks;  // Save for backward (after norm, before RoPE)

    auto projected_vs_2d = xt::linalg::dot(input_2d, xt::transpose(m_vProj.w));
    auto projected_vs = xt::reshape_view(projected_vs_2d, {batch_size, seq_len, d_model});
    m_act_vs = xt::eval(xt::reshape_view(projected_vs, {batch_size, seq_len, n_heads, head_dim}));

    // qs, ks, vs are all (batch_size, seq_len, n_heads, head_dim)

    // apply RoPE
    m_act_qs_with_rope = xt::eval(apply_rope(qs));
    m_act_ks_with_rope = xt::eval(apply_rope(ks));

    // attend - parallelized with TBB
    // Pre-allocate output tensor with proper dimensions
    m_act_attention_output = xt::zeros<float>({batch_size, seq_len, d_model});
    // Attention weights: (batch, n_heads, seq_q, seq_k) - causal, so only lower triangle is meaningful
    m_act_attention_weights = xt::zeros<float>({static_cast<size_t>(batch_size), static_cast<size_t>(n_heads), static_cast<size_t>(seq_len), static_cast<size_t>(seq_len)});

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, batch_size, 0, seq_len),
        [&](const tbb::blocked_range2d<size_t>& r) {
            // Pre-allocate thread-local buffers to avoid repeated allocations
            std::vector<float> logits_buffer(seq_len);
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
                                dot += m_act_qs_with_rope(b, position, h, d) * m_act_ks_with_rope(b, k_pos, h, d);
                            }
                            logits_buffer[k_pos] = dot * scale;
                            max_logit = std::max(max_logit, logits_buffer[k_pos]);
                        }

                        // Second pass: compute softmax weights and store them
                        float sum_exp = 0.0f;
                        for (size_t k_pos = 0; k_pos <= position; ++k_pos) {
                            float weight = std::exp(logits_buffer[k_pos] - max_logit);
                            logits_buffer[k_pos] = weight;  // Reuse buffer for unnormalized weights
                            sum_exp += weight;
                        }

                        // Normalize and store weights, compute weighted sum
                        for (size_t k_pos = 0; k_pos <= position; ++k_pos) {
                            float normalized_weight = logits_buffer[k_pos] / sum_exp;
                            m_act_attention_weights(b, h, position, k_pos) = normalized_weight;

                            // Accumulate weighted values
                            for (size_t d = 0; d < head_dim; ++d) {
                                output_buffer(h * head_dim + d) += normalized_weight * m_act_vs(b, k_pos, h, d);
                            }
                        }
                    }

                    // Copy to output
                    for (size_t i = 0; i < n_heads * head_dim; ++i) {
                        m_act_attention_output(b, position, i) = output_buffer(i);
                    }
                }
            }
        }
    );

    // Output projection using efficient BLAS
    auto output_2d = xt::reshape_view(m_act_attention_output, {batch_size * seq_len, d_model});
    auto result_2d = xt::linalg::dot(output_2d, xt::transpose(m_oProj.w));
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


xt::xtensor<float, 4> OlmoAttention::apply_rope_backward(const xt::xtensor<float, 4>& d_output) {
    // Backward pass for RoPE
    // Forward: out[0:h] = in[0:h]*cos - in[h:]*sin, out[h:] = in[h:]*cos + in[0:h]*sin
    // Backward: d_in[0:h] = d_out[0:h]*cos + d_out[h:]*sin, d_in[h:] = -d_out[0:h]*sin + d_out[h:]*cos

    const auto batch_size = d_output.shape(0);
    const auto seq_len = d_output.shape(1);

    static const auto [pos_sin, pos_cos] = rope_buffers();

    xt::xtensor<float, 4> d_input = xt::zeros<float>({batch_size, seq_len, static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});

    // Parallelize over batch and sequence positions
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, batch_size, 0, seq_len),
        [&](const tbb::blocked_range2d<size_t>& r) {
            for (size_t b = r.rows().begin(); b != r.rows().end(); ++b) {
                for (size_t position = r.cols().begin(); position != r.cols().end(); ++position) {
                    for (size_t h = 0; h < n_heads; ++h) {
                        const size_t half = head_dim / 2;

                        for (size_t d = 0; d < half; ++d) {
                            const float cos_val_first = pos_cos(position, d);
                            const float sin_val_first = pos_sin(position, d);
                            const float cos_val_second = pos_cos(position, d + half);
                            const float sin_val_second = pos_sin(position, d + half);

                            // d_in[0:h] = d_out[0:h] * cos[0:h] + d_out[h:] * sin[h:]
                            d_input(b, position, h, d) =
                                d_output(b, position, h, d) * cos_val_first +
                                d_output(b, position, h, d + half) * sin_val_second;

                            // d_in[h:] = -d_out[0:h] * sin[0:h] + d_out[h:] * cos[h:]
                            d_input(b, position, h, d + half) =
                                -d_output(b, position, h, d) * sin_val_first +
                                d_output(b, position, h, d + half) * cos_val_second;
                        }
                    }
                }
            }
        }
    );
    return d_input;
}


xt::xtensor<float, 3> OlmoAttention::backward(const xt::xtensor<float, 3>& d_output) {
    const size_t batch_size = d_output.shape(0);
    const size_t seq_len = d_output.shape(1);
    const size_t d_model = d_output.shape(2);

    // === Step 1: Output projection backward ===
    // d_attn_output = d_output @ W_O (gradient flows back through output projection)
    // dW_O += d_output^T @ attn_output
    xt::xtensor<float, 2> d_output_2d = xt::reshape_view(d_output, {batch_size * seq_len, d_model});
    xt::xtensor<float, 2> attn_output_2d = xt::reshape_view(m_act_attention_output, {batch_size * seq_len, d_model});

    if (m_oProj.grad.size() == 0)
        m_oProj.grad = xt::zeros_like(m_oProj.w);
    // grad += d_output_2d^T @ attn_output_2d
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<int>(d_model), static_cast<int>(d_model), static_cast<int>(batch_size * seq_len),
        1.0f, d_output_2d.data(), static_cast<int>(d_model),
        attn_output_2d.data(), static_cast<int>(d_model),
        1.0f, m_oProj.grad.data(), static_cast<int>(d_model));

    auto d_attn_output_2d = xt::linalg::dot(d_output_2d, m_oProj.w);
    auto d_attn_output = xt::reshape_view(d_attn_output_2d, {batch_size, seq_len, d_model});

    // Reshape to (batch, seq, n_heads, head_dim) for attention backward
    auto d_weighted_v = xt::reshape_view(d_attn_output, {batch_size, seq_len, static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});

    // === Step 2: Attention backward ===
    // d_V, d_Q_rope, d_K_rope from attention weights and gradients
    xt::xtensor<float, 4> d_vs = xt::zeros<float>({batch_size, seq_len, static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});
    xt::xtensor<float, 4> d_qs_rope = xt::zeros<float>({batch_size, seq_len, static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});
    xt::xtensor<float, 4> d_ks_rope = xt::zeros<float>({batch_size, seq_len, static_cast<size_t>(n_heads), static_cast<size_t>(head_dim)});

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Parallelize over (batch, head) pairs - each pair can be computed independently
    // d_qs_rope and d_ks_rope are accumulated per-position, so we parallelize over batch*head
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, batch_size, 0, n_heads),
        [&](const tbb::blocked_range2d<size_t>& r) {
            // Thread-local buffers to avoid repeated allocations
            std::vector<float> d_weights(seq_len);

            for (size_t b = r.rows().begin(); b != r.rows().end(); ++b) {
                for (size_t h = r.cols().begin(); h != r.cols().end(); ++h) {
                    for (size_t q_pos = 0; q_pos < seq_len; ++q_pos) {
                        // Compute d_weights (gradient w.r.t. attention weights before softmax backward)
                        float dot_sum = 0.0f;

                        for (size_t k_pos = 0; k_pos <= q_pos; ++k_pos) {
                            float dot = 0.0f;
                            for (size_t d = 0; d < head_dim; ++d) {
                                dot += d_weighted_v(b, q_pos, h, d) * m_act_vs(b, k_pos, h, d);
                            }
                            d_weights[k_pos] = dot;
                            dot_sum += dot * m_act_attention_weights(b, h, q_pos, k_pos);
                        }

                        // Softmax backward: d_scores = weights * (d_weights - dot_sum)
                        for (size_t k_pos = 0; k_pos <= q_pos; ++k_pos) {
                            float weight = m_act_attention_weights(b, h, q_pos, k_pos);
                            float d_score = weight * (d_weights[k_pos] - dot_sum);

                            // d_V[k_pos] += weights[q_pos, k_pos] * d_weighted_v[q_pos]
                            for (size_t d = 0; d < head_dim; ++d) {
                                d_vs(b, k_pos, h, d) += weight * d_weighted_v(b, q_pos, h, d);
                            }

                            // d_Q[q_pos] += d_score * K[k_pos] * scale
                            // d_K[k_pos] += d_score * Q[q_pos] * scale
                            for (size_t d = 0; d < head_dim; ++d) {
                                d_qs_rope(b, q_pos, h, d) += d_score * m_act_ks_with_rope(b, k_pos, h, d) * scale;
                                d_ks_rope(b, k_pos, h, d) += d_score * m_act_qs_with_rope(b, q_pos, h, d) * scale;
                            }
                        }
                    }
                }
            }
        }
    );

    // === Step 3: RoPE backward ===
    auto d_qs_normed = apply_rope_backward(d_qs_rope);
    auto d_ks_normed = apply_rope_backward(d_ks_rope);

    // === Step 4: RMSNorm backward (Q and K) ===
    // Reshape back to (batch, seq, d_model) for RMSNorm
    auto d_qs_normed_3d = xt::reshape_view(d_qs_normed, {batch_size, seq_len, d_model});
    auto d_ks_normed_3d = xt::reshape_view(d_ks_normed, {batch_size, seq_len, d_model});

    xt::xtensor<float, 3> d_projected_qs = m_qNorm.backward(d_qs_normed_3d);
    xt::xtensor<float, 3> d_projected_ks = m_kNorm.backward(d_ks_normed_3d);

    // === Step 5: Q/K/V projection backward ===
    xt::xtensor<float, 2> d_projected_qs_2d = xt::reshape_view(d_projected_qs, {batch_size * seq_len, d_model});
    xt::xtensor<float, 2> d_projected_ks_2d = xt::reshape_view(d_projected_ks, {batch_size * seq_len, d_model});
    xt::xtensor<float, 2> d_vs_2d = xt::reshape_view(d_vs, {batch_size * seq_len, d_model});

    // Accumulate weight gradients using cblas_sgemm with beta=1
    const int tokens = static_cast<int>(batch_size * seq_len);
    const int dm = static_cast<int>(d_model);

    if (m_qProj.grad.size() == 0)
        m_qProj.grad = xt::zeros_like(m_qProj.w);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        dm, dm, tokens,
        1.0f, d_projected_qs_2d.data(), dm,
        m_act_input_2d.data(), dm,
        1.0f, m_qProj.grad.data(), dm);

    if (m_kProj.grad.size() == 0)
        m_kProj.grad = xt::zeros_like(m_kProj.w);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        dm, dm, tokens,
        1.0f, d_projected_ks_2d.data(), dm,
        m_act_input_2d.data(), dm,
        1.0f, m_kProj.grad.data(), dm);

    if (m_vProj.grad.size() == 0)
        m_vProj.grad = xt::zeros_like(m_vProj.w);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        dm, dm, tokens,
        1.0f, d_vs_2d.data(), dm,
        m_act_input_2d.data(), dm,
        1.0f, m_vProj.grad.data(), dm);

    // Compute d_input from all three projections
    // d_input = d_Q @ W_Q + d_K @ W_K + d_V @ W_V
    xt::xtensor<float, 3> d_input = xt::empty<float>({batch_size, seq_len, d_model});

    const int M = static_cast<int>(batch_size * seq_len);
    const int N = static_cast<int>(d_model);
    const int K = static_cast<int>(d_model);

    // d_input = d_Q @ W_Q
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0f, d_projected_qs_2d.data(), K, m_qProj.w.data(), N,
        0.0f, d_input.data(), N
    );

    // d_input += d_K @ W_K
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0f, d_projected_ks_2d.data(), K, m_kProj.w.data(), N,
        1.0f, d_input.data(), N
    );

    // d_input += d_V @ W_V
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0f, d_vs_2d.data(), K, m_vProj.w.data(), N,
        1.0f, d_input.data(), N
    );

    return d_input;
}

void OlmoAttention::step(float lr) {
    m_qProj.w -= lr * m_qProj.grad;
    m_kProj.w -= lr * m_kProj.grad;
    m_vProj.w -= lr * m_vProj.grad;
    m_oProj.w -= lr * m_oProj.grad;
    m_qNorm.step(lr);
    m_kNorm.step(lr);
}

void OlmoAttention::zero_grad() {
    m_qProj.grad.fill(0.0f);
    m_kProj.grad.fill(0.0f);
    m_vProj.grad.fill(0.0f);
    m_oProj.grad.fill(0.0f);
    m_qNorm.zero_grad();
    m_kNorm.zero_grad();
}
