#include "FlashAttention.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#ifdef WITH_VECLIB
#include <Accelerate/Accelerate.h>
#elif defined(WITH_OPENBLAS) || defined(WITH_MKLBLAS)
#include <cblas.h>
#endif

#include "model_config.h"

// Block sizes for flash attention
// Trade-off: Larger blocks = fewer matmul calls but more cache pressure
// Using larger blocks to reduce BLAS call overhead
static constexpr size_t BLOCK_SIZE_Q = 256;   // Br: query block size
static constexpr size_t BLOCK_SIZE_KV = 256;  // Bc: key/value block size

FlashAttentionResult flash_attention_forward(
    const xt::xtensor<float, 4>& Q,
    const xt::xtensor<float, 4>& K,
    const xt::xtensor<float, 4>& V,
    float scale,
    bool causal
) {
    const size_t batch_size = Q.shape(0);
    const size_t seq_len = Q.shape(1);
    const size_t num_heads = Q.shape(2);
    const size_t head_dim_local = Q.shape(3);

    // Output tensors
    FlashAttentionResult result;
    result.output = xt::zeros<float>({batch_size, seq_len, num_heads, head_dim_local});
    result.softmax_m = xt::zeros<float>({batch_size, seq_len, num_heads});
    result.softmax_l = xt::zeros<float>({batch_size, seq_len, num_heads});

    // Parallelize over (batch, head) pairs
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, batch_size * num_heads),
        [&](const tbb::blocked_range<size_t>& r) {
            // Allocate per-thread buffers for block computations
            std::vector<float> S_block(BLOCK_SIZE_Q * BLOCK_SIZE_KV);  // Block attention scores
            std::vector<float> P_block(BLOCK_SIZE_Q * BLOCK_SIZE_KV);  // Block attention weights (unnormalized)
            std::vector<float> PV_block(BLOCK_SIZE_Q * head_dim_local); // Block output contribution

            // Running statistics per Q row (within a Q block)
            std::vector<float> m(BLOCK_SIZE_Q);      // Running max
            std::vector<float> l(BLOCK_SIZE_Q);      // Running sum of exp
            std::vector<float> O(BLOCK_SIZE_Q * head_dim_local);  // Running unnormalized output
            std::vector<float> m_local(BLOCK_SIZE_Q);
            std::vector<float> l_local(BLOCK_SIZE_Q);

            for (size_t bh = r.begin(); bh != r.end(); ++bh) {
                const size_t b = bh / num_heads;
                const size_t h = bh % num_heads;

                // Process Q in blocks
                for (size_t q_start = 0; q_start < seq_len; q_start += BLOCK_SIZE_Q) {
                    const size_t Br = std::min(BLOCK_SIZE_Q, seq_len - q_start);

                    // Initialize running statistics for this Q block
                    std::fill(m.begin(), m.begin() + Br, -std::numeric_limits<float>::infinity());
                    std::fill(l.begin(), l.begin() + Br, 0.0f);
                    std::fill(O.begin(), O.begin() + Br * head_dim_local, 0.0f);

                    // Iterate over K/V blocks
                    for (size_t k_start = 0; k_start < seq_len; k_start += BLOCK_SIZE_KV) {
                        const size_t Bc = std::min(BLOCK_SIZE_KV, seq_len - k_start);

                        // Skip fully masked blocks (causal attention)
                        if (causal && q_start < k_start) {
                            // All positions in this Q block are before this K block
                            // So all would be masked - skip
                            continue;
                        }

                        // Step 1: Compute S_block = Q_block @ K_block^T * scale
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<int>(Br), static_cast<int>(Bc), static_cast<int>(head_dim_local),
                            scale,
                            &Q(b, q_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            &K(b, k_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            0.0f,
                            S_block.data(), static_cast<int>(BLOCK_SIZE_KV)
                        );

                        // Step 2: Apply causal mask, compute local max and unnormalized P
                        for (size_t i = 0; i < Br; ++i) {
                            m_local[i] = -std::numeric_limits<float>::infinity();

                            // Find local max (with causal mask)
                            for (size_t j = 0; j < Bc; ++j) {
                                size_t q_pos = q_start + i;
                                size_t k_pos = k_start + j;

                                if (causal && q_pos < k_pos) {
                                    S_block[i * BLOCK_SIZE_KV + j] = -std::numeric_limits<float>::infinity();
                                } else {
                                    m_local[i] = std::max(m_local[i], S_block[i * BLOCK_SIZE_KV + j]);
                                }
                            }

                            // Compute unnormalized P and local sum
                            l_local[i] = 0.0f;
                            for (size_t j = 0; j < Bc; ++j) {
                                float s = S_block[i * BLOCK_SIZE_KV + j];
                                if (s > -std::numeric_limits<float>::infinity()) {
                                    P_block[i * BLOCK_SIZE_KV + j] = std::exp(s - m_local[i]);
                                    l_local[i] += P_block[i * BLOCK_SIZE_KV + j];
                                } else {
                                    P_block[i * BLOCK_SIZE_KV + j] = 0.0f;
                                }
                            }
                        }

                        // Step 3: Compute PV_block = P_block @ V_block (unnormalized contribution)
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(Br), static_cast<int>(head_dim_local), static_cast<int>(Bc),
                            1.0f,
                            P_block.data(), static_cast<int>(BLOCK_SIZE_KV),
                            &V(b, k_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            0.0f,
                            PV_block.data(), static_cast<int>(head_dim_local)
                        );

                        // Step 4: Update running statistics with rescaling
                        for (size_t i = 0; i < Br; ++i) {
                            // Compute new max
                            float m_new = std::max(m[i], m_local[i]);

                            // Compute rescaling factors
                            float alpha = (m[i] > -std::numeric_limits<float>::infinity())
                                ? std::exp(m[i] - m_new) : 0.0f;
                            float beta = (m_local[i] > -std::numeric_limits<float>::infinity())
                                ? std::exp(m_local[i] - m_new) : 0.0f;

                            // Update running sum: l_new = l * alpha + l_local * beta
                            l[i] = l[i] * alpha + l_local[i] * beta;

                            // Update running output: O = O * alpha + PV_block * beta
                            // (O stays UNNORMALIZED - we divide by l only at the end!)
                            for (size_t d = 0; d < head_dim_local; ++d) {
                                O[i * head_dim_local + d] = O[i * head_dim_local + d] * alpha
                                    + PV_block[i * head_dim_local + d] * beta;
                            }

                            // Update running max
                            m[i] = m_new;
                        }
                    }

                    // Step 5: Final normalization, copy to output, and save softmax stats
                    for (size_t i = 0; i < Br; ++i) {
                        // Save softmax statistics for backward pass
                        result.softmax_m(b, q_start + i, h) = m[i];
                        result.softmax_l(b, q_start + i, h) = l[i];

                        if (l[i] > 0.0f) {
                            for (size_t d = 0; d < head_dim_local; ++d) {
                                result.output(b, q_start + i, h, d) = O[i * head_dim_local + d] / l[i];
                            }
                        } else {
                            // No valid attention (shouldn't happen with proper input)
                            for (size_t d = 0; d < head_dim_local; ++d) {
                                result.output(b, q_start + i, h, d) = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    );

    return result;
}


FlashAttentionGradients flash_attention_backward(
    const xt::xtensor<float, 4>& d_output,
    const xt::xtensor<float, 4>& output,     // Forward pass output for D computation
    const xt::xtensor<float, 3>& softmax_m,  // Saved max values from forward
    const xt::xtensor<float, 3>& softmax_l,  // Saved sum-of-exp values from forward
    const xt::xtensor<float, 4>& Q,
    const xt::xtensor<float, 4>& K,
    const xt::xtensor<float, 4>& V,
    float scale,
    bool causal
) {
    const size_t batch_size = Q.shape(0);
    const size_t seq_len = Q.shape(1);
    const size_t num_heads = Q.shape(2);
    const size_t head_dim_local = Q.shape(3);

    // Output gradients
    FlashAttentionGradients grads;
    grads.d_Q = xt::zeros<float>({batch_size, seq_len, num_heads, head_dim_local});
    grads.d_K = xt::zeros<float>({batch_size, seq_len, num_heads, head_dim_local});
    grads.d_V = xt::zeros<float>({batch_size, seq_len, num_heads, head_dim_local});

    // Parallelize over (batch, head) pairs
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, batch_size * num_heads),
        [&](const tbb::blocked_range<size_t>& r) {
            // Per-thread buffers
            std::vector<float> S_block(BLOCK_SIZE_Q * BLOCK_SIZE_KV);
            std::vector<float> P_block(BLOCK_SIZE_Q * BLOCK_SIZE_KV);
            std::vector<float> dP_block(BLOCK_SIZE_Q * BLOCK_SIZE_KV);
            std::vector<float> dS_block(BLOCK_SIZE_Q * BLOCK_SIZE_KV);

            // D[i] = O[i] · d_output[i] - computed directly from forward output
            std::vector<float> D(BLOCK_SIZE_Q);

            // Temporary gradient accumulators
            std::vector<float> d_Q_block(BLOCK_SIZE_Q * head_dim_local);
            std::vector<float> d_K_block(BLOCK_SIZE_KV * head_dim_local);
            std::vector<float> d_V_block(BLOCK_SIZE_KV * head_dim_local);

            for (size_t bh = r.begin(); bh != r.end(); ++bh) {
                const size_t b = bh / num_heads;
                const size_t h = bh % num_heads;

                // Process Q in blocks
                for (size_t q_start = 0; q_start < seq_len; q_start += BLOCK_SIZE_Q) {
                    const size_t Br = std::min(BLOCK_SIZE_Q, seq_len - q_start);

                    // === Compute D[i] = O[i] · d_output[i] directly (no iteration over K blocks!) ===
                    for (size_t i = 0; i < Br; ++i) {
                        D[i] = 0.0f;
                        for (size_t dd = 0; dd < head_dim_local; ++dd) {
                            D[i] += output(b, q_start + i, h, dd) * d_output(b, q_start + i, h, dd);
                        }
                    }

                    // === Single pass: Compute gradients using saved softmax stats ===
                    std::fill(d_Q_block.begin(), d_Q_block.begin() + Br * head_dim_local, 0.0f);

                    for (size_t k_start = 0; k_start < seq_len; k_start += BLOCK_SIZE_KV) {
                        const size_t Bc = std::min(BLOCK_SIZE_KV, seq_len - k_start);

                        if (causal && q_start < k_start) continue;

                        // Recompute S_block and P_block
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<int>(Br), static_cast<int>(Bc), static_cast<int>(head_dim_local),
                            scale,
                            &Q(b, q_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            &K(b, k_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            0.0f,
                            S_block.data(), static_cast<int>(BLOCK_SIZE_KV)
                        );

                        // Compute P_block using saved softmax statistics
                        for (size_t i = 0; i < Br; ++i) {
                            const float m_i = softmax_m(b, q_start + i, h);
                            const float l_i = softmax_l(b, q_start + i, h);
                            for (size_t j = 0; j < Bc; ++j) {
                                size_t q_pos = q_start + i;
                                size_t k_pos = k_start + j;

                                if (causal && q_pos < k_pos) {
                                    P_block[i * BLOCK_SIZE_KV + j] = 0.0f;
                                } else {
                                    float s = S_block[i * BLOCK_SIZE_KV + j];
                                    P_block[i * BLOCK_SIZE_KV + j] = std::exp(s - m_i) / l_i;
                                }
                            }
                        }

                        // Compute dP_block = d_output @ V^T
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<int>(Br), static_cast<int>(Bc), static_cast<int>(head_dim_local),
                            1.0f,
                            &d_output(b, q_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            &V(b, k_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            0.0f,
                            dP_block.data(), static_cast<int>(BLOCK_SIZE_KV)
                        );

                        // Compute dS_block = P * (dP - D) * scale
                        for (size_t i = 0; i < Br; ++i) {
                            for (size_t j = 0; j < Bc; ++j) {
                                dS_block[i * BLOCK_SIZE_KV + j] =
                                    P_block[i * BLOCK_SIZE_KV + j] *
                                    (dP_block[i * BLOCK_SIZE_KV + j] - D[i]) * scale;
                            }
                        }

                        // d_V += P^T @ d_output
                        std::fill(d_V_block.begin(), d_V_block.begin() + Bc * head_dim_local, 0.0f);
                        cblas_sgemm(
                            CblasRowMajor, CblasTrans, CblasNoTrans,
                            static_cast<int>(Bc), static_cast<int>(head_dim_local), static_cast<int>(Br),
                            1.0f,
                            P_block.data(), static_cast<int>(BLOCK_SIZE_KV),
                            &d_output(b, q_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            0.0f,
                            d_V_block.data(), static_cast<int>(head_dim_local)
                        );

                        // Accumulate d_V
                        for (size_t j = 0; j < Bc; ++j) {
                            for (size_t dd = 0; dd < head_dim_local; ++dd) {
                                grads.d_V(b, k_start + j, h, dd) += d_V_block[j * head_dim_local + dd];
                            }
                        }

                        // d_Q += dS @ K
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(Br), static_cast<int>(head_dim_local), static_cast<int>(Bc),
                            1.0f,
                            dS_block.data(), static_cast<int>(BLOCK_SIZE_KV),
                            &K(b, k_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            1.0f,  // Accumulate
                            d_Q_block.data(), static_cast<int>(head_dim_local)
                        );

                        // d_K += dS^T @ Q
                        std::fill(d_K_block.begin(), d_K_block.begin() + Bc * head_dim_local, 0.0f);
                        cblas_sgemm(
                            CblasRowMajor, CblasTrans, CblasNoTrans,
                            static_cast<int>(Bc), static_cast<int>(head_dim_local), static_cast<int>(Br),
                            1.0f,
                            dS_block.data(), static_cast<int>(BLOCK_SIZE_KV),
                            &Q(b, q_start, h, 0), static_cast<int>(num_heads * head_dim_local),
                            0.0f,
                            d_K_block.data(), static_cast<int>(head_dim_local)
                        );

                        // Accumulate d_K
                        for (size_t j = 0; j < Bc; ++j) {
                            for (size_t dd = 0; dd < head_dim_local; ++dd) {
                                grads.d_K(b, k_start + j, h, dd) += d_K_block[j * head_dim_local + dd];
                            }
                        }
                    }

                    // Copy d_Q_block to output
                    for (size_t i = 0; i < Br; ++i) {
                        for (size_t dd = 0; dd < head_dim_local; ++dd) {
                            grads.d_Q(b, q_start + i, h, dd) = d_Q_block[i * head_dim_local + dd];
                        }
                    }
                }
            }
        }
    );

    return grads;
}
