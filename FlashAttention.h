#pragma once

#include <xtensor/containers/xtensor.hpp>

// Flash attention result with saved statistics for backward
struct FlashAttentionResult {
    xt::xtensor<float, 4> output;       // (batch, seq, n_heads, head_dim)
    xt::xtensor<float, 3> softmax_m;    // (batch, seq, n_heads) - max values for each row
    xt::xtensor<float, 3> softmax_l;    // (batch, seq, n_heads) - sum of exp for each row
};

// Flash attention forward pass
// Q, K, V: (batch, seq, n_heads, head_dim)
// Returns: output and saved softmax statistics for backward
FlashAttentionResult flash_attention_forward(
    const xt::xtensor<float, 4>& Q,
    const xt::xtensor<float, 4>& K,
    const xt::xtensor<float, 4>& V,
    float scale,
    bool causal = true
);

// Flash attention backward pass
// d_output: gradient w.r.t. output (batch, seq, n_heads, head_dim)
// Q, K, V: saved from forward pass (batch, seq, n_heads, head_dim)
// Returns: tuple of (d_Q, d_K, d_V) each (batch, seq, n_heads, head_dim)
struct FlashAttentionGradients {
    xt::xtensor<float, 4> d_Q;
    xt::xtensor<float, 4> d_K;
    xt::xtensor<float, 4> d_V;
};

FlashAttentionGradients flash_attention_backward(
    const xt::xtensor<float, 4>& d_output,
    const xt::xtensor<float, 4>& output,     // Forward pass output for D computation
    const xt::xtensor<float, 3>& softmax_m,  // Saved max values from forward
    const xt::xtensor<float, 3>& softmax_l,  // Saved sum-of-exp values from forward
    const xt::xtensor<float, 4>& Q,
    const xt::xtensor<float, 4>& K,
    const xt::xtensor<float, 4>& V,
    float scale,
    bool causal = true
);
