#pragma once

#include <string>
#include <vector>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>
#include "RMSNorm.h"
#include "model_config.h"

class OlmoAttention {
public:
    OlmoAttention(const std::string& folder, unsigned int index);

    xt::xtensor<float, 1> forward(const xt::xtensor<float, 1>& input);

private:
    // parameters
    xt::xtensor<float, 2> m_qProj;
    RMSNorm m_qNorm;
    xt::xtensor<float, 2> m_kProj;
    RMSNorm m_kNorm;
    xt::xtensor<float, 2> m_vProj;
    xt::xtensor<float, 2> m_oProj;

    // RoPE - manually optimized implementation
    static constexpr float theta = 500000;

    xt::xtensor<float, 2> apply_rope(const xt::xtensor<float, 2>& input, size_t position);

    // kv cache
    xt::xtensor<float, 3> m_kCache; // (seq_len, n_heads, head_dim)
    xt::xtensor<float, 3> m_vCache; // (seq_len, n_heads, head_dim)
    size_t m_kvCacheEnd = 0;

    // Precomputed RoPE sin/cos tables (seq_len x head_dim)
    std::vector<float> m_rope_sin;
    std::vector<float> m_rope_cos;
};
