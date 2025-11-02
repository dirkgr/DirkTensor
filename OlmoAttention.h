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

    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);

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

    static xt::xtensor<float, 4> apply_rope(const xt::xtensor<float, 4>& input);

    // Precomputed RoPE sin/cos tables (seq_len x head_dim) - shared across all instances
    static std::vector<float> s_rope_sin;
    static std::vector<float> s_rope_cos;
    static void init_rope_tables();
};
