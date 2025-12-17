#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include "RMSNorm.h"
#include "param.h"


class OlmoAttention {
public:
    OlmoAttention(const std::string& folder, unsigned int index);
    xt::xtensor<float, 3> forward(const xt::xtensor<float, 3>& input);
    xt::xtensor<float, 3> backward(const xt::xtensor<float, 3>& d_output);
    void step(float lr);
    void zero_grad();

private:
    // parameters
    param<2> m_qProj;
    RMSNorm m_qNorm;
    param<2> m_kProj;
    RMSNorm m_kNorm;
    param<2> m_vProj;
    param<2> m_oProj;

    // Saved activations for backward pass
    xt::xtensor<float, 2> m_act_input_2d;           // (batch*seq, d_model)
    xt::xtensor<float, 4> m_act_qs_before_rope;     // (batch, seq, n_heads, head_dim) - Q after norm, before RoPE
    xt::xtensor<float, 4> m_act_ks_before_rope;     // (batch, seq, n_heads, head_dim) - K after norm, before RoPE
    xt::xtensor<float, 4> m_act_qs_with_rope;       // (batch, seq, n_heads, head_dim)
    xt::xtensor<float, 4> m_act_ks_with_rope;       // (batch, seq, n_heads, head_dim)
    xt::xtensor<float, 4> m_act_vs;                 // (batch, seq, n_heads, head_dim)
    xt::xtensor<float, 4> m_act_attention_weights;  // (batch, n_heads, seq, seq) - softmax output
    xt::xtensor<float, 3> m_act_attention_output;   // (batch, seq, d_model)

    static xt::xtensor<float, 4> apply_rope(const xt::xtensor<float, 4>& input);
    static xt::xtensor<float, 4> apply_rope_backward(const xt::xtensor<float, 4>& d_output);
};
