#pragma once

#include <string>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
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

    // RoPE
    static constexpr float theta = 500000;
    static constexpr auto rope_buffers() {
        const auto inv_freq =
            1.0 / (xt::pow(theta, xt::arange<double>(0, head_dim, 2) / head_dim));
        const auto seq = xt::arange<double>(0, seq_len);
        const auto freqs =
            xt::view(seq, xt::all(), xt::newaxis()) * xt::view(inv_freq, xt::newaxis(), xt::all());
        const auto positions = xt::eval(xt::concatenate(std::tuple(freqs, freqs), 1));
        const auto pos_sin = xt::sin(positions);
        const auto pos_cos = xt::cos(positions);

        const xt::xtensor<float, 2> pos_sin_f = xt::cast<float>(pos_sin);
        const xt::xtensor<float, 2> pos_cos_f = xt::cast<float>(pos_cos);

        return std::pair(pos_sin_f, pos_cos_f);
        // rope buffers are (seq_len, head_dim)
    }

    static xt::xtensor<float, 2> apply_rope(const xt::xtensor<float, 2>& input, size_t position);

    // kv cache
    xt::xtensor<float, 3> m_kCache; // (seq_len, n_heads, head_dim)
    xt::xtensor<float, 3> m_vCache; // (seq_len, n_heads, head_dim)
    size_t m_kvCacheEnd = 0;
};
