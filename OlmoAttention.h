#pragma once

#include <string>
#include <cmath>
#include <cassert>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "RMSNorm.h"
#include "model_config.h"

class OlmoAttention {
public:
    OlmoAttention(const std::string& folder, unsigned int index);

    xt::xtensor<float, 1> forward(const auto& input) {
        const auto q = xt::reshape_view(m_qNorm.forward(xt::linalg::dot(input, m_qProj)), {n_heads, head_dim});
        const auto k = xt::reshape_view(m_kNorm.forward(xt::linalg::dot(input, m_kProj)), {n_heads, head_dim});
        const auto v = xt::reshape_view(xt::linalg::dot(input, m_vProj), {n_heads, head_dim});
        // q, k, v are all (n_heads, head_dim)

        // apply RoPE
        const auto k_with_rope = apply_rope(q, m_kvCacheEnd);
        const auto q_with_rope = apply_rope(k, m_kvCacheEnd);

        // put into cache
        xt::view(m_kCache, m_kvCacheEnd) = k_with_rope;
        xt::view(m_vCache, m_kvCacheEnd) = v;
        m_kvCacheEnd += 1;

        const auto ks = xt::view(m_kCache, xt::range(0, m_kvCacheEnd));
        const auto vs = xt::view(m_vCache, xt::range(0, m_kvCacheEnd));
        // ks and vs are (seq, n_heads, head_dim)

        // attend
        const auto logits = xt::sum(xt::view(q_with_rope, xt::newaxis(), xt::all()) * ks, {2}) / std::sqrt(head_dim);
        // logits are (seq, n_heads)

        // softmax
        const auto exp_logits = xt::exp(logits);
        const auto exp_logits_sum = xt::sum(exp_logits, {1});
        const auto softmax = exp_logits / exp_logits_sum;
        // softmax is (seq, n_heads)

        // apply weights to V
        const auto weighted_sums = xt::sum(vs * xt::view(softmax, xt::all(), xt::all(), xt::newaxis()), {0});
        // weighted_sums is (n_heads, head_dim)
        const auto attention_output = xt::reshape_view(weighted_sums, {n_heads * head_dim});
        // attention_output is (d_model,)

        return xt::linalg::dot(attention_output, m_oProj);
    }

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
        const auto positions = xt::concatenate(std::tuple(freqs, freqs), 1);
        const auto pos_sin = xt::sin(positions);
        const auto pos_cos = xt::cos(positions);

        const xt::xtensor<float, 2> pos_sin_f = xt::cast<float>(pos_sin);
        const xt::xtensor<float, 2> pos_cos_f = xt::cast<float>(pos_cos);

        return std::tuple(pos_sin_f, pos_cos_f);
        // rope buffers are (seq_len, head_dim)
    }

    xt::xtensor<float, 2> apply_rope(const auto& input, const size_t position) const {
        // input dimensions are (n_heads, head_dim)

        static const auto [pos_sin, pos_cos] = rope_buffers();

        const auto cos_part = input * xt::view(pos_cos, position, xt::newaxis(), xt::all());

        // rotate input around the head dimension
        // Cool how we're using the word "rotate" to mean two totally different things here.
        const auto rotated_input_first_half = xt::view(
            input,
            xt::all(),
            xt::range(0, head_dim / 2));
        const auto rotated_input_second_half = xt::view(
            input,
            xt::all(),
            xt::range(head_dim / 2, head_dim));
        const auto rotated_input =
            xt::concatenate(std::tuple(-rotated_input_second_half, rotated_input_first_half), 1);
        assert (rotated_input.shape() == input.shape());

        const auto sin_part = rotated_input * xt::view(pos_sin, position, xt::newaxis(), xt::all());

        return cos_part + sin_part;
    }

    // kv cache
    xt::xtensor<float, 3> m_kCache; // (seq_len, n_heads, head_dim)
    xt::xtensor<float, 3> m_vCache; // (seq_len, n_heads, head_dim)
    size_t m_kvCacheEnd = 0;
};
