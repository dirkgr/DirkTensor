#include "OlmoAttention.h"

#include <cmath>
#include <format>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

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

    const auto projected_qs = xt::linalg::tensordot(input, m_qProj, {2}, {1});
    const auto normed_qs = m_qNorm.forward(projected_qs);
    const auto qs = xt::reshape_view(normed_qs, {batch_size, seq_len, n_heads, head_dim});

    const auto projected_ks = xt::linalg::tensordot(input, m_kProj, {2}, {1});
    const auto normed_ks = m_kNorm.forward(projected_ks);
    const auto ks = xt::reshape_view(normed_ks, {batch_size, seq_len, n_heads, head_dim});

    const auto projected_vs = xt::linalg::tensordot(input, m_vProj, {2}, {1});
    const auto vs = xt::eval(xt::reshape_view(projected_vs, {batch_size, seq_len, n_heads, head_dim}));

    // qs, ks, vs are all (batch_size, seq_len, n_heads, head_dim)

    // apply RoPE
    const auto qs_with_rope = xt::eval(apply_rope(qs));
    const auto ks_with_rope = xt::eval(apply_rope(ks));

    // attend
    auto attention_output = xt::zeros_like(input);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t position = 0; position < seq_len; ++position) {
            const auto q = xt::view(qs_with_rope, b, position, xt::newaxis(), xt::all());
            const auto k = xt::view(ks_with_rope, b, xt::range(0, position + 1)); // causal mask is in here
            const auto v = xt::view(vs, b, xt::range(0, position + 1));
            const auto logits = xt::sum(q * k, {2}) / std::sqrt(head_dim);

            auto softmax = xt::eval(xt::exp(logits));
            const auto exp_logits_sum = xt::eval(xt::sum(softmax, {0}));
            softmax /= exp_logits_sum;
            // softmax is (seq, n_heads)

            // apply weights to V
            const auto weighted_sums =
                xt::sum(v * xt::view(softmax, xt::all(), xt::all(), xt::newaxis()), {0});
            // weighted_sums is (n_heads, head_dim)

            xt::view(attention_output, b, position) +=
                xt::reshape_view(weighted_sums, {n_heads * head_dim});
        }
    }

    return xt::linalg::tensordot(attention_output, m_oProj, {2}, {1});
}


xt::xtensor<float, 4> OlmoAttention::apply_rope(const xt::xtensor<float, 4>& input) {
    // Input dimensions: (batch_size, seq_len, n_heads, head_dim)
    const auto batch_size = input.shape(0);
    const auto seq_len = input.shape(1);

    static const auto [pos_sin, pos_cos] = rope_buffers();
    // rope buffers are (seq_len, head_dim)

    auto output = xt::zeros_like(input);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t position = 0; position < seq_len; ++position) {
            for (size_t head = 0; head < n_heads; ++head) {
                // sin part first half
                xt::view(output, b, position, head, xt::range(0, head_dim / 2)) =
                    -xt::view(input, b, position, head, xt::range(head_dim / 2, head_dim));
                xt::view(output, b, position, head, xt::range(head_dim / 2, head_dim)) =
                    xt::view(input, b, position, head, xt::range(0, head_dim / 2));
                xt::view(output, b, position, head, xt::all()) *=
                    xt::view(pos_sin, position);

                // cos part
                xt::view(output, b, position, head, xt::all()) +=
                    xt::view(input, b, position, head, xt::all()) * \
                    xt::view(pos_cos, position);
            }
        }
    }
    return output;
}
