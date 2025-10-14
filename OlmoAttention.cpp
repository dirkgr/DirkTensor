#include "OlmoAttention.h"

#include <format>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

OlmoAttention::OlmoAttention(const std::string& folder, const unsigned int index) :
    m_qNorm(std::format("{}/model.layers.{}.self_attn.q_norm.weight.npy", folder, index)),
    m_kNorm(std::format("{}/model.layers.{}.self_attn.k_norm.weight.npy", folder, index))
{
    m_qProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.q_proj.weight.npy", folder, index));
    m_kProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.k_proj.weight.npy", folder, index));
    m_vProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.v_proj.weight.npy", folder, index));
    m_oProj = xt::load_npy<float>(std::format("{}/model.layers.{}.self_attn.o_proj.weight.npy", folder, index));

    // kv cache
    m_kCache = xt::empty<float>({seq_len, n_heads, head_dim});
    m_vCache = xt::empty<float>({seq_len, n_heads, head_dim});
}

xt::xtensor<float, 1> OlmoAttention::forward(const xt::xtensor<float, 1>& input) {
    const auto q = xt::reshape_view(m_qNorm.forward(xt::linalg::dot(m_qProj, input)), {n_heads, head_dim});
    const auto k = xt::reshape_view(m_kNorm.forward(xt::linalg::dot(m_kProj, input)), {n_heads, head_dim});
    const auto v = xt::reshape_view(xt::linalg::dot(m_vProj, input), {n_heads, head_dim});
    // q, k, v are all (n_heads, head_dim)

    // apply RoPE
    const auto q_with_rope = apply_rope(q, m_kvCacheEnd);
    const auto k_with_rope = apply_rope(k, m_kvCacheEnd);

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
    const auto exp_logits_sum = xt::sum(exp_logits, {0});
    const auto softmax = exp_logits / exp_logits_sum;
    // softmax is (seq, n_heads)

    // apply weights to V
    const auto weighted_sums = xt::sum(vs * xt::view(softmax, xt::all(), xt::all(), xt::newaxis()), {0});
    // weighted_sums is (n_heads, head_dim)
    const auto attention_output = xt::reshape_view(weighted_sums, {n_heads * head_dim});
    // attention_output is (d_model,)

    return xt::linalg::dot(m_oProj, attention_output);
}

xt::xtensor<float, 2> OlmoAttention::apply_rope(const xt::xtensor<float, 2>& input, size_t position) {
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
