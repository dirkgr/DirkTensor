#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <iterator>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "xtutil.h"


xt::xtensor<uint32_t, 1> read_tokens(std::istream& input) {
    std::vector<uint32_t> token_vec;
    uint32_t token;

    // Read tokens until EOF
    while (input.read(reinterpret_cast<char*>(&token), sizeof(token))) {
        token_vec.push_back(token);
    }

    // Convert to xtensor
    xt::xtensor<uint32_t, 1> result = xt::empty<uint32_t>({token_vec.size()});
    std::ranges::copy(token_vec, result.begin());
    return result;
}


static constexpr unsigned int head_dim = 128;
static constexpr unsigned int seq_len = 4096;
static constexpr unsigned int d_model = 2048;
static constexpr unsigned int n_heads = d_model / head_dim;
static constexpr unsigned int n_layers = 16;


class RMSNorm {
public:
    static constexpr float eps = 1e-06;

    RMSNorm(const std::string& filename) {
        m_weight = xt::load_npy<float>(filename);
    }

    xt::xtensor<float, 1> forward(const auto& input) {
        const auto rms = xt::sqrt(xt::mean(xt::square(input)) + eps);
        const auto x = input / rms;
        return x * m_weight;
    }

private:
    xt::xtensor<float, 1> m_weight;
};


class OlmoMlp {
public:
    OlmoMlp(const std::string& folder, const unsigned int index) {
        m_upProjection =
            xt::load_npy<float>(std::format("{}/model.layers.{}.mlp.up_proj.weight.npy", folder, index));
        m_gateProjection =
            xt::load_npy<float>(std::format("{}/model.layers.{}.mlp.gate_proj.weight.npy", folder, index));
        m_downProjection =
            xt::load_npy<float>(std::format("{}/model.layers.{}.mlp.down_proj.weight.npy", folder, index));
    }

    xt::xtensor<float, 1> forward(const auto& input) {
        const auto projected = xt::linalg::dot(input, m_upProjection);
        const auto gate = xt::linalg::dot(input, m_gateProjection);
        const auto sigmoid = 1.0 / (1.0 + xt::exp(-gate));
        const auto silu = gate * sigmoid;
        const auto result = projected * silu;
        return xt::linalg::dot(result, m_downProjection);
    }

private:
    xt::xtensor<float, 2> m_upProjection;
    xt::xtensor<float, 2> m_gateProjection;
    xt::xtensor<float, 2> m_downProjection;
};


class OlmoAttention {
public:
    OlmoAttention(const std::string& folder, const unsigned int index) :
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
        const auto weighted_sums = xt::sum(vs * softmax, {0});
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


class OlmoBlock {
public:
    OlmoBlock(const std::string& folder, const unsigned int index) :
        m_attention(folder, index),
        m_postAttentionNorm(std::format("{}/model.layers.{}.post_attention_layernorm.weight.npy", folder, index)),
        m_mlp(folder, index),
        m_postMlpNorm(std::format("{}/model.layers.{}.post_feedforward_layernorm.weight.npy", folder, index))
    {
        // nothing to do
    }

    xt::xtensor<float, 1> forward(const auto& input) {
        const auto after_attention = m_attention.forward(input);
        const auto normed_after_attention = m_postAttentionNorm.forward(after_attention);
        const auto h = input + normed_after_attention;

        const auto after_mlp = m_mlp.forward(h);
        const auto normed_after_mlp = m_postMlpNorm.forward(after_mlp);
        return h + normed_after_mlp;
    }

private:
    OlmoAttention m_attention;
    RMSNorm m_postAttentionNorm;
    OlmoMlp m_mlp;
    RMSNorm m_postMlpNorm;
};


class OlmoModel {
public:
    explicit OlmoModel(const std::string& folder) : m_norm(folder + "/model.norm.weight.npy") {
        m_embeddings = xt::load_npy<float>(folder + "/model.embed_tokens.weight.npy");
        assert(m_embeddings.shape().size() == 2);
        assert(m_embeddings.shape()[1] == d_model);

        m_lmHead = xt::load_npy<float>(folder + "/lm_head.weight.npy");

        for(size_t i = 0; i < n_layers; i++)
            m_blocks[i] = std::make_unique<OlmoBlock>(folder, i);
    }

    auto forward(const uint32_t token) {
        std::cout << "Processing token: " << token << std::endl;

        // Embedding
        auto x = xt::view(m_embeddings, token, xt::all());

        // Blocks
        for(size_t i = 0; i < n_layers; i++)
            x = m_blocks[i]->forward(x);

        // Norm
        x = m_norm.forward(x);

        // LM Head
        return xt::linalg::dot(x, xt::transpose(m_lmHead));
    }

private:
    xt::xtensor<float, 2> m_embeddings;
    std::array<std::unique_ptr<OlmoBlock>, n_layers> m_blocks;
    RMSNorm m_norm;
    xt::xtensor<float, 2> m_lmHead;
};


int main(int argc, char* argv[]) {
    // Read tokens from binary stream
    xt::xtensor<uint32_t, 1> tokens;
    if (argc > 1) {
        // Read from file
        std::ifstream file(argv[1], std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + std::string(argv[1]));
        }
        tokens = read_tokens(file);
    } else {
        // Read from stdin
        tokens = read_tokens(std::cin);
    }

    std::cout << "Read " << tokens.size() << " tokens" << std::endl;

    OlmoModel model("models/OLMo-2-0425-1B");
    const auto probs = model.forward(tokens(0));
    std::cout << probs << std::endl;

    return 0;
}