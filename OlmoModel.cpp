#include "OlmoModel.h"

#include <cassert>
#include <iostream>
#include <iomanip>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "xtutil.h"

OlmoModel::OlmoModel(const std::string& folder) : m_norm(folder + "/model.norm.weight.npy") {
    m_embeddings = xt::load_npy<float>(folder + "/model.embed_tokens.weight.npy");
    assert(m_embeddings.shape().size() == 2);
    assert(m_embeddings.shape()[1] == d_model);

    m_lmHead = xt::load_npy<float>(folder + "/lm_head.weight.npy");

    for(size_t i = 0; i < n_layers; i++)
        m_blocks[i] = std::make_unique<OlmoBlock>(folder, i);
}

xt::xtensor<float, 3> OlmoModel::forward(const xt::xtensor<uint32_t, 2>& batch) {
    // Embedding
    xt::xtensor<float, 3> x = xt::empty<float>({
        batch.shape(0),
        batch.shape(1),
        m_embeddings.shape(1)
    });
    for (size_t b = 0; b < batch.shape(0); b++) {
        for (size_t i = 0; i < batch.shape(1); i++) {
            xt::noalias(xt::view(x, b, i)) = xt::view(m_embeddings, batch(b, i));
        }
    }

    // Blocks
    for(size_t i = 0; i < n_layers; i++)
        x = m_blocks[i]->forward(x);

    // Print profiling data
    std::cerr << "\n" << std::string(60, '=') << std::endl;
    std::cerr << "PROFILING RESULTS (milliseconds per component)" << std::endl;
    std::cerr << std::string(60, '=') << std::endl;

    double total_attention = 0.0, total_norm1 = 0.0, total_mlp = 0.0, total_norm2 = 0.0;

    for(size_t i = 0; i < n_layers; i++) {
        std::cerr << "Layer " << std::setw(2) << i << ": "
                  << "Attn: " << std::fixed << std::setprecision(1) << std::setw(7)
                  << m_blocks[i]->attention_time_ms << " ms, "
                  << "Norm1: " << std::setw(6) << m_blocks[i]->norm1_time_ms << " ms, "
                  << "MLP: " << std::setw(7) << m_blocks[i]->mlp_time_ms << " ms, "
                  << "Norm2: " << std::setw(6) << m_blocks[i]->norm2_time_ms << " ms"
                  << std::endl;

        total_attention += m_blocks[i]->attention_time_ms;
        total_norm1 += m_blocks[i]->norm1_time_ms;
        total_mlp += m_blocks[i]->mlp_time_ms;
        total_norm2 += m_blocks[i]->norm2_time_ms;
    }

    std::cerr << std::string(60, '-') << std::endl;
    std::cerr << "TOTALS: "
              << "Attn: " << std::setw(7) << total_attention << " ms ("
              << std::setw(5) << std::setprecision(1)
              << (total_attention / (total_attention + total_norm1 + total_mlp + total_norm2) * 100) << "%), "
              << "MLP: " << std::setw(7) << total_mlp << " ms ("
              << std::setw(5) << (total_mlp / (total_attention + total_norm1 + total_mlp + total_norm2) * 100) << "%), "
              << "Norms: " << std::setw(7) << (total_norm1 + total_norm2) << " ms ("
              << std::setw(5) << ((total_norm1 + total_norm2) / (total_attention + total_norm1 + total_mlp + total_norm2) * 100) << "%)"
              << std::endl;
    std::cerr << std::string(60, '=') << "\n" << std::endl;

    // Norm
    x = m_norm.forward(x);

    // LM Head
    return xt::linalg::tensordot(x, m_lmHead, {2}, {1});
}
