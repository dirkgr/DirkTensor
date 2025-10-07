#pragma once

#include <string>
#include <array>
#include <memory>
#include <cstdint>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "OlmoBlock.h"
#include "RMSNorm.h"
#include "model_config.h"

class OlmoModel {
public:
    explicit OlmoModel(const std::string& folder);

    xt::xtensor<float, 1> forward(uint32_t token);

private:
    xt::xtensor<float, 2> m_embeddings;
    std::array<std::unique_ptr<OlmoBlock>, n_layers> m_blocks;
    RMSNorm m_norm;
    xt::xtensor<float, 2> m_lmHead;
};
