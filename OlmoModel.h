#pragma once

#include <array>
#include <memory>
#include <string>
#include <xtensor/containers/xtensor.hpp>
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
