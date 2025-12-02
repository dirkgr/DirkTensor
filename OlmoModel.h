#pragma once

#include <array>
#include <memory>
#include <string>
#include <xtensor/containers/xtensor.hpp>
#include "OlmoBlock.h"
#include "RMSNorm.h"
#include "model_config.h"
#include "param.h"


class OlmoModel {
public:
    explicit OlmoModel(const std::string& folder);

    xt::xtensor<float, 3> forward(const xt::xtensor<uint32_t, 2>& batch) const;

private:
    param<2> m_embeddings;
    std::array<std::unique_ptr<OlmoBlock>, n_layers> m_blocks;
    RMSNorm m_norm;
    param<2> m_lmHead;
};
