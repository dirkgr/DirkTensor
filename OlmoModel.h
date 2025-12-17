#pragma once

#include <array>
#include <memory>
#include <string>
#include <xtensor/containers/xtensor.hpp>
#include "OlmoBlock.h"
#include "OlmoLmHead.h"
#include "RMSNorm.h"
#include "model_config.h"
#include "param.h"


class OlmoModel {
public:
    explicit OlmoModel(const std::string& folder);

    xt::xtensor<float, 3> forward(const xt::xtensor<uint32_t, 2>& batch);
    void backward(const xt::xtensor<float, 3>& d_output);
    void step(float lr);
    void zero_grad();

private:
    // parameters
    param<2> m_embeddings;
    std::array<std::unique_ptr<OlmoBlock>, n_layers> m_blocks;
    RMSNorm m_norm;
    OlmoLmHead m_lmHead;

    // saved for backward
    xt::xtensor<uint32_t, 2> m_batch;
};
