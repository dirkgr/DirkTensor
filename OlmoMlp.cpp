#include "OlmoMlp.h"

#include <format>
#include <xtensor/io/xnpy.hpp>

OlmoMlp::OlmoMlp(const std::string& folder, const unsigned int index) {
    m_upProjection =
        xt::transpose(
            xt::load_npy<float>(
                std::format("{}/model.layers.{}.mlp.up_proj.weight.npy", folder, index)));
    m_gateProjection =
        xt::transpose(
            xt::load_npy<float>(
                std::format("{}/model.layers.{}.mlp.gate_proj.weight.npy", folder, index)));
    m_downProjection =
        xt::transpose(
            xt::load_npy<float>(
                std::format("{}/model.layers.{}.mlp.down_proj.weight.npy", folder, index)));
}
