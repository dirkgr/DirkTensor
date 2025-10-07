#include "RMSNorm.h"

#include <xtensor/io/xnpy.hpp>

RMSNorm::RMSNorm(const std::string& filename) {
    m_weight = xt::load_npy<float>(filename);
}
