#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

xt::xtensor<uint32_t, 1> read_tokens(std::istream& input) {
    uint32_t count;
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input) {
        throw std::runtime_error("Failed to read token count");
    }

    xt::xtensor<uint32_t, 1> tokens = xt::empty<uint32_t>({count});
    input.read(reinterpret_cast<char*>(tokens.data()), count * sizeof(uint32_t));
    if (!input) {
        throw std::runtime_error("Failed to read token IDs");
    }

    return tokens;
}

static const size_t hidden_dim = 2048;

int main(int argc, char* argv[]) {
    // Read tokens
    xt::xtensor<uint32_t, 1> tokens;
    if (argc > 1) {
        std::ifstream file(argv[1], std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + std::string(argv[1]));
        }
        tokens = read_tokens(file);
    } else {
        tokens = read_tokens(std::cin);
    }

    // Load OLMo 2 1B embeddings from .npy file
    auto embeddings = xt::load_npy<float>("models/OLMo-2-0425-1B/model.embed_tokens.weight.npy");

    // Verify shape
    assert(embeddings.shape().size() == 2);
    assert(embeddings.shape()[1] == hidden_dim);

    std::cout << "Loaded embeddings with shape: (" << embeddings.shape()[0] << ", " << embeddings.shape()[1] << ")" << std::endl;

    return 0;
}