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
#include <cstdio>
#include <iterator>

xt::xtensor<uint32_t, 1> read_tokens(std::istream& input) {
    std::vector<uint32_t> token_vec;
    uint32_t token;

    // Read tokens until EOF
    while (input.read(reinterpret_cast<char*>(&token), sizeof(token))) {
        token_vec.push_back(token);
    }

    // Convert to xtensor
    xt::xtensor<uint32_t, 1> result = xt::empty<uint32_t>({token_vec.size()});
    std::copy(token_vec.begin(), token_vec.end(), result.begin());
    return result;
}

static const size_t hidden_dim = 2048;

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

    // Load OLMo 2 1B embeddings from .npy file
    auto embeddings = xt::load_npy<float>("models/OLMo-2-0425-1B/model.embed_tokens.weight.npy");

    // Verify shape
    assert(embeddings.shape().size() == 2);
    assert(embeddings.shape()[1] == hidden_dim);

    std::cout << "Loaded embeddings with shape: (" << embeddings.shape()[0] << ", " << embeddings.shape()[1] << ")" << std::endl;

    return 0;
}