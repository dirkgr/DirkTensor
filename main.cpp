#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

std::vector<std::size_t> parse_tensor_shape(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read number of dimensions
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

    // Read each dimension size
    std::vector<std::size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint64_t dim_size;
        file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
        shape[i] = static_cast<std::size_t>(dim_size);
    }

    return shape;
}

float* mmap_tensor_data(const std::string& filename, const std::vector<std::size_t>& shape) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Calculate header size: 4 bytes for ndim + 8 bytes per dimension
    const size_t header_size = sizeof(uint32_t) + shape.size() * sizeof(uint64_t);

    // Calculate total tensor elements
    std::size_t total_elements = 1;
    for (std::size_t dim : shape) {
        total_elements *= dim;
    }

    // Calculate data size (float32 = 4 bytes per element)
    const size_t data_size = total_elements * sizeof(float);

    // Memory-map from beginning and offset pointer manually
    // (mmap offset must be page-aligned)
    const size_t total_size = header_size + data_size;
    void* mapped = mmap(nullptr, total_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Memory mapping failed for: " + filename);
    }
    const auto data = reinterpret_cast<float*>(static_cast<char*>(mapped) + header_size);

    close(fd);
    return data;
}

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

    // Load OLMo 2 1B embeddings
    const auto embeddings_shape = parse_tensor_shape("models/OLMo-2-0425-1B/model.embed_tokens.weight.bin");
    assert(embeddings_shape.size() == 2);
    assert(embeddings_shape[1] == hidden_dim);
    float* embeddings_data = mmap_tensor_data("models/OLMo-2-0425-1B/model.embed_tokens.weight.bin", embeddings_shape);

    // Create xtensor array from memory-mapped data (non-owning)
    // Use xtensor with explicit type specification to avoid ambiguity
    auto embeddings = xt::adapt(const_cast<const float*>(embeddings_data), embeddings_shape);

    std::cout << "Loaded embeddings with shape: (" << embeddings_shape[0] << ", " << embeddings_shape[1] << ")" << std::endl;

    return 0;
}