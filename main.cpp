#include <xtensor/xarray.hpp>
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
#include <memory>

std::vector<uint64_t> parse_tensor_shape(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read number of dimensions
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

    // Read each dimension size
    std::vector<uint64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
    }

    return shape;
}

float* mmap_tensor_data(const std::string& filename, const std::vector<uint64_t>& shape) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Calculate header size: 4 bytes for ndim + 8 bytes per dimension
    const size_t header_size = sizeof(uint32_t) + shape.size() * sizeof(uint64_t);

    // Calculate total tensor elements
    uint64_t total_elements = 1;
    for (uint64_t dim : shape) {
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

int main() {
    // Load OLMo 2 1B
    const auto embeddings_shape = parse_tensor_shape("models/OLMo-2-0425-1B/model.embed_tokens.weight.bin");
    assert(embeddings_shape.size() == 2);
    assert(embeddings_shape[0] == 100352);
    assert(embeddings_shape[1] == 2048);
    float* embeddings_data = mmap_tensor_data("models/OLMo-2-0425-1B/model.embed_tokens.weight.bin", embeddings_shape);
    
    // Create xtensor array from existing data (copy to owned memory for simplicity)
    std::vector<std::size_t> xt_shape = {embeddings_shape[0], embeddings_shape[1]};
    std::size_t total_elements = embeddings_shape[0] * embeddings_shape[1];
    std::vector<float> data_copy(embeddings_data, embeddings_data + total_elements);
    auto embeddings = xt::adapt(data_copy, xt_shape);

    // Print first few elements of first row of embeddings
    std::cout << "First 10 elements of first row: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << embeddings(0, i) << " ";
    }
    std::cout << std::endl;

    return 0;
}