#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char* argv[]) {
    static const size_t min_array_size = 16 * 1024;
    static const size_t max_array_size = 16 * 1024 * 1024;
    static const unsigned int min_ops = 1;
    static const unsigned int max_ops = 64;

    for (size_t array_size = min_array_size; array_size <= max_array_size; array_size *= 2) {
        std::vector<float> input_array(array_size);
        std::generate(input_array.begin(), input_array.end(), std::rand);

        for (unsigned int ops = min_ops; ops <= max_ops; ops += 1) {
            std::time_t start = std::clock();
            for (size_t i = 0; i < array_size; i += 1) {
                float v = input_array[i];
                for (unsigned int op = 0; op < ops; op += 1) {
                    v *= 0.999999;
                }
                input_array[i] = v;
            }
            std::time_t end = std::clock();

            std::cout << array_size << '\t' << ops << '\t' << end - start << std::endl;
        }
    }
}
