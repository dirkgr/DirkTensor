cmake_minimum_required(VERSION 3.14)  # FetchContent requires 3.14+
project(DirkTensor)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)

# Enable testing
enable_testing()

# Fetch xtl (required dependency for xtensor)
FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG 0.8.0
)
FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG 0.27.0
)
FetchContent_Declare(
    xtensor-blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
    GIT_TAG 0.23.0
)

# Make the libraries available
FetchContent_MakeAvailable(xtl xtensor xtensor-blas)

add_executable(DirkTensor
    main.cpp
    xtutil.h
    model_config.h
    RMSNorm.h RMSNorm.cpp
    OlmoMlp.h OlmoMlp.cpp
    OlmoAttention.h OlmoAttention.cpp
    OlmoBlock.h OlmoBlock.cpp
    OlmoModel.h OlmoModel.cpp
)

# Link xtensor (header-only library)
target_link_libraries(DirkTensor xtensor xtl xtensor-blas)

# Enable optimizations for release builds
target_compile_options(DirkTensor PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Find Python for running tests
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Add Python tests using pytest
add_test(
    NAME test_tokenize_text
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/test_tokenize_text.py -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_test(
    NAME test_convert_checkpoint
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/test_convert_checkpoint.py -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Set test properties to show output
set_tests_properties(test_tokenize_text test_convert_checkpoint
    PROPERTIES
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
)