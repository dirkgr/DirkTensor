cmake_minimum_required(VERSION 3.14)  # FetchContent requires 3.14+
project(DirkTensor)

set(CMAKE_CXX_STANDARD 23)

# Suppress deprecation warnings from dependencies
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

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
FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG 13.0.0
)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_Declare(
    TBB
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG v2021.13.0
)

# Make the libraries available
FetchContent_MakeAvailable(xtl xsimd xtensor xtensor-blas googletest TBB)

add_executable(DirkTensor
    main.cpp
    xtutil.h
    model_config.h
    RMSNorm.h RMSNorm.cpp
    OlmoMlp.h OlmoMlp.cpp
    OlmoAttention.h OlmoAttention.cpp
    OlmoBlock.h OlmoBlock.cpp
    OlmoModel.h OlmoModel.cpp
    Detokenizer.h Detokenizer.cpp
)

# Link xtensor (header-only library)
target_link_libraries(DirkTensor xtensor xtl xsimd xtensor-blas TBB::tbb)

# Enable xsimd for xtensor
target_compile_definitions(DirkTensor PRIVATE XTENSOR_USE_XSIMD)
target_compile_definitions(DirkTensor PRIVATE XTENSOR_USE_TBB)

# Enable optimizations for release builds
target_compile_options(DirkTensor PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Test executable for OlmoModel
add_executable(test_olmo_model
    test_olmo_model.cpp
    model_config.h
    RMSNorm.h RMSNorm.cpp
    OlmoMlp.h OlmoMlp.cpp
    OlmoAttention.h OlmoAttention.cpp
    OlmoBlock.h OlmoBlock.cpp
    OlmoModel.h OlmoModel.cpp
    Detokenizer.h Detokenizer.cpp
)

target_link_libraries(test_olmo_model xtensor xtl xsimd xtensor-blas TBB::tbb GTest::gtest GTest::gtest_main)
target_compile_definitions(test_olmo_model PRIVATE XTENSOR_USE_XSIMD)
target_compile_definitions(test_olmo_model PRIVATE XTENSOR_USE_TBB)
target_compile_options(test_olmo_model PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Test executable for Detokenizer
add_executable(test_detokenizer
    test_detokenizer.cpp
    Detokenizer.h Detokenizer.cpp
)

target_link_libraries(test_detokenizer GTest::gtest GTest::gtest_main)
target_compile_options(test_detokenizer PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Find Python for running tests
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Add C++ test for OlmoModel
add_test(
    NAME test_olmo_model
    COMMAND test_olmo_model
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Add C++ test for Detokenizer
add_test(
    NAME test_detokenizer
    COMMAND test_detokenizer
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

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