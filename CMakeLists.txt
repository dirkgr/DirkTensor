cmake_minimum_required(VERSION 3.14)  # FetchContent requires 3.14+
project(DirkTensor)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)

# Fetch xtl (required dependency for xtensor)
FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG 0.7.7  # Use latest stable version
)

# Fetch xtensor
FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG 0.25.0  # Use latest stable version
)

# Make the libraries available
FetchContent_MakeAvailable(xtl xtensor)

add_executable(DirkTensor main.cpp)

# Link xtensor (header-only library)
target_link_libraries(DirkTensor xtensor xtl)

# Enable optimizations for release builds
target_compile_options(DirkTensor PRIVATE 
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
)