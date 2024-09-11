cmake_minimum_required(VERSION 3.28)

project(Testing LANGUAGES CXX CUDA VERSION 1.0)

# use C++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(/phantom)

#
add_executable(test test.cu)

target_link_libraries(test /usr/local/lib/libPhantom.so)
target_include_directories(test PUBLIC ${CMAKE_SOURCE_DIR}/include)

