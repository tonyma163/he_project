cmake_minimum_required(VERSION 3.28)

project(Testing LANGUAGES CXX CUDA VERSION 1.0)

# use C++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Python
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter)

# Phantom-fhe library
include_directories(/phantom)

# HEaaN
include_directories(../HEaaN)

# Execute the specified file
#add_executable(test main.cu)
add_executable(test heaan.cu)

# Link libraries
target_link_libraries(test /usr/local/lib/libPhantom.so)
target_link_libraries(test /usr/local/lib/libHEaaN.so)
target_include_directories(test PUBLIC ${CMAKE_SOURCE_DIR}/include)
