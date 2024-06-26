# set the required version of CMake
cmake_minimum_required(VERSION 3.20.0)

# set the project name
project(vectorized-memory-access-benchmark VERSION 0.0.1 LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add subdirectories
add_subdirectory(src)