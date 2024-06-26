# set cmake version
cmake_minimum_required(VERSION 3.20.0)

# set the project name
project(vectorized-memory-access-benchmark VERSION 0.0.1 LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# find include dir
find_path(CUDA_COPY_INCLUDE_DIR cuda_copy.hpp HINTS ${CMAKE_SOURCE_DIR}/include)

# add libraries
add_library(cuda_copy
    SHARED
    official_device_memcpy.cu
    custom_device_memcpy.cu
    custom_device_memcpy_shared_memory.cu
    custom_device_memcpy_optimized.cu
    profile_utils.cu
    cuda_utils.cu
)
# include directories for libraries
target_include_directories(cuda_copy PUBLIC ${CUDA_COPY_INCLUDE_DIR})
# set gpu properties
set_target_properties(cuda_copy PROPERTIES CUDA_ARCHITECTURES 70)

# add executables
add_executable(profile_copy profile_copy.cu)
# link libraries to executables
target_link_libraries(profile_copy cuda_copy)
# set gpu properties
set_target_properties(profile_copy PROPERTIES CUDA_ARCHITECTURES 70)