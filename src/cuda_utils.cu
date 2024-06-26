#include <cuda_runtime.h>
#include <iostream>

#include "cuda_utils.hpp"

void check_cuda_error(cudaError_t err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_last_cuda_error(const char *const file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}