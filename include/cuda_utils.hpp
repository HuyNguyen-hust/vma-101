// define all checking errors helper functions here
#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda_error(val, #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)

void check_cuda_error(cudaError_t err, const char *const func, const char *const file, const int line);
void check_last_cuda_error(const char *const file, const int line);