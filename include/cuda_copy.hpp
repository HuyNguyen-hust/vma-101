// define all vma launch functions here
#pragma once

#include <cuda_runtime.h>

template <typename T>
void launch_official_device_memcpy(T *output, const T *input, size_t n, cudaStream_t stream);

template <typename T>
void launch_custom_device_memcpy(T *output, const T *input, size_t n, cudaStream_t stream);

template <typename T>
void launch_custom_device_memcpy_shared_memory(T *output, const T *input, size_t n, cudaStream_t stream);

template <typename T>
void launch_custom_device_memcpy_optimized(T *output, const T *input, size_t n, cudaStream_t stream);