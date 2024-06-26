#include <cuda_runtime.h>

#include "cuda_utils.hpp"
#include "cuda_copy.hpp"

// kernel
template <typename T>
__global__ void custom_device_memcpy_kernel(T *output, const T *input, size_t n)
{
    size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
    if (idx < n)
    {
        output[idx] = input[idx];
    }
}

// launch
template <typename T>
void launch_custom_device_memcpy(T *output, const T *input, size_t n, cudaStream_t stream)
{
    dim3 block{1024U};
    dim3 grid{(static_cast<unsigned int>(n) + block.x - 1U) / block.x};
    custom_device_memcpy_kernel<T><<<grid, block, 0, stream>>>(output, input, n);
}

// explicit instantiation
template void launch_custom_device_memcpy<int8_t>(int8_t *output, const int8_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy<int16_t>(int16_t *output, const int16_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy<int32_t>(int32_t *output, const int32_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy<int64_t>(int64_t *output, const int64_t *input, size_t n, cudaStream_t stream);