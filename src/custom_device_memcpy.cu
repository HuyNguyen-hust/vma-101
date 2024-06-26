#include <cuda_runtime.h>
#include <limits>

#include "cuda_utils.hpp"
#include "cuda_copy.hpp"

// kernel
template <typename T>
__global__ void custom_device_memcpy_kernel(T* __restrict__ output, const T* __restrict__ input, size_t n)
{
    size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
    const size_t GRID_SIZE{gridDim.x * blockDim.x};
    for (size_t i{idx}; i < n; i+= GRID_SIZE)
    {
        output[i] = input[i];
    }
}

// launch
template <typename T>
void launch_custom_device_memcpy(T *output, const T *input, size_t n, cudaStream_t stream)
{
    dim3 block{1024U};
    dim3 grid{
        static_cast<unsigned int>(std::min((static_cast<unsigned int>(n) + block.x - 1U) / block.x, std::numeric_limits<unsigned int>::max()))
    };
    custom_device_memcpy_kernel<T><<<grid, block, 0, stream>>>(output, input, n);

    CHECK_LAST_CUDA_ERROR();
}

// explicit instantiation
template void launch_custom_device_memcpy<int8_t>(int8_t *output, const int8_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy<int16_t>(int16_t *output, const int16_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy<int32_t>(int32_t *output, const int32_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy<int64_t>(int64_t *output, const int64_t *input, size_t n, cudaStream_t stream);