#include <cuda_runtime.h>
#include <limits>

#include "cuda_copy.hpp"
#include "cuda_utils.hpp"

// kernel
template <typename T, size_t NUM_THREADS>
__global__ void custom_device_memcpy_shared_memory_kernel(T* __restrict__ output, const T* __restrict__ input, size_t n)
{
    size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
    const size_t GRID_SIZE{gridDim.x * blockDim.x};

    __shared__ T buffer[NUM_THREADS];

    for (size_t i{idx}; i < n; i+= GRID_SIZE)
    {
        buffer[threadIdx.x] = input[i];
        output[i] = buffer[threadIdx.x];
    }
}

// launch
template <typename T>
void launch_custom_device_memcpy_shared_memory(T *output, const T *input, size_t n, cudaStream_t stream)
{   
    constexpr size_t NUM_THREADS{1024U};
    dim3 block{NUM_THREADS};
    dim3 grid{
        static_cast<unsigned int>(std::min((static_cast<unsigned int>(n) + block.x - 1U) / block.x, std::numeric_limits<unsigned int>::max()))
    };
    custom_device_memcpy_shared_memory_kernel<T, NUM_THREADS><<<grid, block, 0, stream>>>(output, input, n);

    CHECK_LAST_CUDA_ERROR();
}

// explicit instantiation
template void launch_custom_device_memcpy_shared_memory<int8_t>(int8_t *output, const int8_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_shared_memory<int16_t>(int16_t *output, const int16_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_shared_memory<int32_t>(int32_t *output, const int32_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_shared_memory<int64_t>(int64_t *output, const int64_t *input, size_t n, cudaStream_t stream);