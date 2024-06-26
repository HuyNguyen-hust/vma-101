#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

#include "cuda_utils.hpp"
#include "cuda_copy.hpp"

// kernel
template <typename T, typename R>
__global__ void custom_device_memcpy_optimized_kernel(T* __restrict__ output, const T* __restrict__ input, size_t n)
{
    size_t idx{threadIdx.x + blockIdx.x * blockDim.x};
    const size_t GRID_SIZE{gridDim.x * blockDim.x};

    for (size_t i{idx}; i * sizeof(R) / sizeof(T) < n; i+= GRID_SIZE)
    {
        if (i * sizeof(R) / sizeof(T) < n)
        {
            reinterpret_cast<R*>(output)[i] = reinterpret_cast<const R*>(input)[i];
        }
        else
        {
            size_t remainder{n - i * sizeof(R) / sizeof(T)};
            for (size_t j{0}; j < remainder; ++j)
            {
                output[i * sizeof(R) / sizeof(T) + j] = input[i * sizeof(R) / sizeof(T) + j];
            }
        }
    }
}

// launch
template <typename T, typename R = uint64_t>
void launch_custom_device_memcpy_optimized(T *output, const T *input, size_t n, cudaStream_t stream)
{   
    size_t const num_units_to_copy = (n * sizeof(T) + sizeof(R) - 1U) / sizeof(R);
    dim3 block{1024U};
    dim3 grid{
        static_cast<unsigned int>(std::min((static_cast<unsigned int>(num_units_to_copy) + block.x - 1U) / block.x, std::numeric_limits<unsigned int>::max()))
    };
    custom_device_memcpy_optimized_kernel<T, R><<<grid, block, 0, stream>>>(output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

// explicit instantiation
template void launch_custom_device_memcpy_optimized<int8_t, uint64_t>(int8_t *output, const int8_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_optimized<int16_t, uint64_t>(int16_t *output, const int16_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_optimized<int32_t, uint64_t>(int32_t *output, const int32_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_optimized<int64_t, uint64_t>(int64_t *output, const int64_t *input, size_t n, cudaStream_t stream); 
template void launch_custom_device_memcpy_optimized<int8_t, uint4>(int8_t *output, const int8_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_optimized<int16_t, uint4>(int16_t *output, const int16_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_optimized<int32_t, uint4>(int32_t *output, const int32_t *input, size_t n, cudaStream_t stream);
template void launch_custom_device_memcpy_optimized<int64_t, uint4>(int64_t *output, const int64_t *input, size_t n, cudaStream_t stream);