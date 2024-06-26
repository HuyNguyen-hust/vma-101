#include <cuda_runtime.h>
#include <type_traits>

#include "cuda_utils.hpp"
#include "cuda_copy.hpp"

// launch
template <typename T>
void launch_official_device_memcpy(T *output, const T *input, size_t n, cudaStream_t stream)
{
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output, input, n * sizeof(T), cudaMemcpyDeviceToDevice, stream));
}

// explicit instantiation

template void launch_official_device_memcpy<int8_t>(int8_t *output, const int8_t *input, size_t n, cudaStream_t stream);
template void launch_official_device_memcpy<int16_t>(int16_t *output, const int16_t *input, size_t n, cudaStream_t stream);
template void launch_official_device_memcpy<int32_t>(int32_t *output, const int32_t *input, size_t n, cudaStream_t stream);
template void launch_official_device_memcpy<int64_t>(int64_t *output, const int64_t *input, size_t n, cudaStream_t stream);