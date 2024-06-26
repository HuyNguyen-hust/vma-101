#include <cuda_runtime.h>

#include "cuda_utils.hpp"
#include "profile_utils.cuh"

float measure_performance(
    std::function<void(cudaStream_t)> copy_launch_func, cudaStream_t stream, size_t num_repeats, size_t num_warmups
)
{
    // initialization
    float time;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warmup
    for (size_t i{0U}; i < num_warmups; ++i)
    {
        // launch copy
        copy_launch_func(stream);
    }

    // synchronize after warmup
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // measure
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0U}; i < num_repeats; ++i)
    {
        // launch copy
        copy_launch_func(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));

    // synchronize after measure
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_LAST_CUDA_ERROR();

    // get time
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));

    // destroy event
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return time / num_repeats;
}

void print_device_information()
{
    // choose device
    int device_id{0};
    cudaGetDevice(&device_id);
    // get device properties
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    // print device name
    std::cout << "=========================================================================" << std::endl;
    std::cout << "Device name: " << device_prop.name << std::endl;

    // print global memory size in GB
    float global_memory_size{static_cast<float>(device_prop.totalGlobalMem) / (1 << 30)};
    std::cout << "Global memory size: " << global_memory_size << " GB" << std::endl;

    // print peak memory bandwidth in GB/s
    float peak_memory_bandwidth{static_cast<float>((2.0f * device_prop.memoryClockRate * device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak memory bandwidth: " << peak_memory_bandwidth << " GB/s" << std::endl;

    std::cout << "=========================================================================" << std::endl;
    std::cout << std::endl;
}