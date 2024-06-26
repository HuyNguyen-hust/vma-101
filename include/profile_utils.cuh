// define all profiling helper functions here
#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <limits>
#include <cmath>
#include <iomanip>

#include "cuda_utils.hpp"

float measure_performance(
    std::function<void(cudaStream_t)> copy_launch_func, cudaStream_t stream, size_t num_repeats, size_t num_warmups
);

void print_device_information();

template <typename T,
    typename std::enable_if<std::is_integral<T>::value, bool>::type = true
>
void initialize_buffer(T* buffer, size_t n)
{
    for (size_t i{0U}; i < n; ++i)
    {
        buffer[i] = static_cast<T>(i) % static_cast<T>(std::numeric_limits<T>::max());
    }
}

template <typename T,
    typename std::enable_if<std::is_integral<T>::value, bool>::type = true
>
void initialize_zero_buffer(T* buffer, size_t n)
{
    for (size_t i{0U}; i < n; ++i)
    {
        buffer[i] = static_cast<T>(0);
    }
}

template <typename T,
    typename std::enable_if<std::is_integral<T>::value, bool>::type = true
>
void verify_buffer(T* buffer, size_t n)
{
    for (size_t i{0U}; i < n; ++i)
    {
        if (buffer[i] != static_cast<T>(i) % static_cast<T>(std::numeric_limits<T>::max()))
        {
            std::cerr << "Buffer verification failed at index " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

template <typename T>
float profile_copy_kernel(
    size_t n,
    std::function<void(T*, const T*, size_t, cudaStream_t)> copy_launch_func,
    size_t num_warmups,
    size_t num_repeats
)
{
    // create cuda stream
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "Unit size: " << sizeof(T) << "  bytes" << std::endl;
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // initialize buffer
    T* input{static_cast<T*>(malloc(n * sizeof(T)))};
    T* output{static_cast<T*>(malloc(n * sizeof(T)))};
    T* input_device{nullptr};
    T* output_device{nullptr};

    initialize_buffer<T>(input, n);
    initialize_zero_buffer<T>(output, n);

    CHECK_CUDA_ERROR(cudaMalloc(&input_device, n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&output_device, n * sizeof(T)));

    // verify the copy launch function
    CHECK_CUDA_ERROR(cudaMemcpyAsync(input_device, input, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output_device, output, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    copy_launch_func(output_device, input_device, n, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output, output_device, n * sizeof(T), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    verify_buffer<T>(output, n);

    // measure
    size_t total_bytes{n * sizeof(T)};
    float total_gbs{static_cast<float>(total_bytes) / (1 << 30)};

    const float latency {
        measure_performance(
            [&](cudaStream_t stream) {
                copy_launch_func(output_device, input_device, n, stream);
                return;
            },
            stream, num_repeats, num_warmups
        )
    };

    std::cout << std::fixed << std::setprecision(3) << "total gbs: " << total_gbs << " GBs" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "latency: " << latency << " ms" << std::endl;
    std::cout << "effective bandwidth: " << 2.0f * total_gbs / (latency / 1e3) << " GB/s" << std::endl;

    // free memory
    CHECK_CUDA_ERROR(cudaFree(input_device));
    CHECK_CUDA_ERROR(cudaFree(output_device));
    free(input);
    free(output);

    // destroy cuda stream
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    // compare with peak memory bandwidth
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    float peak_memory_bandwidth{static_cast<float>((2.0f * device_prop.memoryClockRate * device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "custom copy bandwidth vs. peak memory bandwidth: " <<
        (2.0f * total_gbs / (latency / 1e3)) / peak_memory_bandwidth * 100 << " %" << std::endl;

    return latency;
}