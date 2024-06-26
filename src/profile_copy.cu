#include <iostream>
#include <cuda_runtime.h>
#include <type_traits>

#include "profile_utils.cuh"
#include "cuda_copy.hpp"

int main()
{   
    // print device information
    print_device_information();

    constexpr size_t num_repeats{10U};
    constexpr size_t num_warmups{10U};

    constexpr size_t tensor_size_small{1U * 64U * 64U * 64U};
    constexpr size_t tensor_size_medium{1U * 128U * 128U * 128U};
    constexpr size_t tensor_size_large{1U * 512U * 512U * 512U};

    // official device memcpy
    std:: cout << "=========================================================================" << std::endl;
    std::cout << "Official device memcpy" << std::endl;
    for (size_t size : {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::cout << "=========================================================================" << std::endl;
        std::cout << "Tensor size: " << size << " Units" << std::endl;

        profile_copy_kernel<int8_t>(size, launch_official_device_memcpy<int8_t>, num_warmups, num_repeats);
        profile_copy_kernel<int16_t>(size, launch_official_device_memcpy<int16_t>, num_warmups, num_repeats);
        profile_copy_kernel<int32_t>(size, launch_official_device_memcpy<int32_t>, num_warmups, num_repeats);
        profile_copy_kernel<int64_t>(size, launch_official_device_memcpy<int64_t>, num_warmups, num_repeats);
    }

    std::cout << std::endl;

    // custom device memcpy
    std:: cout << "=========================================================================" << std::endl;
    std::cout << "Custom device memcpy" << std::endl;
    for (size_t size : {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::cout << "=========================================================================" << std::endl;
        std::cout << "Tensor size: " << size << " Units" << std::endl;

        profile_copy_kernel<int8_t>(size, launch_custom_device_memcpy<int8_t>, num_warmups, num_repeats);
        profile_copy_kernel<int16_t>(size, launch_custom_device_memcpy<int16_t>, num_warmups, num_repeats);
        profile_copy_kernel<int32_t>(size, launch_custom_device_memcpy<int32_t>, num_warmups, num_repeats);
        profile_copy_kernel<int64_t>(size, launch_custom_device_memcpy<int64_t>, num_warmups, num_repeats);
    }
    
    std::cout << std::endl;

    // custom device memcpy shared memory
    std:: cout << "=========================================================================" << std::endl;
    std::cout << "Custom device memcpy shared memory" << std::endl;
    for (size_t size : {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::cout << "=========================================================================" << std::endl;
        std::cout << "Tensor size: " << size << " Units" << std::endl;

        profile_copy_kernel<int8_t>(size, launch_custom_device_memcpy_shared_memory<int8_t>, num_warmups, num_repeats);
        profile_copy_kernel<int16_t>(size, launch_custom_device_memcpy_shared_memory<int16_t>, num_warmups, num_repeats);
        profile_copy_kernel<int32_t>(size, launch_custom_device_memcpy_shared_memory<int32_t>, num_warmups, num_repeats);
        profile_copy_kernel<int64_t>(size, launch_custom_device_memcpy_shared_memory<int64_t>, num_warmups, num_repeats);
    }

    std::cout << std::endl;

    // custom device memcpy optimized using 8 bytes copy
    std:: cout << "=========================================================================" << std::endl;
    std::cout << "Custom device memcpy optimized" << std::endl;
    for (size_t size : {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::cout << "=========================================================================" << std::endl;
        std::cout << "Tensor size: " << size << " Units" << std::endl;

        profile_copy_kernel<int8_t>(size, launch_custom_device_memcpy_optimized<int8_t, uint64_t>, num_warmups, num_repeats);
        profile_copy_kernel<int16_t>(size, launch_custom_device_memcpy_optimized<int16_t, uint64_t>, num_warmups, num_repeats);
        profile_copy_kernel<int32_t>(size, launch_custom_device_memcpy_optimized<int32_t, uint64_t>, num_warmups, num_repeats);
        profile_copy_kernel<int64_t>(size, launch_custom_device_memcpy_optimized<int64_t, uint64_t>, num_warmups, num_repeats);
    }

    std::cout << std::endl;

    // custom device memcpy optimized using 16 bytes copy
    std::cout << "=========================================================================" << std::endl;
    std::cout << "Custom device memcpy optimized" << std::endl;
    for (size_t size : {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::cout << "=========================================================================" << std::endl;
        std::cout << "Tensor size: " << size << " Units" << std::endl;

        profile_copy_kernel<int8_t>(size, launch_custom_device_memcpy_optimized<int8_t, uint4>, num_warmups, num_repeats);
        profile_copy_kernel<int16_t>(size, launch_custom_device_memcpy_optimized<int16_t, uint4>, num_warmups, num_repeats);
        profile_copy_kernel<int32_t>(size, launch_custom_device_memcpy_optimized<int32_t, uint4>, num_warmups, num_repeats);
        profile_copy_kernel<int64_t>(size, launch_custom_device_memcpy_optimized<int64_t, uint4>, num_warmups, num_repeats);
    }   
    return 0;
}