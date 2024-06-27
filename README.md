## Overview
This project explores the concept of vectorized memory access, focusing on benchmarking different versions of copy operations on a device.
The main objective is to understand how vectorized memory access can improve performance compared to other methods.

## Build and Run
This repository includes a CMakeList.txt to automate the building and running of the CUDA programs. Simply run the following commands:

```!
cmake -B build
cmake --build build
./build/src/profile_copy
```

## Key takeaways
1. The more units of data we copy, the higher the bandwidth.
2. The larger the unit of the data, the higher the bandwidth.
3. Shared memory does not help with both the latency or the bandwidth.
4. Copying data in units of 8 bytes or 16 bytes can improve the latency of custom device memcpy.

## Credit

All credit (the code and the explanation) goes to [https://leimao.github.io/blog/CUDA-Vectorized-Memory-Access/]
