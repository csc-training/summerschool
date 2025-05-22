#pragma once

#include <hip/hip_runtime.h>

#define LAUNCH_KERNEL(kernel, ...) launch_kernel(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)

template <typename... Args>
void launch_kernel(const char *kernel_name, const char *file, int32_t line,
                   void (*kernel)(Args...), dim3 blocks, dim3 threads,
                   size_t num_bytes_shared_mem, hipStream_t stream,
                   Args... args) {
    // A call to hipGetLastError() resets the error variable to hipSuccess.
    // We call it before launching the kernel, so we don't get any false errors
    // from earlier API calls.
    auto result = hipGetLastError();

    // Next we launch the kernel with the given argument
    kernel<<<blocks, threads, num_bytes_shared_mem, stream>>>(args...);

    // Now we get the actual error returned by the kernel launch
    result = hipGetLastError();

    // If the result is something other that success, print out a message describing it
    // and stop the program.
    // Note the other two useful HIP API calls:
    // - hipGetErrorName() returns the name of the error, given a hipError_t value
    // - hipGetErrorString() returns a short description of the error, given a hipError_t value
    if (result != hipSuccess) {
        printf("Error with kernel \"%s\" in %s at line %d\n%s: %s\n",
               kernel_name, file, line, hipGetErrorName(result),
               hipGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
