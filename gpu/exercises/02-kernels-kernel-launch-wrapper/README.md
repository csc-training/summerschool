## Exercise: Error reporting from kernel launch

If a kernel launch configuration parameter (number of threads/blocks or the amount of shared memory) is incorrect,
the kernel is not launched at all. It just silently fails. For many devices, 1024 is the maximum number of threads
per block, which is why the kernel failed to launch.

Kernel launches don't return anything, so you don't get an error code. But there are ways to catch errors.

HIP API has a function called
[`hipGetLastError()`](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/error_handling.html)[^1].
It returns the previous error code from any HIP API call, including kernel launches.

It's useful to wrap kernel launches with a function that checks for any errors from the kernel.
Then you get automatic error reporting from kernel launches. To add information about the location of the error,
it's also useful to wrap the function call in a macro that adds the preprocessor definitions
`__FILE__` and `__LINE__` to the call. Those expand to the filename and line of code, respectively, of the call.
This helps you find the erroneous kernel launch easier.

### Example

Instead of
```cpp
void main() {
    const float a = 1.0f;
    kernel<<<1, 1, 0, 0>>>(a);
}
```

we do
```cpp
void main() {
    const float a = 1.0f;
    LAUNCH_KERNEL(kernel, 1, 1, 0, 0, a);
}
```

with
```cpp
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

    // Next we launch the kernel with the given arguments
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
```

The arguments to the `LAUNCH_KERNEL` macro are
- the kernel to launch (examples: `foo`, `kernel_name`, `fill1D`)
- number of blocks per grid (examples: `128`, `dim3(32, 32, 1)`)
- number of threads per block (examples: `128`, `dim3(32, 32, 1)`)
- number of bytes of dynamic shared memory (examples: `0`, `128`, `32 * sizeof(float)`)
- stream to launch the kernel on (examples: `0`, `stream`)
- all the arguments to the kernel (examples: `a, 1.0, n, 16`, `n, d_arr`)

### Examples

```cpp
// Kernels
__global__ void foo();
__global__ void bar(int n);
__global__ void baz(int n, float *arr);

// Launching them

LAUNCH_KERNEL(foo, 1, 1, 0, 0);
// launch kernel 'foo' with
// - 1 block
// - 1 thread
// - 0 bytes of dynamic shared memory
// - on stream 0
// - and without any arguments

LAUNCH_KERNEL(bar, dim3(32, 1, 1), 128, 0, 0, 16);
// launch kernel 'bar' with
// - 32 blocks
// - 128 threads per block
// - 0 bytes of dynamic shared memory
// - on stream 0
// - argument n = 16

LAUNCH_KERNEL(baz, dim3(1, 1, 1), dim3(128, 1, 8), 0, 0, 16, d_arr);
// launch kernel 'baz' with
// - 1 block
// - 1024 threads per block (128 in x & 8 in z dimension)
// - 0 bytes of dynamic shared memory
// - on stream 0
// - argument n = 16, arr = d_arr
```

If you're confused by the C `__VA_ARGS__` macro and C++ `typename... Args` template functions, don't worry.
That is quite advanced c/c++ stuff that you can use even if you don't fully understand.
Think about the `printf` function in C. 
That is a variadic function that I bet all of you already used and none really went into the implementation details!
You don't need to understand it at this point, consider this as a ready made implementation which you can just use.
But, since we're here to learn, if you're interested, you can read more about [variadic macros](https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html)
 (`...` and `__VA_ARGS__`) in C
and [parameter packs](https://en.cppreference.com/w/cpp/language/parameter_pack.html)
(`typename... Args`, `Args... args` and `args...`) in C++.

Ok, enough exposition. In this exercise, your job is to wrap the kernel launch with the macro how it is described above.

Then, compile the code and run it with different block and thread sizes and observe the reported errors.



[^1]: Note for those who are somewhat familiar with Cuda: until ROCm 7.0 `hipGetLastError()`
works differently from `cudaGetLastError()`. `hipGetLastError()` returns the error code from any
HIP API, even if the call returned `hipSuccess`. This means `hipDeviceSynchronize()` call cannot
be put between the kernel launch and `hipGetLastError()`, or the return code from the kernel will
be overwritten by the success from the synchronization. With ROCm >= 7.0 and with Cuda,
you should synchronize with the device immediately after the kernel launch, then call `hipGetLastError()`
to get any errors that happen during the kernel execution, not just at the launch. Do note, that this
synchronization should only be done with debug builds, as it's a very expensive operation.
