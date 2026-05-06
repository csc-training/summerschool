
## Exercise: Better error reporting by querying limits

Ok, so you got an error report from the API. But it's not very helpful is it?
It tells you *something* went wrong, but not *what*. If you give incorrect arguments for threads and blocks,
you get the exact same error, even if you fix one of them. With some manual work we can do much better.

Different devices might have different hardware limits. We can query these limits from the API with a function call
[`hipDeviceGetAttribute`](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html#_CPPv421hipDeviceGetAttributePi20hipDeviceAttribute_ti).

A few useful attributes we're interested in at this point are
- Maximum number of threads per block in each dimension
    - `hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimX`
    - `hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimY`
    - `hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimZ`
- Maximum number of threads per block in total, i.e. `x * y * z`
    - `hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerBlock`
- Maximum number of blocks per grid in each dimension
    - `hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimX`
    - `hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimY`
    - `hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimZ`
- Maximum amount of shared memory in bytes
    - `hipDeviceAttribute_t::hipDeviceAttributeMaxSharedMemoryPerBlock`

To query information from the API, we first choose the device we wish to query the information from,
then we call `hipDeviceGetAttribute`:
```cpp
// Get the current device
int32_t device = 0;
auto result = hipGetDevice(&device);

// Get the value for attribute from device
int32_t value = 0;
result = hipDeviceGetAttribute(&value, hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimX, device);
```

Once we have the value for the attribute from the device, we can use it to check that the value used to launch the kernel
is not larger that the maximum:
```cpp
if (threads.x > value) {
    // The x dimension of the threads per block is larger that the limit for this device
    // We should print out a helpful message and exit
}
```

Since we're going to do a few attribute queries, it's helpful to make a little lambda out of it
(lambdas are pretty similar to functions):
```cpp
int32_t device = 0;
const auto result = hipGetDevice(&device);

// Helper lambda for querying device attributes
auto get_device_attribute = [&device](hipDeviceAttribute_t attribute) {
    int32_t value = 0;
    const auto result = hipDeviceGetAttribute(&value, attribute, device);
    return value;
};
```

This way it's easy to query different attributes:
```cpp
const dim3 max_threads(
    get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimX),
    get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimY),
    get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimZ));
```

A full list of attributes can be found in the [HIP runtime API documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html#ggacc0acd7b9bda126c6bb3dfd6e2796d7ca1421bb450fe736fda9605a607be69836).


---------------------------------------------------

One final note before we move to coding. Error checking is always useful,but it's often not efficient.
Usually the release versions of codes should run as fast as possible,while debug versions should be helpful for debugging.
Since we've wrapped our kernel calls with a debug macro, we might be worried about possible performance hits.

Luckily, there's a very easy fix for this: wrap the debug code in a `#if` preprocessor directive!
One common way to do it is to check for `NDEBUG`, which is commonly defined in release builds and **not** defined in debug builds:
```cpp
template <typename... Args>
void launch_kernel(const char *kernel_name, const char *file, int32_t line,
                   void (*kernel)(Args...), dim3 blocks, dim3 threads,
                   size_t num_bytes_shared_mem, hipStream_t stream,
                   Args... args) {
#if !NDEBUG
    // All the pre launch debug checks go here!
#endif

    // We launch the kernel regardless of the build type
    kernel<<<blocks, threads, num_bytes_shared_mem, stream>>>(args...);

#if !NDEBUG
    // All the post launch debug checks go here!
#endif
}
```

Alternatively, you may use a custom define and pass it to the compiler manually: `CC -DMY_CUSTOM_DEFINE ...`, `#if MY_CUSTOM_DEFINE ...`.

Ok, let's get coding! 
Start by filling in the blanks in the [error checking file](error_checking.hpp).

Then, compile the code, run it and fix any errors you get.
Repeat, until you get a message confirming you fixed the kernel launch parameters.

Note: This time the program doesn't take any input.
