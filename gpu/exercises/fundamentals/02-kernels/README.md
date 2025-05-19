# Kernel exercises

Hello! Welcome to the first of many GPU exercises!

We're going to start with simple programs and learn some fundamental concepts.

You should follow this document along and complete any exercises you meet, then return back to this document.

The exercises will be in separate directories, and you will be pointed to them.

Let's get started!

## Exercise: Compiling HIP code

The first exercise teaches you how to compile HIP code.

You can head over to [01_hello_api](01_hello_api) and follow the [instructions](01_hello_api/README.md) there.
Once you've successfully compiled and run the code, come back to this document and we can continue.

## Exercise: Launching a kernel

Now you know how to compile HIP code. Great!

Next, let's launch a kernel running on the GPU.

The kernel in [02_kernel_launch](02_kernel_launch) prints some values and does assertions.
If you're not familiar with the concept of 'assert', it's a function that takes in a boolean value
and aborts the program if the value is false. On the CPU it's similar to

```cpp
void assert(bool value) {
    if (!value) {
        abort();
    }
}
```
Assertions are useful for catching programmer errors in debug builds: if a value is not something you expect,
it immediately aborts the program.

Ok, enough about assertions.

You should head over to [02_kernel_launch](02_kernel_launch)
and follow the [instructions](02_kernel_launch/README.md) there, then come back here.

------------------------------------------------------------

If you followed the instructions, you should've seen some prints, some asserts, and possibly something
surprising.

Did you launch the program with 1025 threads? What happened?
To find out more, let's move on to the next exercise.

## Exercise: Error reporting from kernel launch

If a kernel launch configuration parameter (number of threads/blocks or the amount of shared memory) is incorrect,
the kernel is not launched at all. It just silently fails. For many devices, 1024 is the maximum number of threads
per block, which is why the kernel failed to launch.

Kernel launches don't return anything, so you don't get an error code. But there are ways to catch errors.

HIP API has a function called
[`hipGetLastError()`](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/error_handling.html)[^1].
It returns the previous error code from any HIP API call, including kernel launches.

It's useful to wrap kernel launches with a function that checks for any errors from the kernel.
They you get automatic error reporting from kernel launches. To add information about the location of the error,
it's also useful to wrap the function call in a macro that adds the preprocessor definitions `__FILE__` and `__LINE__`
to the call. Those expand to the filename and line of code, respectively, of the call. This helps you find the
erroneous kernel launch easier.

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
// launch kernel 'bar' with
// - 1 block
// - 1024 threads per block (128 in x & 8 in z dimension)
// - 0 bytes of dynamic shared memory
// - on stream 0
// - argument n = 16, arr = d_arr
```

If you're confused by the C macro and C++ template function, don't worry.
You don't need to understand it at this point, there's a ready made implementation which you can just use.

If you're interested, you can read more about [variadic macros](https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html)
 (`...` and `__VA_ARGS__`) in C
and [parameter packs](https://en.cppreference.com/w/cpp/language/pack)
(`typename... Args`, `Args... args` and `args...`) in C++.

Ok, enough exposition. Head over to the [next exercise](03_kernel_launch_wrapper) and follow the
[instructions](03_kernel_launch_wrapper/README.md) there.


### TODO

Make an exercise for this.
It can be the same code as before, but we `#include` a header file that has already implemented the thing above.
The task of the student is to
- wrap the kernel launch with the macro
- run the code with different thread/block sizes

The segway to the next exercise: The error reporting isn't very useful. It's very generic. You can use the API for more.

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

Since we're going to do a few attribute queries, it's helpful to make a little lambda out of it:
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

A full list of attributes can be found in the [HIP runtime API documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/global_defines_enums_structs_files/global_enum_and_defines.html#_CPPv420hipDeviceAttribute_t).

Ok, again, enough with the talk, let's get coding! Head over to [the next exercise](04_api_queries) and follow the [instructions](04_api_queries/README.md) there.

---------------------------------------------------

One final note before we move to the next exercise. Error checking is always useful, but it's also often not so efficient.
Usually the release versions of codes want to run as fast as possible, while debug versions want to be helpful for just that, debugging.
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

Ok, let's move on.

### TODO

Make the exercise.
Provide a skeleton of the finished kernel launch macro/function.
Clearly annotate parts that the student should fill in.
Hardcode the dim3 sizes in the main file to better showcase the problem (the same error with different problems vs manual query & print).

## Exercise: Errors from API calls

In the previous exercise we learned to query values from the API: `const auto result = hipDeviceGetAttribute(&value, attribute, device);`.
The API gives us the value for the attribute we query through a reference: `&value`. Why doesn't it just return it?
Because it returns something else, and we chose to call it `result`.

Many of the API calls actually return an error code of type [`hipError_t`](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/global_defines_enums_structs_files/global_enum_and_defines.html#group___global_defs_1ga6742b54e2b83c1a5d6861ede4825aafe).

We already encountered this earlier when we discussed catching errors
from kernel launches in a [previous exercise](03_kernel_launch_wrapper).

It's very useful to catch errors from the API calls.
This can save you *a lot* of debugging time later on.

So how to do it? It's very similar to the way we did it at kernel launch:
```cpp
auto result = hipDeviceGetAttribute(&value, attribute, device);
if (result != hipSuccess) {
    printf("Error in %s at line %d\n%s: %s\n",
           __FILE__,
           __LINE__,
           hipGetErrorName(result),
           hipGetErrorString(result));
    exit(EXIT_FAILURE);
}
```

This is one way to do it, but it can get tedious when you call the API multiple times.
Also, it makes the source code very annoying to read, when the important lines are hidden
in the error checking clutter:
```cpp
result = hipDeviceGetAttribute(&value, attribute1, device);
if (result != hipSuccess) {
    printf("Error in %s at line %d\n%s: %s\n",
           __FILE__,
           __LINE__,
           hipGetErrorName(result),
           hipGetErrorString(result));
    exit(EXIT_FAILURE);
}

result = hipDeviceGetAttribute(&value, attribute2, device);
if (result != hipSuccess) {
    printf("Error in %s at line %d\n%s: %s\n",
           __FILE__,
           __LINE__,
           hipGetErrorName(result),
           hipGetErrorString(result));
    exit(EXIT_FAILURE);
}

result = hipDeviceGetAttribute(&value, attribute3, device);
if (result != hipSuccess) {
    printf("Error in %s at line %d\n%s: %s\n",
           __FILE__,
           __LINE__,
           hipGetErrorName(result),
           hipGetErrorString(result));
    exit(EXIT_FAILURE);
}
```

Due to these reasons, it's recommended and very common to handle the error checking with a function,
and, like with kernel calls, wrap the API call with a macro:
```cpp
HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute1, device));
HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute2, device));
HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute3, device));
```

with
```cpp
#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)

static inline void hip_errchk(hipError_t result, const char *file,
                              int32_t line) {
    if (result != hipSuccess) {
        printf("Error in %s at line %d\n%s: %s\n",
            file,
            line,
            hipGetErrorName(result),
            hipGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
```

This way the error checking is out of the way, but it's still there.

In the [next exercise](05_api_errors) there are a bunch of problems with API calls.
Check the [instructions](05_api_errors/README.md) and fix the errors!

### TODO

The exercise should have some API calls that cause errors. The user should then fix those calls,
first by wrapping the calls with the macro, then by figuring out the meaning of the message.

## Exercise: Kernel for filling a 1D array with a value

So far we've mostly concerned ourselves with checking for errors. Can we finally do something more interesting?
Yes!

Many times in simulation loops we want to reset some arrays to a specific value before other computation.
So let's do that! Implement a kernel that fills an array with a single value.

To do that we need a few things:
- an array of values on the GPU memory
- a value which the elements of the array will be set to
- a kernel that does that

We also need to launch the kernel with enough threads to go over the entire array.
But we've learned that the maximum number of threads per block is 1024. Yes, indeed,
but the limit on the maximum number of blocks per grid is much higher! So we should be able to
easily launch enough *threads per grid* to fill the entire array.

Head over to the [next exercise](06_fill) to figure out the rest of the details.

### TODO

A skeleton of a 1D fill kernel

## Exercise: Re-use threads in a 1D kernel with a for loop

Finally we did something other than just check for errors!

In the previous exercise we discussed the limits on the number of threads per block and blocks per grid.
The strategy used in the previous exercise was to couple the size of data to the launch parameters of the kernel.
It can be a good strategy for some situations, but other times it's better to reuse the threads and process
multiple values per thread.

If you're used to manually multithreaded CPU code, an obvious way to do it
is to divide the array among the threads, such that each thread processes consecutive values:
```cpp
__global__ void fill_for(size_t n, float *arr, float value) {
    // Global thread id, i.e. over the entire grid
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // How many threads in total in the entire grid
    const size_t num_threads = blockDim.x * gridDim.x;

    // How many elements per thread
    size_t num_per_thread = n / num_threads;
    
    // Process num_per_thread consecutive elements
    for (size_t i = 0; i < num_per_thread; i++) {
        // tid      elems
        //   0      [0, num_per_thread - 1]
        //   1      [num_per_thread, 2 * num_per_thread - 1]
        //   2      [2 * num_per_thread, 3 * num_per_thread - 1]
        //   and so on...
        arr[tid * num_per_thread + i] = value;
    }

    // How many are left over
    const size_t left_over = n - num_per_thread * num_threads;

    // The first threads will process one more, so the left over values
    // are also processed
    if (tid < left_over) {
        // tid      elem
        //   0      num_per_thread * num_threads
        //   1      num_per_thread * num_threads + 1
        //   2      num_per_thread * num_threads + 2
        //   and so on...
        arr[num_per_thread * num_threads + tid] = value;
    }
}
```

With GPUs, however, this is a very slow strategy! Remember that threads proceed in lock step,
in warps/wavefronts of 32/64 (Nvidia)/(AMD) threads. We will learn much more about memory accesses,
but for now it's enough to say that the slowup is roughly 10-100x, compared to the non-looped
version of the [previous exercise](06_fill).

A much faster way to achieve the same thing is to use a "strided for loop":
```cpp
__global__ void fill_for(size_t n, float *arr, float value) {
    // Global thread id, i.e. over the entire grid
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // How many threads in total in the entire grid
    const size_t stride = blockDim.x * gridDim.x;

    // Every thread processes a single element, then
    // jumps forward 'stride' elements, as long as
    // i < n
    for (size_t i = tid; i < n; i+= stride) {
        // tid      elems
        //   0      [0, stride,     2 * stride,     ...]
        //   1      [1, stride + 1, 2 * stride + 1, ...]
        //   2      [2, stride + 2, 2 * stride + 2, ...]
        //   3      [3, stride + 3, 2 * stride + 3, ...]
        //   and so on...
        arr[i] = value;
    }
}
```

With this strategy, the consecutive threads in a warp/wavefront process consecutive elements
in the array, then jump forward together and againe process consecutive elements in the array,
until some or all of them jump out of the limit of the array.
Note that there's no chance of out-of-bounds memory access here: if a threads has `i >= n`, 
it doesn't execute the loop but just exits early. Some threads in the warp/wavefront with
a smaller `tid` may still execute the loop for a single iteration, but they'll also
end up with an `i` larger than `n` for the next iteration and then exit the loop.

### TODO

Make the exercise.
Maybe give two versions, the slow and the strided, then ask to use unix command `time` to measure the time?

## Bonus exercises

If you've reached this far, congratulations! You've learned the very basics of launching kernels
on the GPU, calling the HIP API and finding out about and fixing errors related to the kernels & API.

If you're hungry for more, you can find more exercises applying the learned topics in the
[bonus](../../bonus/02-kernels) diretory.

[^1]: Note for those who are somewhat familiar with Cuda: until ROCm 7.0 `hipGetLastError()`
works differently from `cudaGetLastError()`. `hipGetLastError()` returns the error code from any
HIP API, even if the call returned `hipSuccess`. This means `hipDeviceSynchronize()` call cannot
be put between the kernel launch and `hipGetLastError()`, or the return code from the kernel will
be overwritten by the success from the synchronization. With ROCm >= 7.0 and with Cuda,
you should synchronize with the device immediately after the kernel launch, then call `hipGetLastError()`
to get any errors that happen during the kernel execution, not just at the launch. Do note, that this
synchronization should only be done with debug builds, as it's a very expensive operation.
