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

Did you launch the program with 1025 threads and 10 blocks? What happened?
To find out more, let's move on to the next exercise.

### TODO

Change the exercise so it takes two integer parameters from the user:
- num threads
- num blocks

## Exercise: Error reporting from kernel launch

If a kernel launch configuration parameter (number of threads/blocks or the amount of shared memory) is incorrect,
the kernel is not launched at all. It just silently fails.

Kernel launches don't return anything, so you also don't get an error code. But there are ways to catch errors.

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
    [[maybe_unused]] auto result = hipGetLastError();

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
int32_t device = 0;
int32_t value = 0;

// Get the current device
auto result = hipGetDevice(&device));
// Get the value for attribute from device
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

A full list of attributes can be found in the [HIP runtime API documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/global_defines_enums_structs_files/global_enum_and_defines.html#_CPPv420hipDeviceAttribute_t).

### TODO

Make the exercise.
Don't use the exactly same, hardcode the dim3 sizes to better showcase the problem.
Add stuff to the previous kernel launch macro.


## Exercise: Errors from API calls

Similar to the kernel launch, we can check errors of the API calls.
In this doc, tell that API returns error codes (as was briefly mentioned in the kernel wrapping).
Showcase manual checking with multiple ifs.

Then introduce a wrapper macro.

Ask the user to fix the errors in the code by wrapping the API methods with the macro.

### TODO

The exercise should have some API calls that cause errors. The user should then fix those calls,
first by wrapping the calls with the macro, then by figuring out the meaning of the message.

## Exercise: Kernel for filling a 1D array with a value

Something useful finally: create a kernel for filling GPU memory with a given value.

## Exercise: For loops in 1D kernels

Re-use threads in a 1D kernel using a for loop.

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
