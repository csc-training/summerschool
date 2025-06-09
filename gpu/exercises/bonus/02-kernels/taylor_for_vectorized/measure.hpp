#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include <hip/hip_runtime.h>

using namespace std::literals;

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file,
                              int32_t line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

#define LAUNCH_KERNEL(kernel, ...)                                             \
    launch_kernel(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)
template <typename... Args>
void launch_kernel(const char *kernel_name, const char *file, int32_t line,
                   void (*kernel)(Args...), dim3 blocks, dim3 threads,
                   size_t num_bytes_shared_mem, hipStream_t stream,
                   Args... args) {
#if !NDEBUG
    int32_t device = 0;
    HIP_ERRCHK(hipGetDevice(&device));

    // Helper lambda for querying device attributes
    auto get_device_attribute = [&device](hipDeviceAttribute_t attribute) {
        int32_t value = 0;
        HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute, device));
        return value;
    };

    // Get maximum allowed size of block for each dimension
    const dim3 max_threads(
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimX),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimY),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimZ));

    // Get maximum allowed size of grid for each dimension
    const dim3 max_blocks(
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimX),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimY),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimZ));

    // Maximum threads per block in total (i.e. x * y * z)
    const int32_t max_threads_per_block = get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerBlock);

    // Maximum number of bytes of shared memory per block
    const int32_t max_shared_memory_per_block = get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxSharedMemoryPerBlock);

    // Helper lambda for asserting dim3 launch variable is within allowed limits
    auto assert_within_limits = [](const char *name, int32_t value, int32_t min,
                                   int32_t max) {
        if (not(min <= value && value <= max)) {
            std::fprintf(stderr, "%s (%d) not within limits [%d, %d]\n", name,
                         value, min, max);
            exit(EXIT_FAILURE);
        }
    };

    assert_within_limits("threads.x", threads.x, 1, max_threads.x);
    assert_within_limits("threads.y", threads.y, 1, max_threads.y);
    assert_within_limits("threads.z", threads.z, 1, max_threads.z);
    assert_within_limits("blocks.x", blocks.x, 1, max_blocks.x);
    assert_within_limits("blocks.y", blocks.y, 1, max_blocks.y);
    assert_within_limits("blocks.z", blocks.z, 1, max_blocks.z);
    assert_within_limits("block size", threads.x * threads.y * threads.z, 1,
                         max_threads_per_block);

    // Requested amount of shared memory must be below the limit queried above
    if (num_bytes_shared_mem > max_shared_memory_per_block) {
        std::fprintf(stderr, "Shared memory request too large: %ld > %d\n",
                     num_bytes_shared_mem, max_shared_memory_per_block);
        exit(EXIT_FAILURE);
    }

    // Reset the error variable to success
    [[maybe_unused]] auto result = hipGetLastError();
#endif

    kernel<<<blocks, threads, num_bytes_shared_mem, stream>>>(args...);

#if !NDEBUG
    // Quoting from HIP documentation
    // (https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/error_handling.html)
    //
    //  > hipGetLastError() returns the returned error code of the last HIP
    //    runtime API call even if it’s hipSuccess, while cudaGetLastError
    //    returns the error returned by any of the preceding CUDA APIs in the
    //    same host thread. hipGetLastError() behavior will be matched with
    //    cudaGetLastError in ROCm release 7.0.
    //
    // Because of this, using the Cuda recommended pattern of cathcing kernel
    // errors by first synchronizing with the device, then calling
    // hipGetLastError doesn't work. Until ROCm 7.0, HIP will overwrite the
    // error code returned by the kernel with success from hipDeviceSynchronize.
    // This means hipGetLastError can only be used to catch launch parameter
    // errors, i.e. errors that happen during the kernel launch, like too many
    // threads per block. Any errors that happen during the asynchronous kernel
    // execution are missed. To be able to catch even the kernel launch errors,
    // one must not synchronize first, if using ROCm < 7.0, or the errors will
    // be overwritten.

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
    [[maybe_unused]] result = hipDeviceSynchronize();
#endif
    result = hipGetLastError();
    if (result != hipSuccess) {
        printf("Error with kernel \"%s\" in %s at line %d\n%s: %s\n",
               kernel_name, file, line, hipGetErrorName(result),
               hipGetErrorString(result));
        exit(EXIT_FAILURE);
    }
#endif
}

template <typename F, typename... Args>
__global__ void kernel(F f, std::tuple<Args...> args) {
    __shared__ extern uint8_t shmem[];
    std::apply(f, std::tuple_cat(std::make_tuple(threadIdx, blockDim, blockIdx,
                                                 gridDim, shmem),
                                 args));
}

template <typename Validator, typename... Args> struct Measure {
  private:
    std::tuple<Args...> args;
    std::vector<size_t> microseconds = {};
    Validator validator;
    hipStream_t stream = 0;
    hipEvent_t start = 0;
    hipEvent_t stop = 0;

  public:
    Measure(Validator validator, Args... args)
        : args(std::make_tuple(args...)), validator(validator) {
        HIP_ERRCHK(hipStreamCreate(&stream));
        HIP_ERRCHK(hipEventCreate(&start));
        HIP_ERRCHK(hipEventCreate(&stop));
    }

    ~Measure() {
        HIP_ERRCHK(hipStreamDestroy(stream));
        HIP_ERRCHK(hipEventDestroy(start));
        HIP_ERRCHK(hipEventDestroy(stop));
    }

    template <typename F>
    void run_and_measure(std::string_view name, F f, dim3 blocks, dim3 threads,
                         size_t shared_bytes) {
        static constexpr size_t num_measurements = 20ul;
        size_t average_runtime = 0;

        for (size_t i = 0; i < num_measurements; i++) {
            HIP_ERRCHK(hipEventRecord(start, stream));
            LAUNCH_KERNEL(kernel, blocks, threads, shared_bytes, stream, f,
                          args);
            HIP_ERRCHK(hipEventRecord(stop, stream));
            HIP_ERRCHK(hipEventSynchronize(stop));

            float elapsed = 0.0f;
            HIP_ERRCHK(hipEventElapsedTime(&elapsed, start, stop));

            average_runtime += i == 0 ? 0 : elapsed * 1000.0f;
        }

        if (not validator()) {
            std::cerr << "Incorrect results for " << name << std::endl;
        }

        microseconds.push_back(average_runtime / (num_measurements - 1));
    }

    void output() {
        for (size_t i = 0; i < microseconds.size() - 1; i++) {
            std::printf("%ld, ", microseconds[i]);
        }
        std::printf("%ld\n", microseconds[microseconds.size() - 1]);
    }
};
