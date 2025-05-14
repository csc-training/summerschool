#include <hip/hip_runtime.h>

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
    // Helper lambda for querying device attributes
    auto get_device_attribute = [](hipDeviceAttribute_t attribute) {
        int32_t device = 0;
        int32_t value = 0;

        HIP_ERRCHK(hipGetDevice(&device));
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

    // Helper lambda for printing out stuff to stderr and exiting
    auto print_and_exit = [&](auto msg_lambda) {
        std::fprintf(stderr,
                     "Bad launch parameter for %s at %s:%d: ", kernel_name,
                     file, line);
        msg_lambda();
        exit(EXIT_FAILURE);
    };

    // Helper lambda for asserting dim3 launch variable is within allowed limits
#define ASSERT_DIM_WITHIN_LIMITS(dim, ...)                                     \
    assert_dim_within_limits(#dim, dim, __VA_ARGS__)
    auto assert_dim_within_limits = [&](const char *dim_name, dim3 dim,
                                        dim3 min, dim3 max, int32_t max_total) {
        if (not(min.x <= dim.x && dim.x <= max.x)) {
            print_and_exit([&]() {
                std::fprintf(stderr, "%s.x (%d) not within limits [%d, %d]\n",
                             dim_name, dim.x, min.x, max.x);
            });
        }

        if (not(min.y <= dim.y && dim.y <= max.y)) {
            print_and_exit([&]() {
                std::fprintf(stderr, "%s.y (%d) not within limits [%d, %d]\n",
                             dim_name, dim.y, min.y, max.y);
            });
        }

        if (not(min.z <= dim.z && dim.z <= max.z)) {
            print_and_exit([&]() {
                std::fprintf(stderr, "%s.z (%d) not within limits [%d, %d]\n",
                             dim_name, dim.z, min.z, max.z);
            });
        }

        if (not(dim.x * dim.y * dim.z <= max_total)) {
            print_and_exit([&]() {
                std::fprintf(stderr,
                             "Total size of %s (%d * %d * %d = %d) larger "
                             "than maximum (%d)\n",
                             dim_name, dim.x, dim.y, dim.z,
                             dim.x * dim.y * dim.z, max_total);
            });
        }
    };

    ASSERT_DIM_WITHIN_LIMITS(threads, dim3(1, 1, 1), max_threads,
                             max_threads_per_block);
    ASSERT_DIM_WITHIN_LIMITS(blocks, dim3(1, 1, 1), max_blocks,
                             std::numeric_limits<int32_t>::max());

    // Requested amount of shared memory must be below the limit queried above
    if (num_bytes_shared_mem > max_shared_memory_per_block) {
        print_and_exit([&]() {
            std::fprintf(stderr, "Shared memory request too large: %ld > %d\n",
                         num_bytes_shared_mem, max_shared_memory_per_block);
        });
    }

    // Reset the error variable to success
    auto result = hipGetLastError();
#endif
    kernel<<<blocks, threads, num_bytes_shared_mem, stream>>>(args...);
#if !NDEBUG

    // To catch the error by the asynchronous kernel launch, we must synchronize
    // with the device. This is of course a very costly operation, if the device
    // kernel takes a long time to execute. This should be enabled only for
    // debug builds, and skipped for release builds, where speed is important.
    HIP_ERRCHK(hipDeviceSynchronize());
    HIP_ERRCHK(hipGetLastError());
#endif
}
