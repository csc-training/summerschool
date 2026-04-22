#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

// This file include macros for checking the API and kernel launch errors
#include "../../../error_checking.hpp"

__device__ __host__ float taylor(float x, size_t N) {
    float sum = 1.0f;
    float term = 1.0f;
    for (size_t n = 1; n <= N; n++) {
        term *= x / n;
        sum += term;
    }
    return sum;
}

__global__ void taylor_base(float *x, float *y, size_t num_values,
                            size_t num_iters) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_values)
    {
        y[tid] = taylor(x[tid], num_iters);
    }
}

__global__ void taylor_for_consecutive(float *x, float *y, size_t num_values,
                                       size_t num_iters) {
    // Global thread id, i.e. over the entire grid
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // How many threads in total in the entire grid
    const size_t num_threads = blockDim.x * gridDim.x;

    // How many elements per thread
    const size_t num_per_thread = num_values / num_threads;

    // Process num_per_thread consecutive elements
    for (size_t i = 0; i < num_per_thread; i++) {
        // tid      elems
        //   0      [0, num_per_thread - 1]
        //   1      [num_per_thread, 2 * num_per_thread - 1]
        //   2      [2 * num_per_thread, 3 * num_per_thread - 1]
        //   and so on...
        const size_t j = tid * num_per_thread + i;
        y[j] = taylor(x[j], num_iters);
    }

    // How many are left over
    const size_t left_over = num_values - num_per_thread * num_threads;

    // The first threads will process one more, so the left over values
    // are also processed
    if (tid < left_over) {
        // tid      elem
        //   0      num_per_thread * num_threads
        //   1      num_per_thread * num_threads + 1
        //   2      num_per_thread * num_threads + 2
        //   and so on...
        const size_t j = num_per_thread * num_threads + tid;
        y[j] = taylor(x[j], num_iters);
    }
}

__global__ void taylor_for_strided(float *x, float *y, size_t num_values,
                                   size_t num_iters) {
    // TODO: Fill in this kernel
    // Check the lecture slides for HIP kernels for an example on how to do a
    // strided for loop

    for (size_t i = ???; /*TODO: fill me in*/) {
        y[i] = taylor(x[i], num_iters);
    }
}

size_t run_and_measure(void (*kernel)(float *, float *, size_t, size_t),
                       int32_t num_blocks, int32_t num_threads,
                       size_t num_values, size_t num_iters, float *d_x,
                       float *d_y) {
    // Run the kernel N times and compute the average runtime
    // Don't count the first run, as it may contain extra time unrelated to the
    // computation (GPU init etc.)
    static constexpr size_t num_measurements = 20ul;
    size_t average_runtime = 0;
    for (size_t i = 0; i < num_measurements; i++) {
        const auto start = std::chrono::high_resolution_clock::now();

        LAUNCH_KERNEL(kernel, num_blocks, num_threads, 0, 0, d_x, d_y,
                      num_values, num_iters);
        HIP_ERRCHK(hipDeviceSynchronize());

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::micro> dur = end - start;

        average_runtime += i == 0 ? 0 : dur.count();
    }

    return average_runtime / (num_measurements - 1);
}

bool validate_result(float *h_y, float *d_y, float *reference,
                     size_t num_values) {
    const size_t num_bytes = num_values * sizeof(float);
    HIP_ERRCHK(hipMemcpy(h_y, d_y, num_bytes, hipMemcpyDefault));

    float error = 0.0;
    static constexpr float tolerance = 1e-5f;
    for (size_t i = 0; i < num_values; i++) {
        const auto diff = abs(h_y[i] - reference[i]);
        if (diff > tolerance) {
            std::fprintf(
                stderr,
                "Large error at i = %lu: h_y[i] = %f, reference[i] = %f\n", i,
                h_y[i], reference[i]);
            error += diff;
        }
    }

    HIP_ERRCHK(hipMemset(d_y, 0, num_bytes));

    return error < 0.01f;
}

void initialize(float **h_x, float **h_y, float **reference, float **d_x,
                float **d_y, size_t num_values, size_t num_iters) {
    const size_t num_bytes = sizeof(float) * num_values;
    // Allocate host memory for x and y
    *h_x = static_cast<float *>(std::malloc(num_bytes));
    *h_y = static_cast<float *>(std::malloc(num_bytes));
    *reference = static_cast<float *>(std::malloc(num_bytes));

    // Initialize host x with some values
    for (size_t i = 0; i < num_values; i++) {
        (*h_x)[i] = static_cast<float>(i) / num_values * 3.14159265f;
        (*reference)[i] = taylor((*h_x)[i], num_iters);
    }

    // Allocate device memory for x and y
    HIP_ERRCHK(hipMalloc(d_x, num_bytes));
    HIP_ERRCHK(hipMalloc(d_y, num_bytes));

    // Copy host x to device x
    HIP_ERRCHK(hipMemcpy(*d_x, *h_x, num_bytes, hipMemcpyDefault));
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::printf(
            "Give three arguments: number of Taylor's expansion iterations, "
            "size of vector and number of threads\n");
        std::printf("E.g. %s 20 100000000 256\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const size_t num_iters = std::atol(argv[1]);
    const size_t num_values = std::atol(argv[2]);
    const size_t num_threads = std::atol(argv[3]);
    assert(num_iters > 0 && num_iters <= 50 && "Use values up to 50");
    assert(num_values > 0 && num_values <= 100000000 &&
           "Use values up to 100'000'000");
    assert(num_threads > 0 && num_threads <= 1024 && "Use values up to 1024");

    float *h_x = nullptr;
    float *h_y = nullptr;
    float *reference = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;

    initialize(&h_x, &h_y, &reference, &d_x, &d_y, num_values, num_iters);

    // Base kernel with nothing fancy
    int num_blocks = num_values / num_threads;
    num_blocks += num_blocks * num_threads < num_values ? 1 : 0;
    const auto base_runtime = run_and_measure(
        taylor_base, num_blocks, num_threads, num_values, num_iters, d_x, d_y);
    assert(validate_result(h_y, d_y, reference, num_values) &&
           "Base result incorrect");

    // Kernels with thread re-use can use arbitrary grid size. It should be
    // large enough to utilize all the availabe CUs of the GPU, however.
    num_blocks = 1024;

    // Consecutive N per thread
    const auto consecutive_runtime =
        run_and_measure(taylor_for_consecutive, num_blocks, num_threads,
                        num_values, num_iters, d_x, d_y);
    assert(validate_result(h_y, d_y, reference, num_values) &&
           "Consecutive result incorrect");

    // Strided access
    const auto strided_runtime =
        run_and_measure(taylor_for_strided, num_blocks, num_threads, num_values,
                        num_iters, d_x, d_y);
    assert(validate_result(h_y, d_y, reference, num_values) &&
           "Strided result incorrect");

    std::printf("Taylor N, vector size, number of threads, base[us], "
                "strided[us], consecutive[us]\n");
    std::printf("%ld, %ld, %ld, %ld, %ld, %ld\n", num_iters, num_values,
                num_threads, base_runtime, strided_runtime,
                consecutive_runtime);

    std::free(h_x);
    std::free(h_y);
    std::free(reference);

    HIP_ERRCHK(hipFree(d_x));
    HIP_ERRCHK(hipFree(d_y));

    return 0;
}
