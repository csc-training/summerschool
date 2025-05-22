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

__global__ void taylor_no_reuse(float *x, float *y, size_t num_values,
                                size_t num_iters) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_values)
    {
        y[tid] = taylor(x[tid], num_iters);
    }
}

__global__ void taylor_for_cpu(float *x, float *y, size_t num_values,
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

__global__ void taylor_for_gpu(float *x, float *y, size_t num_values,
                               size_t num_iters) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < num_values; i += stride) {
        y[i] = taylor(x[i], num_iters);
    }
}

__global__ void taylor_for_vec(float *x, float *y, size_t num_values,
                               size_t num_iters) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    float4 *xv = reinterpret_cast<float4 *>(x);
    float4 *yv = reinterpret_cast<float4 *>(y);
    for (size_t i = tid; i < num_values / 4; i += stride) {
        const float4 xs = xv[i];
        const float4 ys(taylor(xs.x, num_iters), taylor(xs.y, num_iters),
                        taylor(xs.z, num_iters), taylor(xs.w, num_iters));

        yv[i] = ys;
    }

    const size_t index = num_values - num_values & 3 + tid;
    if (index < num_values) {
        y[index] = taylor(x[index], num_iters);
    }
}

size_t run_and_measure(void (*kernel)(float *, float *, size_t, size_t),
                       int32_t blocks, int32_t threads, size_t num_values,
                       size_t num_iters) {
    const size_t num_bytes = sizeof(float) * num_values;

    // Allocate host memory for x and y
    float *h_x = static_cast<float *>(std::malloc(num_bytes));
    float *h_y = static_cast<float *>(std::malloc(num_bytes));
    float *reference = static_cast<float *>(std::malloc(num_bytes));

    // Initialize host x with some values
    for (size_t i = 0; i < num_values; i++) {
        h_x[i] = static_cast<float>(i) / num_values * 3.14159265f;
        reference[i] = taylor(h_x[i], num_iters);
    }

    // Allocate device memory for x and y
    float *d_x = nullptr;
    float *d_y = nullptr;
    HIP_ERRCHK(hipMalloc(&d_x, num_bytes));
    HIP_ERRCHK(hipMalloc(&d_y, num_bytes));

    // Copy host x to device x
    HIP_ERRCHK(hipMemcpy(d_x, h_x, num_bytes, hipMemcpyDefault));

    // Run the kernel N times and compute the average runtime
    // Don't count the first run, as it may contain extra time unrelated to the
    // computation (GPU init etc.)
    static constexpr size_t num_measurements = 20ul;
    size_t average_runtime = 0;
    for (size_t i = 0; i < num_measurements; i++) {
        const auto start = std::chrono::high_resolution_clock::now();

        LAUNCH_KERNEL(kernel, blocks, threads, 0, 0, d_x, d_y, num_values,
                      num_iters);
        HIP_ERRCHK(hipDeviceSynchronize());

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::micro> dur = end - start;

        average_runtime += i == 0 ? 0 : dur.count();

        // Make sure we actually did the right thing in the kernel:
        // Copy device y to host y, and check y is equal to the taylor
        // computed with the corresponding host x
        HIP_ERRCHK(hipMemcpy(h_y, d_y, num_bytes, hipMemcpyDefault));
        float error = 0.0;
        static constexpr float tolerance = 1e-5f;
        for (size_t i = 0; i < num_values; i++) {
            const auto diff = abs(h_y[i] - reference[i]);
            if (diff > tolerance) {
                error += diff;
            }
        }

        assert(error < 0.01f);
    }
    std::free(h_x);
    std::free(h_y);
    std::free(reference);

    HIP_ERRCHK(hipFree(d_x));
    HIP_ERRCHK(hipFree(d_y));

    return average_runtime / (num_measurements - 1);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::printf(
            "Give two arguments: number of Taylor's expansion iterations "
            "and size of vector\n");
        std::printf("E.g. %s 20 100000000\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const size_t num_iters = std::atol(argv[1]);
    const size_t num_values = std::atol(argv[2]);
    assert(num_iters > 0 && num_iters <= 50 && "Use values up to 50");
    assert(num_values > 0 && num_values <= 100000000 &&
           "Use values up to 100'000'000");

    // Kernel with no thread re-use
    const int threads = 1024;
    int blocks = num_values / threads;
    blocks += blocks * threads < num_values ? 1 : 0;
    const auto no_reuse_runtime = run_and_measure(
        taylor_no_reuse, blocks, threads, num_values, num_iters);

    // Kernels with thread re-use can use arbitrary grid size. It should be
    // large enough to utilize all the availabe CUs of the GPU, however.
    blocks = 1024;

    // CPU style
    const auto cpu_runtime =
        run_and_measure(taylor_for_cpu, blocks, threads, num_values, num_iters);

    // GPU style
    const auto gpu_runtime =
        run_and_measure(taylor_for_gpu, blocks, threads, num_values, num_iters);

    // Vectorized loads style
    const auto vec_runtime =
        run_and_measure(taylor_for_vec, blocks, threads, num_values, num_iters);

    std::printf("%ld, 1.0, %f, %f, %f\n", no_reuse_runtime,
                static_cast<float>(gpu_runtime) / no_reuse_runtime,
                static_cast<float>(cpu_runtime) / no_reuse_runtime,
                static_cast<float>(vec_runtime) / no_reuse_runtime);

    return 0;
}
