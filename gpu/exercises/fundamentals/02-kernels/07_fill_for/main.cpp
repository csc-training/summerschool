#include <cstdio>
#include <hip/hip_runtime.h>

// This file include macros for checking the API and kernel launch errors
#include "../../../error_checking.hpp"

__device__ __host__ float taylor(float x) {
    float sum = 0.0f;
    float xn = 1.0f / x;
    float factorial = 1.0f;

    static constexpr size_t num_iters = 20ul;
    for (size_t n = 0; n < num_iters; n++) {
        xn *= x;
        factorial *= std::max(static_cast<float>(n), 1.0f);
        sum += xn / factorial;
    }
    return sum;
}

__global__ void taylor_no_reuse(float *x, float *y, size_t num_values) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_values)
    {
        y[tid] = taylor(x[tid]);
    }
}

__global__ void taylor_for_cpu_style(float *x, float *y, size_t num_values) {
    // Global thread id, i.e. over the entire grid
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // How many threads in total in the entire grid
    const size_t num_threads = blockDim.x * gridDim.x;

    // How many elements per thread
    size_t num_per_thread = num_values / num_threads;

    // Process num_per_thread consecutive elements
    for (size_t i = 0; i < num_per_thread; i++) {
        // tid      elems
        //   0      [0, num_per_thread - 1]
        //   1      [num_per_thread, 2 * num_per_thread - 1]
        //   2      [2 * num_per_thread, 3 * num_per_thread - 1]
        //   and so on...
        const size_t j = tid * num_per_thread + i;
        y[j] = taylor(x[j]);
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
        y[j] = taylor(x[j]);
    }
}

__global__ void taylor_for_gpu_style(float *x, float *y, size_t num_values) {
    // Global thread id, i.e. over the entire grid
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // How many threads in total in the entire grid
    const size_t stride = blockDim.x * gridDim.x;

    // Every thread processes a single element, then
    // jumps forward 'stride' elements, as long as
    // i < n
    for (size_t i = tid; i < num_values; i += stride) {
        // tid      elems
        //   0      [0, stride,     2 * stride,     ...]
        //   1      [1, stride + 1, 2 * stride + 1, ...]
        //   2      [2, stride + 2, 2 * stride + 2, ...]
        //   3      [3, stride + 3, 2 * stride + 3, ...]
        //   and so on...
        y[i] = taylor(x[i]);
    }
}

void run_and_measure(const char *style_name,
                     void (*kernel)(float *, float *, size_t), int32_t blocks,
                     int32_t threads, size_t num_values) {
    const size_t num_bytes = sizeof(float) * num_values;

    // Allocate host memory for x and y
    float *h_x = static_cast<float *>(std::malloc(num_bytes));
    float *h_y = static_cast<float *>(std::malloc(num_bytes));

    // Initialize host x with some values
    for (size_t i = 0; i < num_values; i++) {
        h_x[i] = static_cast<float>(i) / num_values * 100.0f;
    }

    // Allocate device memory for x and y
    float *d_x = nullptr;
    float *d_y = nullptr;
    HIP_ERRCHK(hipMalloc(&d_x, num_bytes));
    HIP_ERRCHK(hipMalloc(&d_y, num_bytes));

    // Copy host x to device x
    HIP_ERRCHK(hipMemcpy(d_x, h_x, num_bytes, hipMemcpyDefault));

    // Run the kernel 20 times and compute the average runtime of the last 19
    // runs
    constexpr auto n_iter = 20;
    size_t avg = 0;
    for (auto iteration = 0; iteration < n_iter; iteration++) {
        const auto start = std::chrono::high_resolution_clock::now();

        LAUNCH_KERNEL(kernel, blocks, threads, 0, 0, d_x, d_y, num_values);
        HIP_ERRCHK(hipDeviceSynchronize());

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::nano> dur = end - start;

        avg += iteration == 0 ? 0 : dur.count();

        // Make sure we actually did the right thing in the kernel:
        // Copy device y to host y, and check y is equal to the taylor
        // computed with the corresponding host x
        HIP_ERRCHK(hipMemcpy(h_y, d_y, num_bytes, hipMemcpyDefault));
        float error = 0.0;
        static constexpr float tolerance = 1e-6f;
        for (size_t i = 0; i < num_values; i++) {
            const auto diff = abs(h_y[i] - taylor(h_x[i]));
            if (diff > tolerance) {
                error += diff;
            }
        }
        assert(error < 0.01f);
    }
    std::free(h_x);
    std::free(h_y);

    HIP_ERRCHK(hipFree(d_x));
    HIP_ERRCHK(hipFree(d_y));

    std::fprintf(stdout, "Average runtime in nanoseconds for %s: %ld\n",
                 style_name, avg / (n_iter - 1));
}

int main() {
    static constexpr size_t num_values = 1000000;

    // Kernel with no thread re-use
    const int threads = 1024;
    int blocks = num_values / threads;
    blocks += blocks * threads < num_values ? 1 : 0;
    run_and_measure("no reuse", taylor_no_reuse, blocks, threads, num_values);

    // Kernels with thread re-use can use arbitrary grid size
    blocks = 128;

    // CPU style
    run_and_measure("cpu style", taylor_for_cpu_style, blocks, threads,
                    num_values);

    // GPU style
    run_and_measure("gpu style", taylor_for_gpu_style, blocks, threads,
                    num_values);

    return 0;
}
