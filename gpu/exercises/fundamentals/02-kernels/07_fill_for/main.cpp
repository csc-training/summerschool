#include <cstdio>
#include <hip/hip_runtime.h>

// This file include macros for checking the API and kernel launch errors
#include "../../../error_checking.hpp"

__global__ void fill(float *arr, float a, size_t num_values) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_values)
    {
        arr[tid] = a;
    }
}

__global__ void fill_for_cpu_style(float *arr, float a, size_t num_values) {
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
        arr[tid * num_per_thread + i] = a;
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
        arr[num_per_thread * num_threads + tid] = a;
    }
}

__global__ void fill_for(float *arr, float a, size_t num_values) {
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
        arr[i] = a;
    }
}

void run_and_measure(void (*kernel)(float *, float, size_t), int32_t blocks,
                     int32_t threads, float *d_arr, float a,
                     size_t num_values) {
    const size_t num_bytes = sizeof(float) * num_values;
    float *h_arr = static_cast<float *>(std::malloc(num_bytes));

    constexpr auto n_iter = 20;
    size_t avg = 0;
    for (auto iteration = 0; iteration < n_iter; iteration++) {
        const auto start = std::chrono::high_resolution_clock::now();

        LAUNCH_KERNEL(kernel, blocks, threads, 0, 0, d_arr, a, num_values);
        HIP_ERRCHK(hipDeviceSynchronize());

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::nano> dur = end - start;

        avg += iteration == 0 ? 0 : dur.count();

        // Make sure we actually did the right thing in the kernel
        HIP_ERRCHK(hipMemcpy(h_arr, d_arr, num_bytes, hipMemcpyDefault));
        for (size_t i = 0; i < num_values; i++) {
            assert(h_arr[i] == a && "The values are incorrect");
        }
    }
    std::free(h_arr);

    std::fprintf(stdout, "Average runtime in nanoseconds: %ld\n",
                 avg / (n_iter - 1));
}

int main() {
    static constexpr size_t num_values = 1000000;
    static constexpr size_t num_bytes = sizeof(float) * num_values;
    static constexpr float a = 3.4f;

    float *d_arr = nullptr;
    HIP_ERRCHK(hipMalloc(&d_arr, num_bytes));

    // Kernel with no thread re-use
    const int threads = 1024;
    int blocks = num_values / threads;
    blocks += blocks * threads < num_values ? 1 : 0;
    run_and_measure(fill, blocks, threads, d_arr, a, num_values);

    // Kernel with thread re-use, CPU style
    blocks = 128;
    run_and_measure(fill_for_cpu_style, blocks, threads, d_arr, a, num_values);

    // Kernel with thread re-use, GPU style
    run_and_measure(fill_for, blocks, threads, d_arr, a, num_values);

    HIP_ERRCHK(hipFree(d_arr));

    return 0;
}
