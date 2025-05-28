#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

// This file include macros for checking the API and kernel launch errors
#include "../../../../error_checking.hpp"

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

__global__ void taylor_vec(float *x, float *y, size_t num_values,
                           size_t num_iters) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    float4 *xv = reinterpret_cast<float4 *>(x);
    float4 *yv = reinterpret_cast<float4 *>(y);

    if (tid < num_values >> 2) {
        const float4 xs = xv[tid];
        const float4 ys(taylor(xs.x, num_iters), taylor(xs.y, num_iters),
                        taylor(xs.z, num_iters), taylor(xs.w, num_iters));

        yv[tid] = ys;
    }

    const size_t index = num_values - num_values & 3 + tid;
    if (index < num_values) {
        y[index] = taylor(x[index], num_iters);
    }
}

__global__ void taylor_for_consecutive(float *x, float *y, size_t num_values,
                                       size_t num_iters) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t num_threads = blockDim.x * gridDim.x;
    const size_t num_per_thread = num_values / num_threads;
    for (size_t i = 0; i < num_per_thread; i++) {
        const size_t j = tid * num_per_thread + i;
        y[j] = taylor(x[j], num_iters);
    }

    const size_t left_over = num_values - num_per_thread * num_threads;
    if (tid < left_over) {
        const size_t j = num_per_thread * num_threads + tid;
        y[j] = taylor(x[j], num_iters);
    }
}

__global__ void taylor_for_strided(float *x, float *y, size_t num_values,
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
                       size_t num_iters, float *d_x, float *d_y) {
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
    }

    return average_runtime / (num_measurements - 1);
}

bool validate_result(float *h_y, float *d_y, float *reference,
                     size_t num_values) {
    HIP_ERRCHK(
        hipMemcpy(h_y, d_y, num_values * sizeof(float), hipMemcpyDefault));

    float error = 0.0;
    static constexpr float tolerance = 1e-5f;
    for (size_t i = 0; i < num_values; i++) {
        const auto diff = abs(h_y[i] - reference[i]);
        if (diff > tolerance) {
            error += diff;
        }
    }

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
            "Give three arguments: number of Taylor's expansion iterations,"
            " size of vector and size of block\n");
        std::printf("E.g. %s 20 100000000 256\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const size_t num_iters = std::atol(argv[1]);
    const size_t num_values = std::atol(argv[2]);
    const size_t threads = std::atol(argv[3]);
    assert(num_iters > 0 && num_iters <= 50 && "Use values up to 50");
    assert(num_values > 0 && num_values <= 100000000 &&
           "Use values up to 100'000'000");
    assert(threads > 0 && threads <= 1024 && "Use values up to 1024");

    float *h_x = nullptr;
    float *h_y = nullptr;
    float *reference = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;

    initialize(&h_x, &h_y, &reference, &d_x, &d_y, num_values, num_iters);

    // Base kernel with nothing fancy
    int blocks = num_values / threads;
    blocks += blocks * threads < num_values ? 1 : 0;
    const auto base_runtime = run_and_measure(taylor_base, blocks, threads,
                                              num_values, num_iters, d_x, d_y);
    assert(validate_result(h_y, d_y, reference, num_values) &&
           "Base result incorrect");

    // Using float4s to load and process 4 values at a time: reduce the number
    // of blocks by 4
    blocks /= 4;
    blocks += 4 * threads * blocks < num_values ? 1 : 0;
    const auto vec_runtime = run_and_measure(taylor_vec, blocks, threads,
                                             num_values, num_iters, d_x, d_y);
    assert(validate_result(h_y, d_y, reference, num_values) &&
           "Vectorized result incorrect");

    // Kernels with thread re-use can use arbitrary grid size. It should be
    // large enough to utilize all the availabe CUs of the GPU, however.
    blocks = 880;

    // Consecutive N per thread
    const auto consecutive_runtime =
        run_and_measure(taylor_for_consecutive, blocks, threads, num_values,
                        num_iters, d_x, d_y);
    assert(validate_result(h_y, d_y, reference, num_values) &&
           "Consecutive result incorrect");

    // Strided access
    const auto strided_runtime = run_and_measure(
        taylor_for_strided, blocks, threads, num_values, num_iters, d_x, d_y);
    assert(validate_result(h_y, d_y, reference, num_values) &&
           "Strided result incorrect");

    // Vectorized loads in loop
    const auto vec_for_runtime = run_and_measure(
        taylor_for_vec, blocks, threads, num_values, num_iters, d_x, d_y);
    assert(validate_result(h_y, d_y, reference, num_values) &&
           "Vec for result incorrect");

    std::printf("%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld\n", num_iters,
                2 * sizeof(float) * num_values, threads, base_runtime,
                vec_runtime, strided_runtime, consecutive_runtime,
                vec_for_runtime);

    std::free(h_x);
    std::free(h_y);
    std::free(reference);

    HIP_ERRCHK(hipFree(d_x));
    HIP_ERRCHK(hipFree(d_y));

    return 0;
}
