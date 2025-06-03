#include <cstdio>
#include <cstdlib>

#include "measure.hpp"

struct Computation {};

template <typename F>
void initialize(F f, float **h_x, float **h_y, float **reference, float **d_x,
                float **d_y, size_t num_values) {
    const size_t num_bytes = sizeof(float) * num_values;
    // Allocate host memory for x and y
    *h_x = static_cast<float *>(std::malloc(num_bytes));
    *h_y = static_cast<float *>(std::malloc(num_bytes));
    *reference = static_cast<float *>(std::malloc(num_bytes));

    // Initialize host x with some values
    for (size_t i = 0; i < num_values; i++) {
        (*h_x)[i] = static_cast<float>(i) / num_values * 3.14159265f;
        (*reference)[i] = f((*h_x)[i]);
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

    const size_t N = std::atol(argv[1]);
    const size_t num_values = std::atol(argv[2]);
    const size_t threads = std::atol(argv[3]);

    assert(N > 0 && N <= 50 && "Use values up to 50");
    assert(num_values > 0 && num_values <= 100000000 &&
           "Use values up to 100'000'000");
    assert(threads > 0 && threads <= 1024 && "Use values up to 1024");

    float *h_x = nullptr;
    float *h_y = nullptr;
    float *reference = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;

    auto f = [N] __host__ __device__(float x) {
        float sum = 1.0f;
        float term = 1.0f;
        for (size_t n = 1; n <= N; n++) {
            term *= x / n;
            sum += term;
        }
        return sum;
    };

    initialize(f, &h_x, &h_y, &reference, &d_x, &d_y, num_values);

    auto validate_result = [h_y, d_y, reference, num_values]() {
        const size_t num_bytes = num_values * sizeof(float);
        HIP_ERRCHK(hipMemcpy(h_y, d_y, num_bytes, hipMemcpyDefault));

        float error = 0.0;
        static constexpr float tolerance = 1e-5f;
        for (size_t i = 0; i < num_values; i++) {
            const auto diff = abs(h_y[i] - reference[i]);
            if (diff > tolerance) {
                std::fprintf(
                    stderr,
                    "Large error at i = %lu: h_y[i] = %f, reference[i] = %f\n",
                    i, h_y[i], reference[i]);
                error += diff;
            }
        }

        HIP_ERRCHK(hipMemset(d_y, 0, num_bytes));

        return error < 0.01f;
    };

    Measure measure(validate_result, num_values, f, d_x, d_y);

    auto base = [] __host__ __device__(
                    auto threadIdx, auto blockDim, auto blockIdx, auto gridDim,
                    uint8_t *, size_t num_values, auto f, float *x, float *y) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num_values) {
            y[tid] = f(x[tid]);
        }
    };

    auto vectorized = [] __host__ __device__(auto threadIdx, auto blockDim,
                                             auto blockIdx, auto gridDim,
                                             uint8_t *, size_t num_values,
                                             auto f, float *x, float *y) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        float4 *xv = reinterpret_cast<float4 *>(x);
        float4 *yv = reinterpret_cast<float4 *>(y);

        if (tid < num_values >> 2) {
            const float4 xs = xv[tid];
            yv[tid] = float4(f(xs.x), f(xs.y), f(xs.z), f(xs.w));
        }

        const size_t index = num_values - (num_values & 3) + tid;
        if (index < num_values) {
            y[index] = f(x[index]);
        }
    };

    auto consecutive = [] __host__ __device__(auto threadIdx, auto blockDim,
                                              auto blockIdx, auto gridDim,
                                              uint8_t *, size_t num_values,
                                              auto f, float *x, float *y) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        const size_t num_per_thread = num_values / stride;
        for (size_t i = 0; i < num_per_thread; i++) {
            const size_t j = tid * num_per_thread + i;
            y[j] = f(x[j]);
        }

        const size_t index = num_per_thread * stride + tid;
        if (index < num_values) {
            y[index] = f(x[index]);
        }
    };

    auto strided = [] __host__ __device__(auto threadIdx, auto blockDim,
                                          auto blockIdx, auto gridDim,
                                          uint8_t *, size_t num_values, auto f,
                                          float *x, float *y) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;
        for (size_t i = tid; i < num_values; i += stride) {
            y[i] = f(x[i]);
        }
    };

    auto strided_vectorized =
        [] __host__ __device__(auto threadIdx, auto blockDim, auto blockIdx,
                               auto gridDim, uint8_t *, size_t num_values,
                               auto f, float *x, float *y) {
            const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            float4 *xv = reinterpret_cast<float4 *>(x);
            float4 *yv = reinterpret_cast<float4 *>(y);

            for (size_t i = tid; i < num_values >> 2; i += stride) {
                const float4 xs = xv[i];
                yv[i] = float4(f(xs.x), f(xs.y), f(xs.z), f(xs.w));
            }

            const size_t index = num_values - (num_values & 3) + tid;
            if (index < num_values) {
                y[index] = f(x[index]);
            }
        };

    size_t blocks = num_values / threads;
    blocks += blocks * threads < num_values ? 1 : 0;
    measure.run_and_measure("Base"sv, base, blocks, threads, 0);

    blocks = num_values / (4 * threads);
    blocks += 4 * threads * blocks < num_values ? 1 : 0;
    measure.run_and_measure("Vectorized"sv, vectorized, blocks, threads, 0);

    blocks = 1024;
    measure.run_and_measure("Consecutive"sv, consecutive, blocks, threads, 0);
    measure.run_and_measure("Strided"sv, strided, blocks, threads, 0);
    measure.run_and_measure("Strided vectorized"sv, strided_vectorized, blocks,
                            threads, 0);

    std::printf("%ld, %ld, %ld, ", N, 2 * sizeof(float) * num_values, threads);
    measure.output();

    std::free(h_x);
    std::free(h_y);
    std::free(reference);

    HIP_ERRCHK(hipFree(d_x));
    HIP_ERRCHK(hipFree(d_y));

    return 0;
}
