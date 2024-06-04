#include "common.h"
#include <chrono>
#include <hip/hip_runtime.h>

__global__ void saxpy_(int n, float a, float *x, float *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        saxpy(tid, a, x, y);
    }
}

int main() {
    run([](auto n, auto a, auto &x, auto &y) -> auto {
        // Setup code
        const size_t num_bytes = n * sizeof(decltype(x.back()));
        float *d_x = nullptr;
        float *d_y = nullptr;
        hipMalloc(reinterpret_cast<void **>(&d_x), num_bytes);
        hipMalloc(reinterpret_cast<void **>(&d_y), num_bytes);
        hipMemcpy(d_x, x.data(), num_bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_y, y.data(), num_bytes, hipMemcpyHostToDevice);

        const dim3 blocks(32);
        const dim3 threads(256);

        const auto c_start = std::chrono::high_resolution_clock::now();
        saxpy_<<<blocks, threads, 0, 0>>>(n, a, d_x, d_y);
        hipDeviceSynchronize();
        const auto c_end = std::chrono::high_resolution_clock::now();

        hipFree(d_x);
        hipFree(d_y);

        return c_end - c_start;
    });
}
