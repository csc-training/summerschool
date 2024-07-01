#include "common.h"
#include <cstddef>
#include <hip/hip_runtime.h>

__global__ void saxpy_(size_t n, float a, float *x, float *y, float *r) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        saxpy(tid, a, x, y, r);
    }
}

__global__ void init_data(size_t n, float *x, float *y) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        init_x(tid, x);
        init_y(tid, y);
    }
}

void *gpu_allocate(size_t bytes) {
    void *p = nullptr;
    [[maybe_unused]] const auto result = hipMalloc(&p, bytes);
    return p;
}

void gpu_free(void *p) { [[maybe_unused]] const auto result = hipFree(p); }

void gpu_init(size_t n, float *x, float *y) {
    constexpr dim3 blocks(32);
    constexpr dim3 threads(256);
    init_data<<<blocks, threads, 0, 0>>>(n, x, y);
}

int main() {
    run(gpu_allocate, gpu_free, gpu_init,
        [](auto n, auto a, auto *x, auto *y, auto *r) -> auto {
            constexpr dim3 blocks(32);
            constexpr dim3 threads(256);

            saxpy_<<<blocks, threads, 0, 0>>>(n, a, x, y, r);
            [[maybe_unused]] const auto result = hipDeviceSynchronize();
        });
}
