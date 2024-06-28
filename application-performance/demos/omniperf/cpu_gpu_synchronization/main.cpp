#include <cstddef>
#include <hip/hip_runtime.h>

__global__ void saxpy_(size_t n, float a, float *x, float *y, float *r) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        r[tid] = a * x[tid] + y[tid];
    }
}

__global__ void init_data(size_t n, float *x, float *y) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        x[tid] = 2.3f * sin(tid);
        y[tid] = 1.1f * cos(tid);
    }
}

void *gpu_allocate(size_t bytes) {
    void *p = nullptr;
    [[maybe_unused]] const auto result = hipMalloc(&p, bytes);
    return p;
}

int main() {
    constexpr size_t n = 1 << 30;
    constexpr size_t num_bytes = sizeof(float) * n;
    constexpr float a = 3.4f;

    float *const x = static_cast<float *>(gpu_allocate(num_bytes));
    float *const y = static_cast<float *>(gpu_allocate(num_bytes));
    float *const r = static_cast<float *>(gpu_allocate(num_bytes));

    constexpr dim3 blocks(32);
    constexpr dim3 threads(256);
    init_data<<<blocks, threads, 0, 0>>>(n, x, y);

    for (size_t i = 0; i < 10; i++) {
        saxpy_<<<blocks, threads, 0, 0>>>(n, a, x, y, r);
        [[maybe_unused]] const auto result = hipDeviceSynchronize();
    }

    hipFree(x);
    hipFree(y);
    hipFree(r);
}
