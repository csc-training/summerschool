#include <cstddef>
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void kernel1(size_t n, float *x, float *y) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        x[tid] = 0.666f * sin(tid);
        y[tid] = 1.337f * cos(tid);
    }
}

__global__ void kernel2(size_t n, float a, float *x, float *y, float *r) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        r[tid] = a * x[tid] + y[tid];
    }
}

__global__ void kernel3(size_t n, float a, float *x, float *y, float *r) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        const float x1 = x[tid];
        const float x2 = x1 * x1;
        const float x3 = x1 * x2;
        const float x4 = x2 * x2;

        const float y1 = y[tid];
        const float y2 = y1 * y1;
        const float y3 = y1 * y2;
        const float y4 = y2 * y2;
        // clang-format off
        r[tid] = 
              1.0f * a * x1
            - 2.0f * a * x2
            + 3.0f * a * x3
            - 4.0f * a * x4
            + 4.0f * a * y1
            - 3.0f * a * y2
            + 2.0f * a * y3
            - 1.0f * a * y4;
        // clang-format on
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

    constexpr dim3 blocks(1024);
    constexpr dim3 threads(1024);
    kernel1<<<blocks, threads, 0, 0>>>(n, x, y);
    kernel2<<<blocks, threads, 0, 0>>>(n, a, x, y, r);
    kernel3<<<blocks, threads, 0, 0>>>(n, a, x, y, r);
    [[maybe_unused]] auto t = hipDeviceSynchronize();

    hipFree(x);
    hipFree(y);
    hipFree(r);
}
