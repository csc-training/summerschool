#include <cstddef>
#include <hip/hip_runtime.h>

__global__ void init(size_t num_rows, float *a, float *b, float *c) {
    const size_t col = threadIdx.x;
    size_t row = threadIdx.y + blockIdx.x * blockDim.y;
    const size_t row_stride = gridDim.x * blockDim.y;

    for (; row < num_rows; row += row_stride) {
        const size_t i = col + row * blockDim.x;
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
}

__global__ void row_major(size_t num_rows, float *a, float *b, float *c) {
    const size_t col = threadIdx.x;
    size_t row = threadIdx.y + blockIdx.x * blockDim.y;
    const size_t row_stride = gridDim.x * blockDim.y;

    for (; row < num_rows; row += row_stride) {
        const size_t i = col + row * blockDim.x;
        c[i] = a[i] + b[i];
    }
}

__global__ void col_major(size_t num_rows, float *a, float *b, float *c) {
    const size_t col = threadIdx.x;
    size_t row = threadIdx.y + blockIdx.x * blockDim.y;
    const size_t row_stride = gridDim.x * blockDim.y;

    for (; row < num_rows; row += row_stride) {
        const size_t i = row + col * num_rows;
        c[i] = a[i] + b[i];
    }
}

void *gpu_allocate(size_t bytes) {
    void *p = nullptr;
    [[maybe_unused]] const auto result = hipMalloc(&p, bytes);
    return p;
}

int main() {
    constexpr size_t num_rows = 1 << 24;
    constexpr size_t num_cols = 64;
    constexpr size_t n = num_rows * num_cols;
    constexpr size_t num_bytes = sizeof(float) * n;

    float *const a = static_cast<float *>(gpu_allocate(num_bytes));
    float *const b = static_cast<float *>(gpu_allocate(num_bytes));
    float *const c = static_cast<float *>(gpu_allocate(num_bytes));

    constexpr dim3 blocks(1024);
    constexpr dim3 threads(64, 16);
    row_major<<<blocks, threads, 0, 0>>>(num_rows, a, b, c);
    col_major<<<blocks, threads, 0, 0>>>(num_rows, a, b, c);

    [[maybe_unused]] auto t = hipDeviceSynchronize();

    hipFree(a);
    hipFree(b);
    hipFree(c);
}
