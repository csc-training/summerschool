#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>

__global__ void fill(size_t n, float a, float *arr) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        arr[i] = a;
    }
}

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}


int main() {
    static constexpr size_t n = 100000;
    static constexpr size_t num_bytes = sizeof(float) * n;
    static constexpr float a = 3.4f;

    std::vector<float> h_arr(n);

    // Allocate
    void *d_arr = nullptr;
    HIP_ERRCHK(hipMalloc(&d_arr, num_bytes));

    dim3 blocks(0, 0, 0);
    dim3 threads(1025, 1, 1);
    // TODO add error checking
    // KERNEL_LAUNCH(fill, ...)
    fill<<<blocks, threads>>>(n, a, static_cast<float *>(d_arr));

    // Copy results back to CPU
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(h_arr.data()), d_arr, num_bytes,
                         hipMemcpyDefault));

    // Free device memory
    HIP_ERRCHK(hipFree(d_arr));

    printf("ref value: %f, result: %f %f %f %f ... %f %f\n", a, h_arr[0],
           h_arr[1], h_arr[2], h_arr[3], h_arr[n - 2], h_arr[n - 1]);

    // Check result of computation on the GPU
    float error = 0.0;
    static constexpr float tolerance = 1e-6f;
    for (size_t i = 0; i < n; i++) {
        const auto diff = abs(a - h_arr[i]);
        if (diff > tolerance)
            error += diff;
    }
    printf("total error: %f\n", error);

    return 0;
}
