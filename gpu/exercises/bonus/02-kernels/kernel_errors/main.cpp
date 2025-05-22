#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>

#include "../../../error_checking.hpp"

__global__ void fill(size_t n, float a, float *arr) {
    const size_t tid = threadIdx.x + blockDim.y * blockDim.x;
    const size_t stride = blockDim.x * gridDim.z;

    assert(threadIdx.y == 0 && "This kernel should be run with a 1D "
                               "configuration, change the launch parameters");
    assert(threadIdx.z == 0 && "This kernel should be run with a 1D "
                               "configuration, change the launch parameters");
    assert(blockIdx.y == 0 && "This kernel should be run with a 1D "
                              "configuration, change the launch parameters");
    assert(blockIdx.z == 0 && "This kernel should be run with a 1D "
                              "configuration, change the launch parameters");

    assert(tid < blockDim.x * gridDim.x &&
           "tid should be less than total number of threads");
    assert(blockIdx.x * blockDim.x <= tid &&
           "tid should be larger than or equal to the number of threads before "
           "this block");
    assert(tid < blockIdx.x * blockDim.x + blockDim.x &&
           "tid should be less than the number of threads before + number of "
           "threads in this block");
    assert(tid < stride && "tid should be smaller than the stride");

    for (size_t i = tid; i < n; i += stride) {
        arr[i] = a;
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

    dim3 blocks(10, 2, 0);
    dim3 threads(2048, 1, 32);

    LAUNCH_KERNEL(fill, blocks, threads, 0, 0, n, a,
                  static_cast<float *>(d_arr));

    // Copy results back to CPU
    HIP_ERRCHK(hipMemcpy(static_cast<void *>(h_arr.data()), d_arr, num_bytes,
                         hipMemcpyDefault));

    // Free device memory
    HIP_ERRCHK(hipFree(d_arr));

    printf("ref value: %f, result: %f %f %f %f ... %f %f\n", a, h_arr[0],
           h_arr[1], h_arr[2], h_arr[3], h_arr[n - 2], h_arr[n - 1]);

    // Check result of computation on the GPU
    float error = 0.0f;
    static constexpr float tolerance = 1e-6f;
    for (size_t i = 0; i < n; i++) {
        const auto diff = abs(a - h_arr[i]);
        if (diff > tolerance)
            error += diff;
    }
    printf("total error: %f\n", error);

    if (error == 0.0f) {
        printf("Hooray, you fixed it \\o/!\n");
    } else {
        printf("Something's wrong, error for fill should be exactly 0.0f. "
               "Maybe compile without the '-DNDEBUG' flag and try again?\n");
    }

    return 0;
}
