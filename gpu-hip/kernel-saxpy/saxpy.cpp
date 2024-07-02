#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

// TODO: add a device kernel that calculates y = a * x + y
__global__ void saxpy_(int n, float a, float *x, float *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (; tid < n; tid += stride) {
        y[tid] += a * x[tid];
    }
}

__global__ void get_indices_(int* output){
    *output = threadIdx.x;
}

int main(void)
{
    int i;
    const int n = 10000;
    float a = 3.4;
    float x[n], y[n], y_ref[n];
    float *x_, *y_;
    int threadIdx_x;
    int* threadIdx_x_;

    // initialise data and calculate reference values on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        y_ref[i] = a * x[i] + y[i];
    }

    // TODO: allocate vectors x_ and y_ on the GPU
    hipMalloc(&x_, sizeof(float) * n);
    hipMalloc(&y_, sizeof(float) * n);
    hipMalloc(&threadIdx_x_, sizeof(int));

    // TODO: copy initial values from CPU to GPU (x -> x_ and y -> y_)
    hipMemcpy(x_, x, sizeof(float) * n, hipMemcpyHostToDevice);
    hipMemcpy(y_, y, sizeof(float) * n, hipMemcpyHostToDevice);

    // TODO: define grid dimensions
    dim3 blocks(32);
    dim3 threads(256);

    // TODO: launch the device kernel
    // Grid dimensions are obligatory, shmem, and stream are optional (set to 0)
    //hipLaunchKernelGGL(saxpy_, blocks, threads, 0, 0, n, a, x_, y_);
    saxpy_<<<blocks, threads, 0, 0>>>(n, a, x_, y_);
    get_indices_<<<blocks, threads, 0, 0>>>(threadIdx_x_);

    // TODO: copy results back to CPU (y_ -> y)
    hipMemcpy(y, y_, sizeof(float) * n, hipMemcpyDeviceToHost);
    hipMemcpy(&threadIdx_x, threadIdx_x_, sizeof(int), hipMemcpyDeviceToHost);
    printf("\nthreadIdx.x: %d \n", threadIdx_x);
    //printf("\nthreadIdx.x: %d, blockIdx.x: %d, blockDim.x: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
    //printf("threadIdx.y: %d, blockIdx.y: %d, blockDim.y: %d\n", threadIdx.y, blockIdx.y, blockDim.y);

    // confirm that results are correct
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[n-2], y_ref[n-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);
    
    float error = 0.0;
    float tolerance = 1e-6;
    float diff;
    for (i=0; i < n; i++) {
        diff = abs(y_ref[i] - y[i]);
        if (diff > tolerance)
            error += diff;
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42)\n", y_ref[42]);
    printf("     result: %f at (42)\n", y[42]);

    return 0;
}
