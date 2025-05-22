
/*
 * This code uses default stream
 * Task is 
 * - to place kernel_{a,b,c} to separate streams and execute kernels asynchronously
 * - validate that kernels execute concurrently with `srun ... rocprof --hip-trace ./02-fourstreams.cpp`
 */
#include <stdio.h>
#include <stdlib.h>
#include "../../error_checking.hpp"

__device__ int factorial(int m) {
  int retval(1);
  for (int k = 2; k<=m; ++k) {
    retval *= k;
  }
  return retval;
}

// GPU kernel definition
__global__ void kernel_a(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  constexpr int Q = 10;

  if (tid < n) {
    for (int l = 0; l < Q; ++l) {
      float x = (float)tid;
      float s = sinf(x+l);
      float c = cosf(x+l);
      a[tid] = a[tid] + sqrtf(s*s+c*c);
    }
  }
}

__global__ void kernel_b(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  constexpr int Q = 10;
  if (tid < n) {
    for (int i=tid-Q;i<=tid+Q;++i) {
      a[tid] += pow(16,tid-i) * (4/(8*i+1) - 2/(8*i+4) - 1/(8*i+5) - 1/(8*i+6));
    }
  }
}

__global__ void kernel_c(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  constexpr int Q = 20;
  if (tid < n) {
    float x = float(20)*float(tid)/n;
    for (size_t m = 0; m < Q; ++m) {
      a[tid] += pow(-1,m)*(pow(x/2, 2*m+1)/factorial(m))/factorial(m+1);
    }
  }
}


int main() {
  constexpr size_t N = 1<<12; // 4096 items

  constexpr int blocksize = 256;
  constexpr int gridsize =(N-1+blocksize)/blocksize;
  constexpr size_t N_bytes = N*sizeof(float);

  // Host & device pointers
  float *a; float *d_a;
  float *b; float *d_b;
  float *c; float *d_c;

  // Host allocations
  a = (float*) malloc(N_bytes);
  b = (float*) malloc(N_bytes);
  c = (float*) malloc(N_bytes);

  hipStream_t stream_a; 
  hipStream_t stream_b; 
  hipStream_t stream_c; 

  hipStreamCreate(&stream_a);
  hipStreamCreate(&stream_b);
  hipStreamCreate(&stream_c);

  // Device allocations
  HIP_ERRCHK(hipMalloc((void**)&d_a, N_bytes));
  HIP_ERRCHK(hipMalloc((void**)&d_b, N_bytes));
  HIP_ERRCHK(hipMalloc((void**)&d_c, N_bytes));

  // Set host memory
  memset(a, 0, N_bytes); memset(b, 0, N_bytes); memset(c, 0, N_bytes);

  // Send data (zeros here) to device
  HIP_ERRCHK(hipMemcpy(d_a, a, N_bytes, hipMemcpyDefault));
  HIP_ERRCHK(hipMemcpy(d_b, a, N_bytes, hipMemcpyDefault));
  HIP_ERRCHK(hipMemcpy(d_c, a, N_bytes, hipMemcpyDefault));

  // Execute kernels in sequence
  kernel_a<<<gridsize, blocksize,0,stream_a>>>(d_a, N);
  HIP_ERRCHK(hipGetLastError());

  kernel_b<<<gridsize, blocksize,0,stream_b>>>(d_b, N);
  HIP_ERRCHK(hipGetLastError());

  kernel_c<<<gridsize, blocksize,0,stream_c>>>(d_c, N);
  HIP_ERRCHK(hipGetLastError());


  // Copy results back
  HIP_ERRCHK(hipMemcpy(a, d_a, N_bytes, hipMemcpyDefault));
  HIP_ERRCHK(hipMemcpy(b, d_b, N_bytes, hipMemcpyDefault));
  HIP_ERRCHK(hipMemcpy(c, d_c, N_bytes, hipMemcpyDefault));

  // Free device and host memory allocations
  HIP_ERRCHK(hipFree(d_a));
  HIP_ERRCHK(hipFree(d_b));
  HIP_ERRCHK(hipFree(d_c));
  free(a);
  free(b);
  free(c);

}
