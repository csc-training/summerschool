// SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

/*
 * This code in its current form uses the default stream
 * Task:
 * - Place kernel_{a,b,c} in separate streams and execute them asynchronously
 * - Validate that kernels are executing concurrently with `run_tue ... rocprof --hip-trace ./<your->
 *   - Open chromium url chrome://tracing or https://ui.perfetto.dev, open file "results.json"
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "error_checking.hpp"

// GPU kernel definition
__global__ void kernel_a(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n) {
    float x = tid;

    for (int i = 0; i < 30; ++i) {
      x = sinf(x) + cosf(x);
    }

    a[tid] = x;
  }
}

__global__ void kernel_b(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n) {
    float x = tid;

    for (int i = 0; i < 30; ++i) {
      x = sqrtf(x + 1.0f);
    }

    a[tid] = x;
  }
}

__global__ void kernel_c(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n) {
    float x = tid;

    for (int i = 0; i < 30; ++i) {
      x = logf(x + 1.0f);
    }

    a[tid] = x;
  }
}


int main() {
  constexpr size_t N = 1<<26; // ~68 million items

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

  #error create three separate streams

  // Device allocations
  HIP_ERRCHK(hipMalloc((void**)&d_a, N_bytes));
  HIP_ERRCHK(hipMalloc((void**)&d_b, N_bytes));
  HIP_ERRCHK(hipMalloc((void**)&d_c, N_bytes));

  // warmup
  kernel_c<<<gridsize, blocksize>>>(d_a, N);
  HIP_ERRCHK(hipMemcpy(a, d_a, N_bytes/100, hipMemcpyDefault));
  HIP_ERRCHK(hipDeviceSynchronize());

  // Execute kernels in sequence
  #error Launch each kernel in a different stream
  kernel_a<<<gridsize, blocksize,0,0>>>(d_a, N);
  HIP_ERRCHK(hipGetLastError());

  kernel_b<<<gridsize, blocksize,0,0>>>(d_b, N);
  HIP_ERRCHK(hipGetLastError());

  kernel_c<<<gridsize, blocksize,0,0>>>(d_c, N);
  HIP_ERRCHK(hipGetLastError());

  // Copy results back (in the default stream with hipMemCpy)
  #error synchronize the host with stream A, before copying d_A back
  HIP_ERRCHK(hipMemcpy(a, d_a, N_bytes, hipMemcpyDefault));
  #error synchronize the host with stream B, before copying d_B back
  HIP_ERRCHK(hipMemcpy(b, d_b, N_bytes, hipMemcpyDefault));
  #error synchronize the host with stream C, before copying d_C back
  HIP_ERRCHK(hipMemcpy(c, d_c, N_bytes, hipMemcpyDefault));

  for (int i = 0; i < 10; ++i) printf("%f ", a[i]);
  printf("\n");

  for (int i = 0; i < 10; ++i) printf("%f ", b[i]);
  printf("\n");

  for (int i = 0; i < 10; ++i) printf("%f ", c[i]);
  printf("\n");
  // Free device and host memory allocations
  HIP_ERRCHK(hipFree(d_a));
  HIP_ERRCHK(hipFree(d_b));
  HIP_ERRCHK(hipFree(d_c));

  #error destroy all streams

  free(a);
  free(b);
  free(c);
}
