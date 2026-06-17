// SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

/*
 * This exercise builds upon the solution of
 * 03-streams-asyncmemcpy.
 * Task is to:
 * - Initialize six events
 *  - start_a, start_b, start_c
 *  - end_a, end_b, end_c
 * - Record execution start and end of each kernel in the program
 * - Print out the results
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

  // Timing
  float t_kernel_a_ms;
  float t_kernel_b_ms;
  float t_kernel_c_ms;

  // Host allocations
  HIP_ERRCHK(hipHostMalloc((void**)&a, N_bytes));
  HIP_ERRCHK(hipHostMalloc((void**)&b, N_bytes));
  HIP_ERRCHK(hipHostMalloc((void**)&c, N_bytes));

  hipStream_t stream_a;
  hipStream_t stream_b;
  hipStream_t stream_c;

  HIP_ERRCHK(hipStreamCreate(&stream_a));
  HIP_ERRCHK(hipStreamCreate(&stream_b));
  HIP_ERRCHK(hipStreamCreate(&stream_c));

  // Make events
  hipEvent_t start_a, end_a, start_b, end_b, start_c, end_c;
  hipEvent_t* all_events[6] = {&start_a, &end_a, &start_b, &end_b, &start_c, &end_c};

  // Create all events
  for (int i = 0; i < 6; ++i) HIP_ERRCHK(hipEventCreate(all_events[i]));

  // Device allocations
  HIP_ERRCHK(hipMalloc((void**)&d_a, N_bytes));
  HIP_ERRCHK(hipMalloc((void**)&d_b, N_bytes));
  HIP_ERRCHK(hipMalloc((void**)&d_c, N_bytes));

  // warmup
  kernel_c<<<gridsize, blocksize>>>(d_a, N);
  HIP_ERRCHK(hipMemcpy(a, d_a, N_bytes/100, hipMemcpyDefault));
  HIP_ERRCHK(hipDeviceSynchronize());
  // warmup ends

  // Record timing events around each kernel launch
  HIP_ERRCHK(hipEventRecord(start_a, stream_a));
  kernel_a<<<gridsize, blocksize,0,stream_a>>>(d_a, N);
  HIP_ERRCHK(hipGetLastError());
  HIP_ERRCHK(hipEventRecord(end_a, stream_a));

  HIP_ERRCHK(hipEventRecord(start_b, stream_b));
  kernel_b<<<gridsize, blocksize,0,stream_b>>>(d_b, N);
  HIP_ERRCHK(hipGetLastError());
  HIP_ERRCHK(hipEventRecord(end_b, stream_b));

  HIP_ERRCHK(hipEventRecord(start_c, stream_c));
  kernel_c<<<gridsize, blocksize,0,stream_c>>>(d_c, N);
  HIP_ERRCHK(hipGetLastError());
  HIP_ERRCHK(hipEventRecord(end_c, stream_c));

  // Copy results back
  HIP_ERRCHK(hipMemcpyAsync(a, d_a, N_bytes, hipMemcpyDefault, stream_a));
  HIP_ERRCHK(hipMemcpyAsync(b, d_b, N_bytes, hipMemcpyDefault, stream_b));
  HIP_ERRCHK(hipMemcpyAsync(c, d_c, N_bytes, hipMemcpyDefault, stream_c));

  // Synchronize each stream with host before printing out results
  HIP_ERRCHK(hipStreamSynchronize(stream_a));
  for (int i = 0; i < 20; ++i) printf("%f ", a[i]);
  printf("\n");

  HIP_ERRCHK(hipStreamSynchronize(stream_b));
  for (int i = 0; i < 20; ++i) printf("%f ", b[i]);
  printf("\n");

  HIP_ERRCHK(hipStreamSynchronize(stream_c));
  for (int i = 0; i < 20; ++i) printf("%f ", c[i]);
  printf("\n");

  HIP_ERRCHK(hipEventElapsedTime(&t_kernel_a_ms, start_a, end_a));
  // Print in milliseconds
  printf("kernel_a time: %f ms\n", t_kernel_a_ms);

  HIP_ERRCHK(hipEventElapsedTime(&t_kernel_b_ms, start_b, end_b));
  printf("kernel_b time: %f ms\n", t_kernel_b_ms);

  HIP_ERRCHK(hipEventElapsedTime(&t_kernel_c_ms, start_c, end_c));
  printf("kernel_c time: %f ms\n", t_kernel_c_ms);

  for (int i=0; i<6; ++i) HIP_ERRCHK(hipEventDestroy(*all_events[i]));

  // Free device and host memory allocations
  HIP_ERRCHK(hipFree(d_a));
  HIP_ERRCHK(hipFree(d_b));
  HIP_ERRCHK(hipFree(d_c));

  HIP_ERRCHK(hipStreamDestroy(stream_a));
  HIP_ERRCHK(hipStreamDestroy(stream_b));
  HIP_ERRCHK(hipStreamDestroy(stream_c));

  HIP_ERRCHK(hipHostFree(a));
  HIP_ERRCHK(hipHostFree(b));
  HIP_ERRCHK(hipHostFree(c));
}
