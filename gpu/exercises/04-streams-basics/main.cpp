// SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

/*
 * This code in its current form uses the default stream
 * Task is to:
 *   - create a stream
 *   - copy memory to/from device with that stream
 *   - launch the provided kernel using that stream
 *   - copy data back to the host using the stream
 *   - destroy the stream
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "error_checking.hpp"

// GPU kernel definition
__global__ void kernel(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  if (tid < n) {
    float x = (float)tid;
    float s = sinf(x);
    float c = cosf(x);
    a[tid] = a[tid] + sqrtf(s*s+c*c);
  }
}

float max_error(float *a, int n)
{
  float max_err = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > max_err) max_err = error;
  }
  return max_err;
}

int main() {
  const size_t N = 1<<9;

  constexpr int blocksize = 256;
  constexpr int gridsize =(N-1+blocksize)/blocksize;
  constexpr size_t N_bytes = N*sizeof(float);

  float *a;
  float *d_a;

  #error declare a new stream variable with hipStream_t
  #error create the HIP stream with hipStreamCreate

  a = (float*) malloc(N_bytes);
  HIP_ERRCHK(hipMalloc((void**)&d_a, N_bytes));

  memset(a, 0, N_bytes);

  #error replace hipMemcpy with its Async counterpart, to copy data to the device using your stream
  HIP_ERRCHK(hipMemcpy(d_a, a, N_bytes, hipMemcpyHostToDevice));

  #error specify your stream at kernel launch
  kernel<<<gridsize, blocksize,0,0>>>(d_a, N);
  HIP_ERRCHK(hipGetLastError());

  #error replace hipMemcpy with its Async counterpart to copy data back to host using your stream
  HIP_ERRCHK(hipMemcpy(a, d_a, N_bytes, hipMemcpyDeviceToHost));

  #error synchronize the host with your stream, before continuing

  // Print out 10 indexes in the array a
  for (int i = 0; i < 10; i++)
    printf("%f ", a[i]);

  printf("\n");

  printf("error: %f", max_error(a, N));
  HIP_ERRCHK(hipFree(d_a));
  free(a);

  #error destroy the stream
}
