/*
 * Managed memory basics
 *
 * Task is to:
 * - Allocate and free managed memory 
 * - trace with 'rocprof --hip-trace'. Do you see any memory copy call?
 */

#include <stdio.h>
#include <stdlib.h>
#include "../../error_checking.hpp"

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
    if (i<10) printf("%f ", a[i]);
    if (error > max_err) max_err = error;
  }
  printf("\n");
  return max_err;
}

int main() {
  const size_t N = 1<<9;

  constexpr int blocksize = 256;
  constexpr int gridsize =(N-1+blocksize)/blocksize;
  constexpr size_t N_bytes = N*sizeof(float);

  float *a;

  // TODO: allocate hip managed memory for pointer a

  memset(a, 0, N_bytes);

  kernel<<<gridsize, blocksize>>>(a, N);
  HIP_ERRCHK(hipGetLastError());
  
  // TODO: Synchronize

  printf("error: %f\n", max_error(a, N));


  // TODO: Free memory
}
