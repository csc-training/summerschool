#include <stdio.h>
/* #include <omp.h> */
#include <iostream>
#include "error_checking.hpp"

__global__ void sum_vecs(float* a, float* b, float* c, size_t n) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t k = tid; k < n; k += stride) {
      c[k] = a[k] +  b[k];
    }
}

void hip_sum(float* a, float* b, float* c, size_t n) {
  sum_vecs<<< dim3((n + 1024 -1)/1024, 1,1), dim3(1024, 1, 1), 0, 0>>>
    (a, b, c, n);
  HIP_ERRCHK(hipGetLastError());
}

/* extern "C" { */
/*   void hip_sum(float* a, float* b, float* c, size_t n) { */
/*     hip_sum_(a,b,c,n); */
/*   } */
/* } */
