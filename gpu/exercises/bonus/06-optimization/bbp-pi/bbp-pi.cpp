/*
 * This code calculates digits of PI in hexadecimal using
 * Bailey-Borwein-Plouffe formula.
 *
 * This code is quite naively written in terms of GPU performance.
 * Your task is optimize the code. No solution is given.
 *
 * See end for spoilers/ideas.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../../../error_checking.hpp"

__device__ int ipow_mod(int m, int n, int mod) {
  int ret(1);
  while ( n!=0) {
    if (n%2) ret = (ret*m) % mod;
    m = (m*m) % mod;
    n >>= 1;
  }
  return ret;
}

// This function looks suspicious
__device__ float S(int j,int n) {
  float s = 0.0;
  __shared__ int ipow_tbl[1<<11];
  for (int k = 0; k<n;++k) s += ipow_mod(16, n-k, 8*k+j) / (8.0*k+j);
  float t = 0.0;
  int k = n;
  for (k = n; k< n+3; ++k) t += pow(16,n-k) / (8*k+j);

  float r = s+t;
  return s+t-(int)(s+t);
}

__global__ void hex_pi(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  float S1, S2, S3, S4;

  if (tid < n) {
      S1 = S(1,tid);
      S2 = S(4,tid);
      S3 = S(5,tid);
      S4 = S(6,tid),
    a[tid] = 4*S1-2*S2-S3-S4;
    a[tid] = (a[tid] > 0) ? (a[tid] - (int)a[tid]) : (a[tid]-(int)a[tid] + 1);
    a[tid] = a[tid]*16;
  }
}

int main() {
  const size_t N = 1<<17;

  constexpr int blocksize = 256;
  constexpr int gridsize =(N-1+blocksize)/blocksize;
  constexpr size_t N_bytes = N*sizeof(float);

  float *a;
  float *d_a;

  hipEvent_t start, stop;

  a = (float*) malloc(N_bytes);
  HIP_ERRCHK(hipMalloc((void**)&d_a, N_bytes));

  // warmup
  hex_pi<<<gridsize, blocksize,0,0>>>(d_a, 1000);

  HIP_ERRCHK(hipEventCreate(&start));
  HIP_ERRCHK(hipEventCreate(&stop));

  memset(a, 0, N_bytes);

  HIP_ERRCHK(hipMemcpy(d_a, a, N_bytes, hipMemcpyHostToDevice));
  HIP_ERRCHK(hipEventRecord(start, 0));
  hex_pi<<<gridsize, blocksize,0,0>>>(d_a, N);
  HIP_ERRCHK(hipGetLastError());
  HIP_ERRCHK(hipEventRecord(stop, 0));
  HIP_ERRCHK(hipMemcpy(a, d_a, N_bytes, hipMemcpyDeviceToHost));

  float duration; 
  HIP_ERRCHK(hipEventElapsedTime(&duration, start,stop));

  std::cout << std::hex;
  std::cout << 3 <<".";
  for (size_t k = 0; k < 100; ++k) {
    size_t digit = floor(a[k]);
    std::cout <<  digit;
  }
  std::cout << std::endl << std::dec;
  std::cout << N << " digits in: " << duration << " ms (" << N / duration <<  " digit/ms)\n";
  HIP_ERRCHK(hipFree(d_a));
  free(a);

}












/* Spoilers/ideas
 * - Threads get assigned different amounts of iterations (`float S()` function)
 * - Same numbers within blocks are calculated over and over (`float S()`)
 * - Lot of branching
 * - 
 */
