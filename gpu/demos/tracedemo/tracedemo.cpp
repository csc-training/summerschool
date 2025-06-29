
/*
 * This code is built on upon solution of 02-asynchkernel.cpp
 * Task is 
 * - measure kernel execution time for kernel_b (with events)
 * - replace all hipStreamSynchronize calls with hipEventSynchronize
 * - execute kernel_b after kernel_a has been executed (use hipStreamWaitEvent)
 */

#include <stdio.h>
#include <stdlib.h>
#include "error_checking.hpp"

#include "helperfuns.h"

// GPU kernel definition
int main() {
  constexpr size_t N = 1<<10; // 4096 items

  constexpr int blocksize = 256;
  constexpr int gridsize =(N-1+blocksize)/blocksize;
  constexpr size_t N_bytes = N*sizeof(float);

  // Host & device pointers
  float *a; float *d_a;
  float *b; float *d_b;
  float *c; float *d_c;

  float t_kernel_b_ms;

  // Host allocations
  a = (float*) malloc(N_bytes);
  b = (float*) malloc(N_bytes);
  c = (float*) malloc(N_bytes);

  hipStream_t stream_a; 
  hipStream_t stream_b; 
  hipStream_t stream_c; 

  HIP_ERRCHK(hipStreamCreate(&stream_a));
  HIP_ERRCHK(hipStreamCreate(&stream_b));
  HIP_ERRCHK(hipStreamCreate(&stream_c));

  // Make events

  hipEvent_t start_a, end_a, start_b, end_b, start_c, end_c;
  hipEvent_t* all_events[6] = {&start_a, &end_a, &start_b, &end_b, &start_c, &end_c};

  for (int i = 0; i < 6; ++i) HIP_ERRCHK(hipEventCreate(all_events[i]));

  // Device allocations
  HIP_ERRCHK(hipMalloc((void**)&d_a, N_bytes));
  HIP_ERRCHK(hipMalloc((void**)&d_b, N_bytes));
  HIP_ERRCHK(hipMalloc((void**)&d_c, N_bytes));

  
  // warmup
  kernel_c<<<gridsize, blocksize>>>(d_a, N);
  HIP_ERRCHK(hipMemcpy(a, d_a, N_bytes/100, hipMemcpyDefault));
  HIP_ERRCHK(hipDeviceSynchronize());

  // Execute kernels in sequence
  HIP_ERRCHK(hipEventRecord(start_a));
  kernel_a<<<gridsize, blocksize,0,stream_a>>>(d_a, N);
  HIP_ERRCHK(hipGetLastError());
  HIP_ERRCHK(hipEventRecord(end_a));

  HIP_ERRCHK(hipEventRecord(start_b));
  kernel_b<<<gridsize, blocksize,0,stream_b>>>(d_b, N);
  HIP_ERRCHK(hipGetLastError());
  HIP_ERRCHK(hipEventRecord(end_b));

  HIP_ERRCHK(hipEventRecord(start_c));
  kernel_c<<<gridsize, blocksize,0,stream_c>>>(d_c, N);
  HIP_ERRCHK(hipGetLastError());
  HIP_ERRCHK(hipEventRecord(end_c));

  // Copy results back
  HIP_ERRCHK(hipMemcpyAsync(a, d_a, N_bytes, hipMemcpyDefault, stream_a));
  HIP_ERRCHK(hipMemcpyAsync(b, d_b, N_bytes, hipMemcpyDefault, stream_b));
  HIP_ERRCHK(hipMemcpyAsync(c, d_c, N_bytes, hipMemcpyDefault, stream_c));

  HIP_ERRCHK(hipEventSynchronize(end_a));
  for (int i = 0; i < 20; ++i) printf("%f ", a[i]);
  printf("\n");

  HIP_ERRCHK(hipEventSynchronize(end_b));
  for (int i = 0; i < 20; ++i) printf("%f ", b[i]);
  printf("\n");

  HIP_ERRCHK(hipEventSynchronize(end_c));
  for (int i = 0; i < 20; ++i) printf("%f ", c[i]);
  printf("\n");

  HIP_ERRCHK(hipEventElapsedTime(&t_kernel_b_ms, start_b, end_b));
  printf("kernel_b time: %f us\n", 1000*t_kernel_b_ms);

  // Free device and host memory allocations
  HIP_ERRCHK(hipFree(d_a));
  HIP_ERRCHK(hipFree(d_b));
  HIP_ERRCHK(hipFree(d_c));

  HIP_ERRCHK(hipStreamDestroy(stream_a));
  HIP_ERRCHK(hipStreamDestroy(stream_b));
  HIP_ERRCHK(hipStreamDestroy(stream_c));

  for (int i=0; i<6; ++i) HIP_ERRCHK(hipEventDestroy(*all_events[i]));

  free(a);
  free(b);
  free(c);

}
