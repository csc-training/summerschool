#include "hip/hip_runtime.h"
#include <stdio.h>

#define USE_PINNED_HOST_MEM 1

__global__ void kernel(float *a, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  float x = (float)i;
  float s = sinf(x); 
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s*s+c*c);
}

float maxError(float *a, int n) 
{
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const int blockSize = 256, nStreams = 4;
  const int n = 4 * 1024 * blockSize * nStreams;
  const int bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, devId);
  printf("Device : %s\n", prop.name);
  hipSetDevice(devId);
  
  // Allocate pinned host memory and device memory
  float *a, *d_a;

  #if USE_PINNED_HOST_MEM == 1
    hipHostMalloc((void**)&a, bytes);      // host pinned
  #else
    a=(float *)malloc(n * sizeof(float));  // host pageable
  #endif
  hipMalloc((void**)&d_a, bytes);          // device pinned

  float duration; 
  
  // Create events
  hipEvent_t startEvent, stopEvent;
  hipEventCreate(&startEvent);
  hipEventCreate(&stopEvent);
  
  // Sequential transfer and execute
  memset(a, 0, bytes);
  hipEventRecord(startEvent,0);
  hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(kernel, n/blockSize, blockSize, 0, 0, d_a, 0);
  hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost);
  hipEventRecord(stopEvent, 0);
  hipEventSynchronize(stopEvent);
  hipEventElapsedTime(&duration, startEvent, stopEvent);
  printf("Duration for sequential transfer and execute: %f (ms)\n", duration);
  printf("  max error: %e\n", maxError(a, n));

  // TODO: Create `nStream` streams here (nStream is defined already to be 4)

  // Async case 1: loop over {kernel}
  {
    memset(a, 0, bytes);
    hipEventRecord(startEvent,0);
    hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice);
  
    // TODO: loop over nStreams and split the case 1 kernel for 4 kernel calls (one for each stream)
    // TODO: Each stream should handle 1/nStreams of work
  
    hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost);
    hipEventRecord(stopEvent, 0);
    hipEventSynchronize(stopEvent);
    hipEventElapsedTime(&duration, startEvent, stopEvent);
    printf("Case 1 - Duration for asynchronous kernels: %f (ms)\n", duration);
    printf("  max error: %e\n", maxError(a, n));
  }

  // Async case 2: loop over {async copy, kernel, async copy}
{
    memset(a, 0, bytes);
  
    // TODO: Same as case 1, except use asynchronous memcopies 
    // TODO: Here split also the memcopies for each stream 
    // TODO: Ie, looping over {async copy, kernel, async copy}
    // TODO: You should also add the missing hipEvent function calls (cf. case 1)

    printf("Case 2 - Duration for asynchronous transfer+kernels: %f (ms)\n", duration);
    printf("  max error: %e\n", maxError(a, n));
  }

  // Async case 3: loop over {async copy}, loop over {kernel}, loop over {async copy}
  {
    memset(a, 0, bytes);
    // TODO: Same as case 2, except create 3 loops over the streams
    // TODO: Ie, loop 1 {async copy} loop 2 {kernel}. loop 3 {async copy}
    // TODO: You should also add the missing hipEvent function calls (cf. case 1)
    
    printf("Case 3 - Duration for asynchronous transfer+kernels: %f (ms)\n", duration);
    printf("  max error: %e\n", maxError(a, n));
  }

  // Clean memory
  hipEventDestroy(startEvent);
  hipEventDestroy(stopEvent);
  hipFree(d_a);
  hipHostFree(a);

  // TODO: Destroy streams here

  return 0;
}
