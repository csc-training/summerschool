#include "hip/hip_runtime.h"
#include <stdio.h>

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
// for the exercises case 1 and 2, define also what is the n per stream the bytes of each data transfer
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
 // a=(float *)malloc(n * sizeof(float));
  hipHostMalloc((void**)&a, bytes);      // host pinned
  hipMalloc((void**)&d_a, bytes); // device

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
  printf("Duration for sequential transfer and execute (ms): %f\n", duration);
  printf("  max error: %e\n", maxError(a, n));

// Add code for case 1 here

// Add code for case 2 here

// Add code for case 3 here

  // Clean memory
  hipEventDestroy(startEvent);
  hipEventDestroy(stopEvent);
  hipFree(d_a);
  hipHostFree(a);

  return 0;
}
