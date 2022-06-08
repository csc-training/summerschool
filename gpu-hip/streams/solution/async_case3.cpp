#include "hip/hip_runtime.h"
#include <stdio.h>

inline hipError_t hipCheck(hipError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "HIP Runtime Error: %s\n", hipGetErrorString(result));
    assert(result == hipSuccess);
  }
#endif
  return result;
}

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
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  hipDeviceProp_t prop;
  hipCheck( hipGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  hipCheck( hipSetDevice(devId) );
  
  // Allocate pinned host memory and device memory
  float *a, *d_a;
  hipCheck( hipHostMalloc((void**)&a, bytes) );      
  hipCheck( hipMalloc((void**)&d_a, bytes) ); 

  float duration; 
  
  // create events and streams
  hipEvent_t startEvent, stopEvent;
  hipStream_t stream[nStreams];
  hipCheck( hipEventCreate(&startEvent) );
  hipCheck( hipEventCreate(&stopEvent) );
  for (int i = 0; i < nStreams; ++i)
    hipCheck( hipStreamCreate(&stream[i]) );
  
  // sequential transfer and execute
  memset(a, 0, bytes);
  hipCheck( hipEventRecord(startEvent,0) );
  hipCheck( hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice) );
  hipLaunchKernelGGL(kernel, n/blockSize, blockSize, 0, 0, d_a, 0);
  hipCheck( hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost) );
  hipCheck( hipEventRecord(stopEvent, 0) );
  hipCheck( hipEventSynchronize(stopEvent) );
  hipCheck( hipEventElapsedTime(&duration, startEvent, stopEvent) );
  printf("Duration for sequential transfer and execute (ms): %f\n", duration);
  printf("  max error: %e\n", maxError(a, n));

 // Async kernels case 1
  memset(a, 0, bytes);
  hipCheck( hipEventRecord(startEvent,0) );
  hipCheck( hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice) );
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    hipLaunchKernelGGL(kernel, streamSize/blockSize, blockSize, 0, stream[i], d_a, offset);
  }
  hipCheck( hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost) );
  hipCheck( hipEventRecord(stopEvent, 0) );
  hipCheck( hipEventSynchronize(stopEvent) );
  hipCheck( hipEventElapsedTime(&duration, startEvent, stopEvent) );
  printf("Duration for asynchronous kernels, execute (ms): %f\n", duration);
  printf("  max error: %e\n", maxError(a, n));


  // asynchronous case 2: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  hipCheck( hipEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    hipCheck( hipMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, hipMemcpyHostToDevice, 
                               stream[i]) );
    hipLaunchKernelGGL(kernel, streamSize/blockSize, blockSize, 0, stream[i], d_a, offset);
    hipCheck( hipMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, hipMemcpyDeviceToHost,
                               stream[i]) );
  }
  hipCheck( hipEventRecord(stopEvent, 0) );
  hipCheck( hipEventSynchronize(stopEvent) );
  hipCheck( hipEventElapsedTime(&duration, startEvent, stopEvent) );
  printf("Duration for asynchronous transfer case 1 and execute (ms): %f %d\n", duration, streamSize);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous case 3: loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  hipCheck( hipEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    hipCheck( hipMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, hipMemcpyHostToDevice,
                               stream[i]) );
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    hipLaunchKernelGGL(kernel, streamSize/blockSize, blockSize, 0, stream[i], d_a, offset);
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    hipCheck( hipMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, hipMemcpyDeviceToHost,
                               stream[i]) );
  }
  hipCheck( hipEventRecord(stopEvent, 0) );
  hipCheck( hipEventSynchronize(stopEvent) );
  hipCheck( hipEventElapsedTime(&duration, startEvent, stopEvent) );
  printf("Duration for asynchronous transfer case 2 and execute (ms): %f %d\n", duration,streamBytes);
  printf("  max error: %e\n", maxError(a, n));

  // Clean memory
  hipCheck( hipEventDestroy(startEvent) );
  hipCheck( hipEventDestroy(stopEvent) );
  for (int i = 0; i < nStreams; ++i)
    hipCheck( hipStreamDestroy(stream[i]) );
  hipFree(d_a);
  hipHostFree(a);

  return 0;
}
