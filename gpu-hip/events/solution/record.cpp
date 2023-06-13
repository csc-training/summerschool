#include <cstdio>
#include <time.h>
#include <hip/hip_runtime.h>

/* A simple GPU kernel definition */
__global__ void kernel(int *d_a, int n_total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_total)
    d_a[idx] = idx;
}

/* The main function */
int main(){
  
  // Problem size
  constexpr int n_total = 4194304; // pow(2, 22);

  // Device grid sizes
  constexpr int blocksize = 256;
  constexpr int gridsize = (n_total - 1 + blocksize) / blocksize;

  // Allocate host and device memory
  int *a, *d_a;
  const int bytes = n_total * sizeof(int);
  hipHostMalloc((void**)&a, bytes); // host pinned
  hipMalloc((void**)&d_a, bytes);   // device pinned

  // Create events
  hipEvent_t start_kernel_event;
  hipEventCreate(&start_kernel_event);
  hipEvent_t start_d2h_event;
  hipEventCreate(&start_d2h_event);
  hipEvent_t stop_event;
  hipEventCreate(&stop_event);

  // Create stream
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Start timed GPU kernel
  clock_t start_kernel_clock = clock();
  hipEventRecord(start_kernel_event, stream);
    for(int i=0;i<100;i++)
  kernel<<<gridsize, blocksize, 0, stream>>>(d_a, n_total);

  // Start timed device-to-host memcopy
  clock_t start_d2h_clock = clock();
  hipEventRecord(start_d2h_event, stream);
    for(int i=0;i<100;i++)
  hipMemcpyAsync(a, d_a, bytes, hipMemcpyDeviceToHost, stream);

  // Stop timing
  clock_t stop_clock = clock();
  hipEventRecord(stop_event, stream);
  hipEventSynchronize(stop_event);

  // Exctract elapsed timings from event recordings
  float time_kernel, time_d2h, time_total;
  hipEventElapsedTime(&time_kernel, start_kernel_event, start_d2h_event);
  hipEventElapsedTime(&time_d2h, start_d2h_event, stop_event);
  hipEventElapsedTime(&time_total, start_kernel_event, stop_event);

  // Check that the results are right
  int error = 0;
  for(int i = 0; i < n_total; ++i){
    if(a[i] != i)
      error = 1;
  }

  // Print results
  if(error)
    printf("Results are incorrect!\n");
  else
    printf("Results are correct!\n");

  // Print event timings
  printf("Event timings:\n");
  printf("  %.3f ms - kernel\n", time_kernel);
  printf("  %.3f ms - device to host copy\n", time_d2h);
  printf("  %.3f ms - total time\n", time_total);

  // Print clock timings
  printf("clock_t timings:\n");
  printf("  %.3f ms - kernel\n", 1e3 * (double)(start_d2h_clock - start_kernel_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - device to host copy\n", 1e3 * (double)(stop_clock - start_d2h_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - total time\n", 1e3 * (double)(stop_clock - start_kernel_clock) / CLOCKS_PER_SEC);

  // Destroy Stream
  hipStreamDestroy(stream);

  // Destroy events
  hipEventDestroy(start_kernel_event);
  hipEventDestroy(start_d2h_event);
  hipEventDestroy(stop_event);

  // Deallocations
  hipFree(d_a); // Device
  hipHostFree(a); // Host
}
