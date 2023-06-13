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
  #error create the required timing events here

  // Create stream
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Start timed GPU kernel and device-to-host copy
  #error record the events somewhere across the below lines of code
  #error such that you can get the timing for the kernel, the
  #error memory copy, and the total combined time of these
  clock_t start_kernel_clock = clock();
  kernel<<<gridsize, blocksize, 0, stream>>>(d_a, n_total);

  clock_t start_d2h_clock = clock();
  hipMemcpyAsync(a, d_a, bytes, hipMemcpyDeviceToHost, stream);

  clock_t stop_clock = clock();
  hipStreamSynchronize(stream);

  // Exctract elapsed timings from event recordings
  #error get the elapsed time from the timing events

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
  #error print event timings here

  // Print clock timings
  printf("clock_t timings:\n");
  printf("  %.3f ms - kernel\n", 1e3 * (double)(start_d2h_clock - start_kernel_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - device to host copy\n", 1e3 * (double)(stop_clock - start_d2h_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - total time\n", 1e3 * (double)(stop_clock - start_kernel_clock) / CLOCKS_PER_SEC);

  // Destroy Stream
  hipStreamDestroy(stream);

  // Destroy events
  #error destroy events here

  // Deallocations
  hipFree(d_a); // Device
  hipHostFree(a); // Host
}
