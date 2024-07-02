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
  //#error create the required timing events here
  hipEvent_t start_kernel_event, start_d2h_event, stop_event;
  hipEventCreate(&start_kernel_event);
  hipEventCreate(&start_d2h_event);
  hipEventCreate(&stop_event);

  // Create stream
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Start timed GPU kernel and device-to-host copy

  // #error record the events somewhere across the below lines of code
  // #error such that you can get the timing for the kernel, the
  // #error memory copy, and the total combined time of these

  // Start timed GPU kernel
  clock_t start_kernel_clock = clock();
  hipEventRecord(start_kernel_event, stream);  // Capture in event the contents of stream at the time of this call.
  kernel<<<gridsize, blocksize, 0, stream>>>(d_a, n_total);  // Launch kernel.

  // Start timed device-to-host memcopy
  clock_t start_d2h_clock = clock();
  hipEventRecord(start_d2h_event, stream);
  hipMemcpyAsync(a, d_a, bytes, hipMemcpyDeviceToHost, stream);  // Asynchronous D2H copy (d_a to a) on stream. The host can now perform tasks independently of D2H copy.
  //hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost);  // D2H copy (d_a to a) on stream.

  // Stop timing
  clock_t stop_clock = clock();
  hipEventRecord(stop_event, stream);
  hipEventSynchronize(stop_event);  // Wait for event to complete
  
  //hipStreamSynchronize(stream);  // Synchronize stream


  // Exctract elapsed timings from event recordings
  // #error get the elapsed time from the timing events
  float time_kernel, time_d2h, time_total;
  hipEventElapsedTime(&time_kernel, start_kernel_event, start_d2h_event);  // Compute the elapsed time (start_d2h_event-start_kernel_event) in milliseconds and store it in time_kernel.
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
  // #error print event timings here
  printf("  %.3f ms - kernel\n", time_kernel);
  printf("  %.3f ms - device to host copy\n", time_d2h);
  printf("  %.3f ms - total time\n\n", time_total);

  // Print clock timings
  printf("clock_t timings:\n");
  printf("  %.3f ms - kernel\n", 1e3 * (double)(start_d2h_clock - start_kernel_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - device to host copy\n", 1e3 * (double)(stop_clock - start_d2h_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - total time\n", 1e3 * (double)(stop_clock - start_kernel_clock) / CLOCKS_PER_SEC);

  // Destroy Stream
  hipStreamDestroy(stream);

  // Destroy events
  // #error destroy events here
  hipEventDestroy(start_kernel_event);  //Destroy event object
  hipEventDestroy(start_d2h_event);  //Destroy event object
  hipEventDestroy(stop_event);  //Destroy event object

  // Deallocations
  hipFree(d_a); // Device
  hipHostFree(a); // Host
}
