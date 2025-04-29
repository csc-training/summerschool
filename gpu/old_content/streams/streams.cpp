#include <stdio.h>
#include <string>
#include "hip/hip_runtime.h"

// Switch between pinned and pageable host memory
#define USE_PINNED_HOST_MEM 1

// GPU kernel definition
__global__ void kernel(float *a, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  for (; tid < n; tid += stride) {
    float x = (float)tid;
    float s = sinf(x);
    float c = cosf(x);
    a[tid] = a[tid] + sqrtf(s*s+c*c);
  }
}

// Calculate the max error
float max_error(float *a, int n)
{
  float max_err = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > max_err) max_err = error;
  }
  return max_err;
}

// Auxiliary function to print the results
void print_results(const std::string strategy, float *timing, float max_err, int n_streams)
{
  // Print the strategy and max error
  printf("%s (max error = %e):\n", strategy.c_str(), max_err);

  // Print the timings for individual streams
  for(int i = 0; i < n_streams; i++)
    printf("  stream[%d] time: %f ms\n", i, timing[i]);

  // Print the total time
  printf("  total time:     %f ms\n", timing[n_streams]);
}

// Case 0: Run memory copies and kernel sequentially
void case_0(hipEvent_t *start_event, hipEvent_t *stop_event, hipStream_t *stream, float *a, float *d_a, int n_streams, int gridsize, int blocksize, int n_total)
{
  // Record the start event for the total time
  hipEventRecord(start_event[0], 0);

  // Copy data to device, launch kernel, copy data back to host
  hipMemcpy(d_a, a, n_total * sizeof(float), hipMemcpyHostToDevice);
  kernel<<<gridsize, blocksize>>>(d_a, n_total);
  hipMemcpy(a, d_a, n_total * sizeof(float), hipMemcpyDeviceToHost);

  // Record the stop event for the total time
  hipEventRecord(stop_event[0], 0);

  // Synchronize with the event and capture timing between start_event and stop_event
  float timing[1];
  hipEventSynchronize(stop_event[0]);
  hipEventElapsedTime(&timing[0], start_event[0], stop_event[0]);

  // Print timings and the maximum error
  print_results("Case 0 - Duration for sequential transfers+kernel", timing, max_error(a, n_total), 0);
}

// Case 1: Run memory copies sequentially, distribute kernel for multiple streams
void case_1(hipEvent_t *start_event, hipEvent_t *stop_event, hipStream_t *stream, float *a, float *d_a, int n_streams, int gridsize, int blocksize, int n_total)
{
  // Calculate per-stream problem size
  int stream_size = n_total / n_streams;

  // Record the start event for the total time
  hipEventRecord(start_event[n_streams], 0);

  // Copy data to device
  hipMemcpy(d_a, a, n_total * sizeof(float), hipMemcpyHostToDevice);

  // Distribute kernel for 'n_streams' streams, and record each stream's timing
  #error loop over 'n_stream' and split the case 0 kernel for 4 kernel calls (one for each stream)
  #error each stream should handle 'n_total / n_streams' of work

  // Copy data back to host
  hipMemcpy(a, d_a, n_total * sizeof(float), hipMemcpyDeviceToHost);

  // Record the stop event for the total time
  hipEventRecord(stop_event[n_streams], 0);

  // Synchronize with the events and capture timings between start_events and stop_events
  float timing[n_streams + 1];
  for (int i = 0; i < n_streams + 1; ++i) {
    hipEventSynchronize(stop_event[i]);
    hipEventElapsedTime(&timing[i], start_event[i], stop_event[i]);
  }

  // Print timings and the maximum error
  print_results("Case 1 - Duration for asynchronous kernels", timing, max_error(a, n_total), n_streams);
}

// Case 2: Distribute the memory copies and the kernel for multiple streams (scheduling order 1)
void case_2(hipEvent_t *start_event, hipEvent_t *stop_event, hipStream_t *stream, float *a, float *d_a, int n_streams, int gridsize, int blocksize, int n_total)
{
  // Calculate per-stream problem size and byte size
  int stream_size = n_total / n_streams;
  int stream_bytes = stream_size * sizeof(float);

  // Record the start event for the total time
  #error record a start event for the total timing

  // Distribute memcopies and the kernel for 'n_streams' streams, and record each stream's timing
  #error Loop over 'n_stream'.
  #error Split the data copy from host to device into 'n_stream' asynchronous memcopies,
  #error one memcopy for each stream (make sure the memcopies are split evenly).
	#error Launch the kernel for each stream similarly to Case 1.
	#error Split the data copy from device to host into 'n_stream' asynchronous memcopies,
  #error one for each stream (make sure the memcopies are split evenly).
  #error Ie, looping over {async copy, kernel, async copy}.

  // Record the stop event for the total time
  #error record a stop event for the total timing

  // Synchronize with the events and capture timings between start_events and stop_events
  float timing[n_streams + 1];
  #error synchronize all each stop_event[i]
  #error get timings between each corresponding start_event[i] and stop_event[i]

  // Print timings and the maximum error
  print_results("Case 2 - Duration for asynchronous transfers+kernels", timing, max_error(a, n_total), n_streams);
}

// Case 3: Distribute the memory copies and the kernel for multiple streams (scheduling order 2)
void case_3(hipEvent_t *start_event, hipEvent_t *stop_event, hipStream_t *stream, float *a, float *d_a, int n_streams, int gridsize, int blocksize, int n_total)
{
  #error Copy the case 2 here
  #error Instead of doing the asynchronous memcopies and the kernel in the same loop,
  #error create a separate loop for each (3 loops in total, one for H-to-D memcopies,
  #error one for kernel calls, and one for D-to-H memcopies).
  #error Ie, loop 1 {async copy} loop 2 {kernel}. loop 3 {async copy}.

  // Print timings and the maximum error
  print_results("Case 3 - Duration for asynchronous transfers+kernels", timing, max_error(a, n_total), n_streams);
}

int main()
{
  // Problem size
  constexpr int n_total = 67108864; // 2**26

  // Device grid sizes
  constexpr int n_streams = 4;
  constexpr int blocksize = 256;
  // constexpr int gridsize = (n_total - 1 + blocksize) / blocksize;  // fails on LUMI with large n_total (July 2024)
  constexpr int gridsize = 512;

  // Allocate host and device memory
  float *a, *d_a;
  const int bytes = n_total * sizeof(float);

  #if USE_PINNED_HOST_MEM == 1
    hipHostMalloc((void**)&a, bytes);      // host pinned
  #else
    a=(float *)malloc(bytes);              // host pageable
  #endif
  hipMalloc((void**)&d_a, bytes);          // device pinned

  // Create events
  hipEvent_t start_event[n_streams + 1];
  hipEvent_t stop_event[n_streams + 1];
  for (int i = 0; i < n_streams + 1; ++i){
    hipEventCreate(&start_event[i]);
    hipEventCreate(&stop_event[i]);
  }

  // Create streams
  hipStream_t stream[n_streams];
  #error create `n_stream` streams using the above `hipStream_t` array

  for (int i = 0; i < 3; ++i){
    // Initialize memory and run case 0
    memset(a, 0, bytes);
    case_0(start_event, stop_event, stream, a, d_a, n_streams, gridsize, blocksize, n_total);

    // Initialize memory and run case 1
    memset(a, 0, bytes);
    case_1(start_event, stop_event, stream, a, d_a, n_streams, gridsize, blocksize, n_total);

    // Initialize memory and run case 2
    memset(a, 0, bytes);
    case_2(start_event, stop_event, stream, a, d_a, n_streams, gridsize, blocksize, n_total);

    // Initialize memory and run case 3
    memset(a, 0, bytes);
    case_3(start_event, stop_event, stream, a, d_a, n_streams, gridsize, blocksize, n_total);
  }

  // Destroy events
  for (int i = 0; i < n_streams + 1; ++i){
    hipEventDestroy(start_event[i]);
    hipEventDestroy(stop_event[i]);
  }

  // Destroy Streams
  #error destroy `n_stream` streams

  // Free host memory
  #if USE_PINNED_HOST_MEM == 1
    hipHostFree(a);
  #else
    free(a);
  #endif

  //Free device memory
  hipFree(d_a);
}
