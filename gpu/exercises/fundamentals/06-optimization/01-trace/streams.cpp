#include <stdio.h>
#include <string>
#include "hip/hip_runtime.h"

#define HIP_ERRCHK(result) (hip_errchk(result, __FILE__, __LINE__))
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

// DO 2^WORK loops of work in kernel
#define WORK 0

// Switch between pinned and pageable host memory
#define USE_PINNED_HOST_MEM 1

// GPU kernel definition 
__global__ void kernel(float *a, int n_total, int bias)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n_total){
    for (int k=0; k < 1<<WORK; ++k) {
      float x = (float)(bias+i);
      float s = sinf(x); 
      float c = cosf(x);
      a[i] = a[i] + sqrtf(s*s+c*c);
    }
  }
}

// Calculate the max error
float max_error(float *a, int n)
{
  float max_err = 0;
  constexpr float target = float(1<<WORK);
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-target);
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
  HIP_ERRCHK(hipEventRecord(start_event[0], 0));

  // Copy data to device, launch kernel, copy data back to host
  HIP_ERRCHK(hipMemcpy(d_a, a, n_total * sizeof(float), hipMemcpyHostToDevice));
  kernel<<<gridsize, blocksize>>>(d_a, n_total, 0);
  HIP_ERRCHK(hipMemcpy(a, d_a, n_total * sizeof(float), hipMemcpyDeviceToHost));

  // Record the stop event for the total time
  HIP_ERRCHK(hipEventRecord(stop_event[0], 0));

  // Synchronize with the event and capture timing between start_event and stop_event
  float timing[1];
  HIP_ERRCHK(hipEventSynchronize(stop_event[0]));
  HIP_ERRCHK(hipEventElapsedTime(&timing[0], start_event[0], stop_event[0]));
  
  // Print timings and the maximum error
  print_results("Case 0 - Duration for sequential transfers+kernel", timing, max_error(a, n_total), 0);
}

// Case 1: Run memory copies sequentially, distribute kernel for multiple streams
void case_1(hipEvent_t *start_event, hipEvent_t *stop_event, hipStream_t *stream, float *a, float *d_a, int n_streams, int gridsize, int blocksize, int n_total) 
{
  // Calculate per-stream problem size
  int stream_size = n_total / n_streams;
  size_t stream_bytes = sizeof(float)*stream_size;

  // Record the start event for the total time
  HIP_ERRCHK(hipEventRecord(start_event[n_streams], 0));

  // Copy data to device
  HIP_ERRCHK(hipMemcpy(d_a, a, n_total * sizeof(float), hipMemcpyHostToDevice));

  // Distribute kernel for 'n_streams' streams, and record each stream's timing
  for(int n = 0; n<n_streams; ++n) {
    HIP_ERRCHK(hipEventRecord(start_event[n],stream[n]));
    kernel<<<gridsize, blocksize, 0, stream[n]>>>(&(d_a[stream_size*n]), stream_size, stream_size*n);
    HIP_ERRCHK(hipEventRecord(stop_event[n], stream[n]));
  }

  // Copy data back to host
  HIP_ERRCHK(hipMemcpy(a, d_a, n_total * sizeof(float), hipMemcpyDeviceToHost));

  // Record the stop event for the total time
  HIP_ERRCHK(hipEventRecord(stop_event[n_streams], 0));

  // Synchronize with the events and capture timings between start_events and stop_events
  float timing[n_streams + 1];
  for (int i = 0; i < n_streams + 1; ++i) {
    HIP_ERRCHK(hipEventSynchronize(stop_event[i]));
    HIP_ERRCHK(hipEventElapsedTime(&timing[i], start_event[i], stop_event[i]));
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
  HIP_ERRCHK(hipEventRecord(start_event[n_streams], 0));

  // Distribute memcopies and the kernel for 'n_streams' streams, and record each stream's timing
  for (int n=0; n<n_streams; ++n) {
    HIP_ERRCHK(hipEventRecord(start_event[n], stream[n]));
    HIP_ERRCHK(hipMemcpyAsync(&(d_a[n*stream_size]), &(a[n*stream_size]), stream_bytes, hipMemcpyHostToDevice, stream[n]));
    kernel<<<gridsize, blocksize, 0, stream[n]>>>(&d_a[n*stream_size], stream_size, n*stream_size);
    HIP_ERRCHK(hipMemcpyAsync(&(a[n*stream_size]), &(d_a[n*stream_size]), stream_bytes, hipMemcpyDeviceToHost, stream[n]));
    HIP_ERRCHK(hipEventRecord(stop_event[n], stream[n]));
  }

  // Record the stop event for the total time
  HIP_ERRCHK(hipEventRecord(stop_event[n_streams], 0));

  // Synchronize with the events and capture timings between start_events and stop_events
  float timing[n_streams + 1];
  for (int n = 0; n<=n_streams; ++n) {
    HIP_ERRCHK(hipEventSynchronize(stop_event[n]));
    HIP_ERRCHK(hipEventElapsedTime(&timing[n], start_event[n], stop_event[n]));
  }
  
  // Print timings and the maximum error
  print_results("Case 2 - Duration for asynchronous transfers+kernels", timing, max_error(a, n_total), n_streams);
}

// Case 3: Distribute the memory copies and the kernel for multiple streams (scheduling order 2)
void case_3(hipEvent_t *start_event, hipEvent_t *stop_event, hipStream_t *stream, float *a, float *d_a, int n_streams, int gridsize, int blocksize, int n_total) 
{
  // Calculate per-stream problem size and byte size
  int stream_size = n_total / n_streams;
  int stream_bytes = stream_size * sizeof(float);

  // Record the start event for the total time
  HIP_ERRCHK(hipEventRecord(start_event[n_streams], 0));

  // Distribute memcopies and the kernel for 'n_streams' streams, and record each stream's timing
  for (int n=0; n<n_streams; ++n) {
    HIP_ERRCHK(hipEventRecord(start_event[n], stream[n]));
    HIP_ERRCHK(hipMemcpyAsync(&(d_a[n*stream_size]), &(a[n*stream_size]), stream_bytes, hipMemcpyHostToDevice, stream[n]));
  }
  for (int n=0; n<n_streams; ++n) 
    kernel<<<gridsize, blocksize, 0, stream[n]>>>(&d_a[n*stream_size], stream_size, n*stream_size);

  for (int n=0; n<n_streams; ++n) {
    HIP_ERRCHK(hipMemcpyAsync(&(a[n*stream_size]), &(d_a[n*stream_size]), stream_bytes, hipMemcpyDeviceToHost, stream[n]));
    HIP_ERRCHK(hipEventRecord(stop_event[n], stream[n]));
  }

  // Record the stop event for the total time
  HIP_ERRCHK(hipEventRecord(stop_event[n_streams], 0));

  // Synchronize with the events and capture timings between start_events and stop_events
  float timing[n_streams + 1];
  for (int n = 0; n<=n_streams; ++n) {
    HIP_ERRCHK(hipEventSynchronize(stop_event[n]));
    HIP_ERRCHK(hipEventElapsedTime(&timing[n], start_event[n], stop_event[n]));
  }
  
  // Print timings and the maximum error
  
  // Print timings and the maximum error
  print_results("Case 3 - Duration for asynchronous transfers+kernels", timing, max_error(a, n_total), n_streams);
}

int main(){
  
  // Problem size
  constexpr int n_total = 1<<24; // pow(2, 22);

  // Device grid sizes
  constexpr int n_streams = 4;
  constexpr int blocksize = 256;
  constexpr int gridsize = (n_total - 1 + blocksize) / blocksize;

  // Allocate host and device memory
  float *a, *d_a;
  const int bytes = n_total * sizeof(float);

  #if USE_PINNED_HOST_MEM == 1
    HIP_ERRCHK(hipHostMalloc((void**)&a, bytes));      // host pinned
  #else
    a=(float *)malloc(bytes);              // host pageable
  #endif
  HIP_ERRCHK(hipMalloc((void**)&d_a, bytes));          // device pinned

  // Create events
  hipEvent_t start_event[n_streams + 1];
  hipEvent_t stop_event[n_streams + 1];
  for (int i = 0; i < n_streams + 1; ++i){
    HIP_ERRCHK(hipEventCreate(&start_event[i]));
    HIP_ERRCHK(hipEventCreate(&stop_event[i]));
  }

  // Create streams
  hipStream_t streams[n_streams];
  for (int i = 0; i < n_streams; ++i) HIP_ERRCHK(hipStreamCreate(&streams[i]));

  // Initialize memory and run case 0
  memset(a, 0, bytes);
  case_0(start_event, stop_event, streams, a, d_a, n_streams, gridsize, blocksize, n_total);

  // Initialize memory and run case 1
  memset(a, 0, bytes);
  case_1(start_event, stop_event, streams, a, d_a, n_streams, gridsize, blocksize, n_total);
  // Initialize memory and run case 2
  memset(a, 0, bytes);
  case_2(start_event, stop_event, streams, a, d_a, n_streams, gridsize, blocksize, n_total);

  // Initialize memory and run case 3
  memset(a, 0, bytes);
  case_3(start_event, stop_event, streams, a, d_a, n_streams, gridsize, blocksize, n_total);

  // Destroy events
  for (int i = 0; i < n_streams + 1; ++i){
    HIP_ERRCHK(hipEventDestroy(start_event[i]));
    HIP_ERRCHK(hipEventDestroy(stop_event[i]));
  }

  // Destroy Streams
  /* #error destroy `n_stream` streams */
  for (int i = 0; i < n_streams; ++i) HIP_ERRCHK(hipStreamDestroy(streams[i]));

  // Free host memory
#if USE_PINNED_HOST_MEM == 1
  HIP_ERRCHK(hipHostFree(a));
#else
  free(a);
#endif

  //Free device memory
  HIP_ERRCHK(hipFree(d_a));
}
