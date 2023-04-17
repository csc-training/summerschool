#include <cstdio>
#include <string>
#include <time.h>
#include <hip/hip_runtime.h>


/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

// HIP error checking
#define HIP_ERR(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

/* GPU kernel definition */
/*__global__ void hipKernel_coalesced(int* const A, const int size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_workers = blockDim.x * gridDim.x; 
  for (int i=idx; i < size; i+=num_workers)
    A[i] = A[i] + i;
}


__global__ void hipKernel_uncoalesced(int* const A, const int size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_workers = blockDim.x * gridDim.x;
  const int num_points_per_thread = size/num_workers;
  const int num_thread_with_extra_point = size%num_workers;
  const int my_start = idx < num_thread_with_extra_point ? idx * (num_points_per_thread + 1) : num_thread_with_extra_point * (num_points_per_thread+1) + (idx-num_thread_with_extra_point) *num_points_per_thread;
  const int my_end = idx < num_thread_with_extra_point ? my_start + num_points_per_thread + 1 : my_start + num_points_per_thread;
  for(int i=my_start; i<my_end; ++i)
    A[i] = A[i] + i;
}*/

template< bool coalesced >
__global__ void hipKernel(int* const A, const int size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_workers = blockDim.x * gridDim.x;
  if constexpr (coalesced)
  {
    for (int i=idx; i < size; i+=num_workers)
    A[i] = A[i] + i;
  }
  else
  {
    const int num_points_per_thread = size/num_workers;
    const int num_thread_with_extra_point = size%num_workers;
    const int my_start = idx < num_thread_with_extra_point ? idx * (num_points_per_thread + 1) : num_thread_with_extra_point * (num_points_per_thread+1) + (idx-num_thread_with_extra_point) *num_points_per_thread;
    const int my_end = idx < num_thread_with_extra_point ? my_start + num_points_per_thread + 1 : my_start + num_points_per_thread;
    for(int i=my_start; i<my_end; ++i)
      A[i] = A[i] + i;
  }
}

/* Auxiliary function to check the results */
void checkTiming(const std::string strategy, const double timing)
{
  printf("%.3f ms - %s\n", timing * 1e3, strategy.c_str());
}

/*warmup, not interested in performances*/
void ignoreTiming(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  int *d_A;
  // Allocate pinned device memory
  HIP_ERR(hipMalloc((void**)&d_A, sizeof(int) * size));

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<true><<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Synchronization
    HIP_ERR(hipStreamSynchronize(0));
  }
  // Free allocation
  HIP_ERR(hipFree(d_A));
}

/* Run without recurring allocation */
template< bool coalesced> 
void noRecurringAlloc(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  int *d_A;
  // Start timer, allocate and do things
  clock_t tStart = clock();
  
  // Allocate pinned device memory
  HIP_ERR(hipMalloc((void**)&d_A, sizeof(int) * size));
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<coalesced><<<gridsize, blocksize, 0, 0>>>(d_A, size);
  }
  // Synchronization
  HIP_ERR(hipStreamSynchronize(0));
  // Check results and print timings
  // Free allocation
  HIP_ERR(hipFree(d_A));
  checkTiming("noRecurringAlloc", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Do recurring allocation without memory pooling */
template< bool coalesced> 
void recurringAllocNoMemPools(int nSteps, int size)
{
  clock_t tStart = clock();
  
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate pinned device memory
    HIP_ERR(hipMalloc((void**)&d_A, sizeof(int) * size));
    // Launch GPU kernel
    hipKernel<coalesced><<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Free allocation
    HIP_ERR(hipFree(d_A));
  }
  // Synchronization
  HIP_ERR(hipStreamSynchronize(0));
  // Check results and print timings
  checkTiming("recurringAllocNoMemPools", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}

/* Do recurring allocation with memory pooling */
template< bool coalesced> 
void recurringAllocMallocAsync(int nSteps, int size)
{
  clock_t tStart = clock();
  // Create HIP stream
  hipStream_t stream;
  HIP_ERR(hipStreamCreate(&stream));

  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate pinned device memory
    cudaMallocAsync((void**)&d_A, sizeof(int) * size, stream);
    // Launch GPU kernel
    hipKernel<coalesced><<<gridsize, blocksize, 0, stream>>>(d_A, size);
    // Free allocation
    cudaFreeAsync(d_A, stream);
  }
  // Synchronization
  HIP_ERR(hipStreamSynchronize(stream));
  // Check results and print timings
  // Destroy the stream
  HIP_ERR(hipStreamDestroy(stream));

  checkTiming("recurringAllocMallocAsync", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  
}


/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 1D grid dimensions
  int nSteps = 1e4, size = 1e6;
  
  // Ignore first run, first kernel is slower
  ignoreTiming(nSteps, size);

  printf("coalesced accesses: \n");
  // Run with different memory allocatins strategies
  noRecurringAlloc<true>(nSteps, size);
  recurringAllocNoMemPools<true>(nSteps, size);
  recurringAllocMallocAsync<true>(nSteps, size);
  
  printf("uncoalesced accesses: \n");
  // Run with different memory allocatins strategies
  noRecurringAlloc<false>(nSteps, size);
  recurringAllocNoMemPools<false>(nSteps, size);
  recurringAllocMallocAsync<false>(nSteps, size);

}