// SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

/*
 * This code compares different GPU memory allocation strategies.
 *
 * Task is to:
 *   - allocate device memory outside of a loop using hipMalloc()
 *   - perform recurring allocations in a loop using hipMalloc()/hipFree()
 *   - perform recurring allocations in a loop using hipMallocAsync()/hipFreeAsync()
 *   - create and synchronize a HIP stream for async allocations
 *   - compare the timing between the approaches
 *
 * Observe how stream-ordered memory allocation can reduce
 * recurring allocation overhead through memory pooling.
 */
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include "error_checking.hpp"

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

/* GPU kernel definition */
__global__ void hipKernel(int* const A, const int size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    A[idx] = idx;
}

/* Auxiliary function to check the results */
void checkTiming(const std::string strategy, const float timing_ms)
{
  printf("%.3f ms - %s\n", timing_ms, strategy.c_str());
}

/* Run without timing as a warmup */
void warmupRun(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  size_t bytes = size * sizeof(int);
  
  int *d_A;
  // Allocate device memory
  HIP_ERRCHK(hipMalloc((void**)&d_A, bytes));

  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    // Synchronization
    HIP_ERRCHK(hipStreamSynchronize(0));
  }
  // Free allocation
  HIP_ERRCHK(hipFree(d_A));
}

/* Run using a single device allocation outside of the loop */
void noRecurringAlloc(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  size_t bytes = size * sizeof(int);

  int *d_A;
  // Allocate device memory
  #error allocate memory with hipMalloc for d_A of size `bytes`

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {    
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
  }
  // Synchronization
  #error synchronize the default stream here before stopping the timer

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkTiming("noRecurringAlloc", timing);

  // Free allocation
  #error free d_A allocation using hipFree
}

/* Do recurring allocation without memory pooling */
void recurringAllocNoMemPools(int nSteps, int size)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  size_t bytes = size * sizeof(int);

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate device memory
    #error allocate memory with hipMalloc for d_A of size `bytes`
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, 0>>>(d_A, size);
    HIP_ERRCHK(hipGetLastError());
    // Free allocation
    #error free d_A allocation using hipFree
  }
  // Synchronization
  // Ensure all queued allocations and kernels complete before stopping timing
  #error synchronize the default stream here before stopping the timer

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkTiming("recurringAllocNoMemPools", timing);
}

/* Do recurring allocation with memory pooling */
void recurringAllocMallocAsync(int nSteps, int size)
{
  // Create HIP stream
  hipStream_t stream;
  HIP_ERRCHK(hipStreamCreate(&stream));

  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (size - 1 + blocksize) / blocksize;

  size_t bytes = size * sizeof(int);

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    int *d_A;
    // Allocate device memory
    #error allocate memory with hipMallocAsync for d_A of size `bytes` in stream
    // Launch GPU kernel
    hipKernel<<<gridsize, blocksize, 0, stream>>>(d_A, size);
    HIP_ERRCHK(hipGetLastError());
    // Free allocation
    #error free d_A allocation using hipFreeAsync in stream
  }
  // Synchronization
  #error synchronize stream here

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkTiming("recurringAllocMallocAsync", timing);

  // Destroy the stream
  HIP_ERRCHK(hipStreamDestroy(stream));
}

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 1D grid dimensions
  int nSteps = 1e4, size = 1e6;
  
  // Ignore first run, first kernel is slower (warmup)
  warmupRun(10, size);

  // Run with different memory allocation strategies
  noRecurringAlloc(nSteps, size);
  recurringAllocNoMemPools(nSteps, size);
  recurringAllocMallocAsync(nSteps, size);
}
