// SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

/*
 * Task is to modify four functions in the code:
 * - implement explicit memory management with hipMalloc and hipMemcpy
 * - implement pinned host memory with hipHostMalloc
 * - implement unified memory with hipMallocManaged
 * - implement unified memory prefetching with hipMemPrefetchAsync
 * - compare execution times between the approaches
 */
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include "error_checking.hpp"

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

/* GPU kernel definition */
__global__ void hipKernel(int* const A, const int nx, const int ny)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nx * ny)
  {
    const int i = idx % nx;
    const int j = idx / nx;
    A[j * nx + i] += idx;
  }
}

/* Auxiliary function to check the results */
void checkResults(int* const A, const int nx, const int ny,
                  const std::string strategy, const float timing_ms)
{
  // Check the results are correct
  int errored = 0;
  for(unsigned int i = 0; i < nx * ny; i++)
    if(A[i] != i)
      errored = 1;

  if(errored)
    printf("The results are incorrect!\n");
  else
    printf("The results are OK! (%.3f ms - %s)\n",
           timing_ms, strategy.c_str());
}

/* Run without timing as a warmup */
void warmupRun(int nSteps, int nx, int ny)
{
  // Determine grid and block size
  const int blocksize = BLOCKSIZE;
  const int gridsize = (nx * ny - 1 + blocksize) / blocksize;

  size_t bytes = nx * ny * sizeof(int);

  int *d_A;
  // Allocate device memory
  HIP_ERRCHK(hipMalloc((void**)&d_A, bytes));

  for(unsigned int i = 0; i < nSteps; i++)
  {
    // Launch GPU kernel
    hipKernel<<<gridsize, BLOCKSIZE, 0, 0>>>(d_A, nx, ny);
    HIP_ERRCHK(hipGetLastError());
  }

  // Synchronization
  HIP_ERRCHK(hipStreamSynchronize(0));

  // Free allocation
  HIP_ERRCHK(hipFree(d_A));
}

/* Run using explicit memory management */
void explicitMem(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A, *d_A;
  size_t size = nx * ny * sizeof(int);

  #error Allocate pageable host memory of size `size` for the pointer A

  #error Allocate device memory (d_A)

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent a common
     * workflow of a GPU accelerated program:
     * Accessing the array from host,
     * copying data from host to device,
     * and running a GPU kernel.
     */

    // Initialize array from host
    memset(A, 0, size);

    #error Copy data to device (A to d_A)

    #error Launch GPU kernel hipKernel

    #error Synchronization
  }

  #error Copy data back to host (d_A to A)

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkResults(A, nx, ny, "ExplicitMemCopy", timing);

  #error Free device array (d_A)

  #error Free host array (A)
}

/* Run using explicit memory management and pinned host allocations */
void explicitMemPinned(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A, *d_A;
  size_t size = nx * ny * sizeof(int);

  #error Allocate pinned host memory of size `size` for the pointer A

  #error Allocate device memory (d_A)

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent a common
     * workflow of a GPU accelerated program:
     * Accessing the array from host,
     * copying data from host to device,
     * and running a GPU kernel.
     */

    // Initialize array from host
    memset(A, 0, size);

    #error Copy data to device (A to d_A)

    #error Launch GPU kernel hipKernel

    #error Synchronization
  }

  #error Copy data back to host (d_A to A)

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkResults(A, nx, ny, "ExplicitMemPinnedCopy", timing);

  #error Free device array (d_A)

  #error Free host array (A)
}

/* Run using Unified Memory */
void unifiedMem(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A;
  size_t size = nx * ny * sizeof(int);

  #error Allocate Unified Memory of size `size` for the pointer A

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent
     * a common workflow of a GPU accelerated program:
     * Accessing the array from host,
     * (data copy from host to device is handled automatically),
     * and running a GPU kernel.
     */

    // Initialize array from host
    memset(A, 0, size);

    #error Launch GPU kernel hipKernel

    #error Synchronization
  }

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkResults(A, nx, ny, "UnifiedMemNoPrefetch", timing);

  #error Free Unified Memory array (A)
}

/* Run using Unified Memory and prefetching */
void unifiedMemPrefetch(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int device;
  #error Get device id number for prefetching
  
  int *A;
  size_t size = nx * ny * sizeof(int);

  #error Allocate Unified Memory of size `size` for the pointer A

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent a common
     * workflow of a GPU accelerated program:
     * Accessing the array from host,
     * prefetching data from host to device,
     * and running a GPU kernel.
     */

    // Initialize array from host
    memset(A, 0, size);

    #error Prefetch data from host to device (A)

    #error Launch GPU kernel hipKernel

    #error Synchronization
  }

  #error Prefetch data from device to host (A)

  #error Synchronization

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkResults(A, nx, ny, "UnifiedMemPrefetch", timing);

  #error Free Unified Memory array (A)
}

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 2D grid dimensions
  int nSteps = 100, nx = 8000, ny = 2000;

  // Ignore first run, first kernel is slower (warmup)
  warmupRun(5, nx, ny);

  // Run with different memory management strategies
  explicitMem(nSteps, nx, ny);
  explicitMemPinned(nSteps, nx, ny);
  unifiedMem(nSteps, nx, ny);
  unifiedMemPrefetch(nSteps, nx, ny);
}