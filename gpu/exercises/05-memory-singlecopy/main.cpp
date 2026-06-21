// SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

/*
 * Task is to optimize two functions in the code based on the previous exercise:
 * - avoid recurring host-to-device memory transfers
 * - keep data resident on the GPU during the iterative loop
 * - initialize memory directly on the device using hipMemset()
 * - compare explicit memory management against unified memory
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
void checkResults(int* const A, const int nx, const int ny, const std::string strategy, const float timing_ms)
{
  // Check that the results are correct
  int errored = 0;
  for(unsigned int i = 0; i < nx * ny; i++)
    if(A[i] != i)
      errored = 1;

  // Indicate if the results are correct
  if(errored)
    printf("The results are incorrect!\n");
  else
    printf("The results are OK! (%.3f ms - %s)\n", timing_ms, strategy.c_str());
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
    HIP_ERRCHK(hipMemset(d_A, 0, bytes));
    // Launch GPU kernel
    #error Launch GPU kernel hipKernel
  }

  // Synchronization
  HIP_ERRCHK(hipStreamSynchronize(0));
  // Free allocation
  HIP_ERRCHK(hipFree(d_A));
}

/* Run using explicit memory management without recurring host/device memcopies */
void explicitMemNoCopy(int nSteps, int nx, int ny)
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
    /* The order of calls inside this loop represent an optimal
     * workflow of a GPU accelerated program where all operations
     * are performed using device (i.e., recurring memcopy is avoided):
     * Initializing the array directly on the GPU and running a GPU kernel.
     */

    #error Initialize array A to zeros on the device using hipMemset

    // Launch GPU kernel
    hipKernel<<<gridsize, BLOCKSIZE, 0, 0>>>(d_A, nx, ny);
    HIP_ERRCHK(hipGetLastError());
  }

  #error Copy data back to host (d_A to A)

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkResults(A, nx, ny, "ExplicitMemNoCopy", timing);

  #error Free device array (d_A)

  #error Free host array (A)
}

/* Run using Unified Memory without recurring host/device memcopies */
void unifiedMemNoCopy(int nSteps, int nx, int ny)
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
    /* The order of calls inside this loop represent an optimal
     * workflow of a GPU accelerated program where all operations
     * are performed using device (i.e., recurring memcopy is avoided):
     * Initializing the array directly on the GPU and running a GPU kernel.
     */

    #error Initialize array A to zeros on the device using hipMemset

    // Launch GPU kernel
    #error Launch GPU kernel hipKernel
  }
  #error Prefetch data (A) from device to host

  #error Synchronization

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkResults(A, nx, ny, "UnifiedMemNoCopy", timing);

  #error Free Unified Memory array (A)
}

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 2D grid dimensions
  int nSteps = 100, nx = 8000, ny = 2000;

  // Ignore first run, first kernel is slower (warmup)
  warmupRun(5, nx, ny);

  // Compare timing
  explicitMemNoCopy(nSteps, nx, ny);
  unifiedMemNoCopy(nSteps, nx, ny);
}
