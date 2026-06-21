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
    hipKernel<<<gridsize, BLOCKSIZE, 0, 0>>>(d_A, nx, ny);
    HIP_ERRCHK(hipGetLastError());
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

  // Allocate pageable host memory
  A = (int*)malloc(size);

  // Allocate device memory
  HIP_ERRCHK(hipMalloc((void**)&d_A, size));

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent an optimal
     * workflow of a GPU accelerated program where all operations
     * are performed using device (i.e., recurring memcopy is avoided):
     * Initializing the array directly on the GPU and running a GPU kernel.
     */

    // Initialize array from device
    HIP_ERRCHK(hipMemset(d_A, 0, size));

    // Launch GPU kernel
    hipKernel<<<gridsize, BLOCKSIZE, 0, 0>>>(d_A, nx, ny);
    HIP_ERRCHK(hipGetLastError());
  }

  // Copy data back to host
  HIP_ERRCHK(hipMemcpy(A, d_A, size, hipMemcpyDeviceToHost));

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkResults(A, nx, ny, "ExplicitMemNoCopy", timing);

  // Free device array
  HIP_ERRCHK(hipFree(d_A));

  // Free host array
  free(A);
}

/* Run using Unified Memory without recurring host/device memcopies */
void unifiedMemNoCopy(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A;
  size_t size = nx * ny * sizeof(int);

  // Allocate Unified Memory
  HIP_ERRCHK(hipMallocManaged((void**)&A, size));

  // Start timer and begin stepping loop
  auto tStart = std::chrono::steady_clock::now();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent an optimal
     * workflow of a GPU accelerated program where all operations
     * are performed using device (i.e., recurring memcopy is avoided):
     * Initializing the array directly on the GPU and running a GPU kernel.
     */

    // Initialize array from device
    HIP_ERRCHK(hipMemset(A, 0, size));

    // Launch GPU kernel
    hipKernel<<<gridsize, BLOCKSIZE, 0, 0>>>(A, nx, ny);
    HIP_ERRCHK(hipGetLastError());

  }
  // Prefetch data from device to host
  HIP_ERRCHK(hipMemPrefetchAsync(A, size, hipCpuDeviceId, 0));

  // Synchronization
  HIP_ERRCHK(hipStreamSynchronize(0));

  // Check results and print timings
  auto tStop = std::chrono::steady_clock::now();
  float timing = std::chrono::duration<float, std::milli>(tStop - tStart).count();
  checkResults(A, nx, ny, "UnifiedMemNoCopy", timing);

  // Free Unified Memory array
  HIP_ERRCHK(hipFree(A));
}

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 2D grid dimensions
  int nSteps = 100, nx = 8000, ny = 2000;

  // Ignore first run, first kernel is slower (warmup)
  warmupRun(5, nx, ny);

  // Run with different memory management strategies
  explicitMemNoCopy(nSteps, nx, ny);
  unifiedMemNoCopy(nSteps, nx, ny);
}
