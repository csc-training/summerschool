#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

/* Namespaces "comms" and "devices" declared here */
#include "comms.h"

/* Parameters for the numerical problem */
#define NITER 50
#define NX 4096
#define NY 4096

int main(int argc, char *argv []){

  /* The task to do for each rank */
  enum RankTask{   
      cpu_slow = 3, 
      cpu_fast = 2, 
      gpu_slow = 1,
      gpu_fast = 0
  };

  /* Initialize processes and devices */
  comms::init_procs(&argc, &argv);
  auto my_rank = static_cast<RankTask>(comms::get_rank());

  /* Set discretization and the number of time steps */
  const int niters = NITER;
  const int nx = NX;
  const int ny = NY;

  /* Calculate element sizes */
  const double dx = 1.0 / double(nx);
  const double dy = 1.0 / double(ny);

  /* Allocate memory for arrays */
  double *A, *L;
  A = (double*)devices::allocate(nx * ny * sizeof(double));
  L = (double*)devices::allocate(nx * ny * sizeof(double));

  /* Initialize arrays on host */
  double x = 0.0, y;
  for (int i = 0; i < nx; i++){
    y = 0.0; 
    for (int j = 0; j < ny; j++)
      {
        A[i * ny + j] = x * x + y * y;
        L[i * ny + j] = 0.0;
        y += dy;
      }
    x += dx;
  }

  /* Timing variables */
  std::chrono::steady_clock::time_point t0;
  std::chrono::steady_clock::time_point t1;

  /* Compute Laplacian with different strategies (host/device) depending on the process rank */
  switch(my_rank) {

    case cpu_slow: 

      t0 = std::chrono::steady_clock::now(); /* Begin timer */

      for (int iter = 0; iter < niters; iter++)
        for (int j = 1; j < ny - 1; j++)
          for (int i = 1; i < nx - 1; i++)
            L[i * ny + j] = (A[(i - 1) * ny + j] - 2.0 * A[i * ny + j] + A[(i + 1) * ny + j]) / 
              (dx * dx) + (A[i * ny + j - 1] - 2.0 * A[i * ny + j] + A[i * ny + j + 1]) / (dy * dy);
      
      t1 = std::chrono::steady_clock::now(); /* End timer */
      std::cout << "Rank " << my_rank << " on CPU (slow): " << std::chrono::duration_cast<std::chrono::milliseconds>
        (t1 - t0).count() << "[ms]" << std::endl; /* Print the timing */
      break;

    case cpu_fast:  

      t0 = std::chrono::steady_clock::now(); /* Begin timer */
      
      for (int iter = 0; iter < niters; iter++)
        for (int i = 1; i < nx - 1; i++)
          for (int j = 1; j < ny - 1; j++)
            L[i * ny + j] = (A[(i - 1) * ny + j] - 2.0 * A[i * ny + j] + A[(i + 1) * ny + j]) / 
              (dx * dx) + (A[i * ny + j - 1] - 2.0 * A[i * ny + j] + A[i * ny + j + 1]) / (dy * dy);
      
      t1 = std::chrono::steady_clock::now(); /* End timer */
      std::cout << "Rank " << my_rank << " on CPU (fast): " << std::chrono::duration_cast<std::chrono::milliseconds>
        (t1 - t0).count() << "[ms]" << std::endl; /* Print the timing */
      break;

    case gpu_slow:

      t0 = std::chrono::steady_clock::now(); /* Begin timer */

      for (int iter = 0; iter < niters; iter++){
        devices::parallel_for(nx - 2, ny - 2,
          DEVICE_LAMBDA(int i, int j) {
            i += 1; j += 1;
            L[i * ny + j] = (A[(i - 1) * ny + j] - 2.0 * A[i * ny + j] + A[(i + 1) * ny + j]) / 
              (dx * dx) + (A[i * ny + j - 1] - 2.0 * A[i * ny + j] + A[i * ny + j + 1]) / (dy * dy);
          }
        );
      }

      t1 = std::chrono::steady_clock::now(); /* End timer */
      std::cout << "Rank " << my_rank << " on device (slow): " << std::chrono::duration_cast<std::chrono::milliseconds>
        (t1 - t0).count() << "[ms]" << std::endl; /* Print the timing */
      break;

    case gpu_fast:

      t0 = std::chrono::steady_clock::now(); /* Begin timer */

      for (int iter = 0; iter < niters; iter++){
        devices::parallel_for(ny - 2, nx - 2,
          DEVICE_LAMBDA(int j, int i) {
            i += 1; j += 1;
            L[i * ny + j] = (A[(i - 1) * ny + j] - 2.0 * A[i * ny + j] + A[(i + 1) * ny + j]) / 
              (dx * dx) + (A[i * ny + j - 1] - 2.0 * A[i * ny + j] + A[i * ny + j + 1]) / (dy * dy);
          }
        );
      }

      t1 = std::chrono::steady_clock::now(); /* End timer */
      std::cout << "Rank " << my_rank << " on device (fast): " << std::chrono::duration_cast<std::chrono::milliseconds>
        (t1 - t0).count() << "[ms]" << std::endl; /* Print the timing */
      break;
  }  
    
  /* Check the result */
  double mean_L = 0.0;
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
      mean_L += L[i * ny + j];

  mean_L /= ((nx - 1) * (ny - 1));

  /* Print the numerical and the analytical result */
  printf("Rank %d, numerical/analytical solution: %6.4f/ %6.4f\n", my_rank, mean_L, 4.0);

  /* Deallocate memory */
  devices::free((void*)A);
  devices::free((void*)L);

  /* Sync processes */
  comms::barrier_procs();

  /* Finalize processes and devices */
  comms::finalize_procs();

  return 0;
}