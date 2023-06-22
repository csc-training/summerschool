#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Namespaces "comms" and "devices" declared here */
#include "comms.h"

/* Parameters for the numerical problem */
#define NITER 50
#define NX 4096
#define NY 4096

/* The task to do for each rank */
enum RankTask{   
    cpu_slow = 3, 
    cpu_fast = 2, 
    gpu_slow = 1,
    gpu_fast = 0
};

int main(int argc, char *argv []){

  /* Initialize processes and devices */
  comms_init_procs(&argc, &argv);
  int my_rank = comms_get_rank();

  /* Set discretization and the number of time steps */
  const int niters = NITER;
  const int nx = NX;
  const int ny = NY;

  /* Calculate element sizes */
  const double dx = 1.0 / (double)nx;
  const double dy = 1.0 / (double)ny;

  /* Allocate memory for arrays */
  double *A, *L;
  A = (double*)devices_allocate(nx * ny * sizeof(double));
  L = (double*)devices_allocate(nx * ny * sizeof(double));

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
  clock_t t0;
  clock_t diff;
  int msec;

  /* Compute Laplacian with different strategies (host/device) depending on the process rank */
  switch(my_rank) {

    case cpu_slow: 

      t0 = clock(); /* Begin timer */

      for (int iter = 0; iter < niters; iter++)
        for (int j = 1; j < ny - 1; j++)
          for (int i = 1; i < nx - 1; i++)
            L[i * ny + j] = (A[(i - 1) * ny + j] - 2.0 * A[i * ny + j] + A[(i + 1) * ny + j]) / 
              (dx * dx) + (A[i * ny + j - 1] - 2.0 * A[i * ny + j] + A[i * ny + j + 1]) / (dy * dy);
      
      diff = clock() - t0; /* End timer */
      msec = diff  * 1000.0 / CLOCKS_PER_SEC; 
      printf("Rank %d on CPU (slow): %d[ms]\n", my_rank,  msec); /* Print the timing */
      break;

    case cpu_fast:  

      t0 = clock(); /* Begin timer */
      
      for (int iter = 0; iter < niters; iter++)
        for (int i = 1; i < nx - 1; i++)
          for (int j = 1; j < ny - 1; j++)
            L[i * ny + j] = (A[(i - 1) * ny + j] - 2.0 * A[i * ny + j] + A[(i + 1) * ny + j]) / 
              (dx * dx) + (A[i * ny + j - 1] - 2.0 * A[i * ny + j] + A[i * ny + j + 1]) / (dy * dy);
      
      diff = clock() - t0; /* End timer */
      msec = diff  * 1000.0 / CLOCKS_PER_SEC; 
      printf("Rank %d on CPU (fast): %d[ms]\n", my_rank,  msec); /* Print the timing */
      break;

    case gpu_slow:

      t0 = clock(); /* Begin timer */

      for (int iter = 0; iter < niters; iter++){
        int i, j;
        devices_parallel_for((nx - 2), (ny - 2), i, j, 1, 1,
        {
          L[i * ny + j] = (A[(i - 1) * ny + j] - 2.0 * A[i * ny + j] + A[(i + 1) * ny + j]) / 
            (dx * dx) + (A[i * ny + j - 1] - 2.0 * A[i * ny + j] + A[i * ny + j + 1]) / (dy * dy);
        });
      }

      diff = clock() - t0; /* End timer */
      msec = diff  * 1000.0 / CLOCKS_PER_SEC; 
      printf("Rank %d on device (slow): %d[ms]\n", my_rank,  msec); /* Print the timing */
      break;

    case gpu_fast:

      t0 = clock(); /* Begin timer */

      for (int iter = 0; iter < niters; iter++){
        int i, j;
        devices_parallel_for((ny - 2), (nx - 2), j, i, 1, 1,
        {
          L[i * ny + j] = (A[(i - 1) * ny + j] - 2.0 * A[i * ny + j] + A[(i + 1) * ny + j]) / 
            (dx * dx) + (A[i * ny + j - 1] - 2.0 * A[i * ny + j] + A[i * ny + j + 1]) / (dy * dy);
        });
      }

      diff = clock() - t0; /* End timer */
      msec = diff  * 1000.0 / CLOCKS_PER_SEC; 
      printf("Rank %d on device (fast): %d[ms]\n", my_rank,  msec); /* Print the timing */
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
  devices_free((void*)A);
  devices_free((void*)L);

  /* Sync processes */
  comms_barrier_procs();

  /* Finalize processes and devices */
  comms_finalize_procs();

  return 0;
}