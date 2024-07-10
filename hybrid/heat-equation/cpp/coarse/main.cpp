/* Heat equation solver in 2D. */

#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "heat.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

double wtime();

int main(int argc, char **argv)
{

    const int image_interval = 100;    // Image output interval

    int nsteps;                 // Number of time steps

    int num_threads = 1;

    Field current, previous;    // Current and previous temperature fields
  
  #pragma omp parallel  // Tells OpenMP to create a parallel region, in which threads are spawned. The number of threads spawned in parallel regions is specified by the environment variable OMP_NUM_THREADS.
  {
    // TODO: determine number of threads
    #pragma omp master
    num_threads = omp_get_num_threads();
    
    
    // TODO end

    initialize(argc, argv, current, previous, nsteps);

    #pragma omp single
    {
      // Output the initial field
      write_field(current, 0);

      auto average_temp = average(current);
      std::cout << "Simulation parameters: " 
                << "rows: " << current.nx_full << " columns: " << current.ny_full
                << " time steps: " << nsteps << std::endl;
      std::cout << "Number of OpenMP threads: " << num_threads << std::endl;
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "Average temperature at start: " << average_temp << std::endl;
    }
    
    const double a = 0.5;     // Diffusion constant 
    auto dx2 = current.dx * current.dx;
    auto dy2 = current.dy * current.dy;
    // Largest stable time step 
    auto dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    //Get the start time stamp 
    auto start_clock = wtime();

    // Time evolve
    //#pragma omp for
    for (int iter = 1; iter <= nsteps; iter++) {
        evolve(current, previous, a, dt);

        #pragma omp single
        {
          if (iter % image_interval == 0) {
              write_field(current, iter);
          }
          
          // Swap current field so that it will be used
          // as previous for next iteration step
          std::swap(current, previous);
        }
    }

    auto stop_clock = wtime();

    #pragma omp master
    {
      // Average temperature for reference 
      auto average_temp = average(previous);
      std::cout << "Iteration took " << (stop_clock - start_clock)
                << " seconds." << std::endl;
      std::cout << "Average temperature: " << average_temp << std::endl;
      if (1 == argc) {
        std::cout << "Reference value with default arguments: " 
                  << 59.281239 << std::endl;
      }

      // Output the final field
      write_field(previous, nsteps);
    }
  
  } // end #pragma omp parallel
     return 0;
}

double wtime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  using clock = std::chrono::high_resolution_clock;
  auto time = clock::now();
  auto duration = std::chrono::duration<double>(time.time_since_epoch());
  return duration.count();
#endif
}
