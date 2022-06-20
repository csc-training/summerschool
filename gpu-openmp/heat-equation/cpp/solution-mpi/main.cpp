/* Heat equation solver in 2D. */

#include <string>
#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "heat.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv)
{

    const int image_interval = 100;    // Image output interval

    int nsteps;                 // Number of time steps
    Field current, previous;    // Current and previous temperature fields

    int provided;   // MPI thread support level

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        printf("MPI_THREAD_SERIALIZED thread support level required\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    ParallelData parallelization; // Parallelization info

    int num_devices = 0;
#ifdef _OPENMP
    num_devices = omp_get_num_devices();
#endif



    initialize(argc, argv, current, previous, nsteps, parallelization);

    // Output the initial field
    write_field(current, 0, parallelization);

    auto average_temp = average(current, parallelization);
    if (0 == parallelization.rank) {
        std::cout << "Simulation parameters: " 
                  << "rows: " << current.nx_full << " columns: " << current.ny_full
                  << " time steps: " << nsteps << std::endl;
        std::cout << "Number of MPI tasks: " << parallelization.size << std::endl;
        std::cout << "Number of devices: " << num_devices << std::endl;
#ifdef GPU_MPI
        std::cout << "Using GPU aware MPI" << std::endl;
#endif
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average temperature at start: " << average_temp << std::endl;
    }

    const double a = 0.5;     // Diffusion constant 
    auto dx2 = current.dx * current.dx;
    auto dy2 = current.dy * current.dy;
    // Largest stable time step 
    auto dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    //Get the start time stamp 
    auto start_clock = MPI_Wtime();

    enter_data(current, previous);

    // Time evolve
    for (int iter = 1; iter <= nsteps; iter++) {
#ifndef GPU_MPI
        update_host(previous);
#endif
        exchange(previous, parallelization);
#ifndef GPU_MPI
        update_device(previous);
#endif
        evolve(current, previous, a, dt);
        if (iter % image_interval == 0) {
            update_host(current);
            write_field(current, iter, parallelization);
        }
        // Swap current field so that it will be used
        // as previous for next iteration step
        std::swap(current, previous);
    }

    exit_data(current, previous);

    auto stop_clock = MPI_Wtime();

    // Average temperature for reference 
    average_temp = average(previous, parallelization);

    if (0 == parallelization.rank) {
        std::cout << "Iteration took " << (stop_clock - start_clock)
                  << " seconds." << std::endl;
        std::cout << "Average temperature: " << average_temp << std::endl;
        if (1 == argc) {
            std::cout << "Reference value with default arguments: " 
                      << 59.281239 << std::endl;
        }
    }

    
    // Output the final field
    write_field(previous, nsteps, parallelization);

    MPI_Finalize();

    return 0;
}
