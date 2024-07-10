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

    initialize(argc, argv, current, previous, nsteps);
    int num_devices=0, device_num=0;
    num_devices = omp_get_num_devices();
    device_num = omp_get_device_num();

    // Output the initial field
    write_field(current, 0);
    
    auto average_temp = average(current);

    std::cout << "Simulation parameters: " 
            << "rows: " << current.nx_full << " columns: " << current.ny_full
            << " time steps: " << nsteps << std::endl;
    std::cout << "Number of devices: " << num_devices << std::endl;
    std::cout << "Device number: " << device_num << std::endl << std::endl;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average temperature at start: " << average_temp << std::endl;

    const double a = 0.5;     // Diffusion constant 
    auto dx2 = current.dx * current.dx;
    auto dy2 = current.dy * current.dy;
    // Largest stable time step 
    auto dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    //Get the start time stamp 
    auto start_clock = omp_get_wtime();

    // Time evolve
    enter_data(current, previous);  // Start of an unstructured data region.
    
    for (int iter = 1; iter <= nsteps; iter++) {
        evolve(current, previous, a, dt);

        if (iter % image_interval == 0) {
            update_host_data(current);  // Must update host data before writing to file.
            write_field(current, iter);
        }
        // Swap current field so that it will be used
        // as previous for next iteration step
        std::swap(current, previous);
    }
    exit_data(current, previous);  // End of an unstructured data region.

    auto stop_clock = omp_get_wtime();

    // Average temperature for reference 
    average_temp = average(previous);

    std::cout << "Iteration took " << (stop_clock - start_clock)
          << " seconds." << std::endl;
    std::cout << "Average temperature: " << average_temp << std::endl;
    if (1 == argc) {
        std::cout << "Reference value with default arguments: " 
                << 59.281239 << std::endl;
    }
    
    // Output the final field
    write_field(previous, nsteps);

    return 0;
}
