/* Heat equation solver in 2D. */

#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "heat.hpp"

int main(int argc, char **argv)
{

    const int image_interval = 100;    // Image output interval

    int nsteps;                 // Number of time steps
    Field current, previous;    // Current and previous temperature fields
    initialize(argc, argv, current, previous, nsteps);

    // Output the initial field
    write_field(current, 0);

    auto average_temp = average(current);
    std::cout << "Simulation parameters: "
              << "rows: " << current.nx << " columns: " << current.ny
              << " time steps: " << nsteps << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average temperature at start: " << average_temp << std::endl;


    const double a = 0.5;     // Diffusion constant
    auto dx2 = current.dx * current.dx;
    auto dy2 = current.dy * current.dy;
    // Largest stable time step
    auto dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    //Get the start time stamp
    using clock = std::chrono::high_resolution_clock;

    auto start_clock = clock::now();

    // Time evolve
    for (int iter = 1; iter <= nsteps; iter++) {
        evolve(current, previous, a, dt);
        if (iter % image_interval == 0) {
            write_field(current, iter);
        }
        // Swap current field so that it will be used
        // as previous for next iteration step
        std::swap(current, previous);
    }

    auto stop_clock = clock::now();
    std::chrono::duration<double> elapsed_seconds = stop_clock - start_clock;

    // Average temperature for reference
    average_temp = average(previous);

    std::cout << "Iteration took " << elapsed_seconds.count()
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
