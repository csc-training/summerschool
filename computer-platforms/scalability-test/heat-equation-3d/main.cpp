/*****************************************************************************
MIT License

Copyright (c) 2021 CSC HPC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

// Simple 3d heat equation solver

#include <string>
#include <iostream>
#include <iomanip>
#ifndef NO_MPI
#include <mpi.h>
#ifdef CHECK_GPU_MPI
#include <mpi-ext.h> // Needed for CUDA-aware check
#endif
#endif
#include "heat.hpp"
#include "parallel.hpp"
#include "functions.hpp"

#include <hip/hip_runtime_api.h>

int main(int argc, char **argv)
{

#ifndef NO_MPI
    MPI_Init(&argc, &argv);
#ifdef CHECK_GPU_MPI
    if (1 != MPIX_Query_cuda_support()) {
        std::cout << "CUDA aware MPI required" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
#endif
#endif

    const int image_interval = 15000;    // Image output interval

    ParallelData parallelization; // Parallelization info

    int nsteps;                 // Number of time steps
    Field current, previous;    // Current and previous temperature fields
    initialize(argc, argv, current, previous, nsteps, parallelization);

// Create streams for computing (edge computation with three different streams)
    hipStream_t streams[3];
    for (int i=0; i < 3; i++)
      GPU_CHECK( hipStreamCreateWithFlags(&streams[i],hipStreamNonBlocking) );

    // Output the initial field
    write_field(current, 0, parallelization);

    auto average_temp = average(current);
    if (0 == parallelization.rank) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average temperature at start: " << average_temp << std::endl;
    }    

    const double a = 0.5;     // Diffusion constant 
    auto dx2 = current.dx * current.dx;
    auto dy2 = current.dy * current.dy;
    auto dz2 = current.dz * current.dz;
    // Largest stable time step 
    auto dt = dx2 * dy2 * dz2 / (2.0 * a * (dx2 + dy2 + dz2));

    allocate_data(current, previous);
    //Get the start time stamp 
    auto start_clock = timer();

    auto start_mem = timer();
    enter_data(current, previous);
    auto t_mem = timer() - start_mem;

    double start_mpi, start_comp;
    double t_mpi = 0.0;
    double t_comp = 0.0;

    // Time evolve
    for (int iter = 1; iter <= nsteps; iter++) {
        start_comp = timer();
        evolve_interior(current, previous, a, dt, streams);
        t_comp += timer() - start_comp;
        start_mpi = timer();
        exchange_init(previous, parallelization);
        t_mpi += timer() - start_mpi;
        start_mpi = timer();
        exchange_finalize(previous, parallelization);
        t_mpi += timer() - start_mpi;
        start_comp = timer();
        evolve_edges(current, previous, a, dt, streams);
        t_comp += timer() - start_comp;
        if (iter % image_interval == 0) {
            update_host(current);
            write_field(current, iter, parallelization);
        }
        // Swap current field so that it will be used
        // as previous for next iteration step
        std::swap(current, previous);
    }

    start_mem = timer();
    exit_data(current, previous);
    t_mem += timer() - start_mem;

    auto stop_clock = timer();

    free_data(current, previous);

    // Average temperature for reference 
    average_temp = average(previous);

    if (0 == parallelization.rank) {
        std::cout << "Iteration took " << (stop_clock - start_clock)
                  << " seconds." << std::endl;
        std::cout << "  Memory copies " << t_mem << " s." << std::endl;
        std::cout << "  MPI           " << t_mpi << " s." << std::endl;
        std::cout << "  Compute       " << t_comp << " s." << std::endl;
        std::cout << "Average temperature: " << average_temp << std::endl;
        if (1 == argc) {
            std::cout << "Reference value with default arguments: " 
                      << 63.834223 << std::endl;
        }
    }

    // Output the final field
    write_field(previous, nsteps, parallelization);

    for (int i=0; i < 3; i++)
      GPU_CHECK( hipStreamDestroy(streams[i]) );

#ifndef NO_MPI
    MPI_Finalize();
#endif

    return 0;
}
