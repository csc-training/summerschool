#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include <time.h>

#include "heat_mpi.h"

int main(int argc, char **argv)
{
    double a = 0.5;             //!< Diffusion constant
    field current, previous;    //!< Current and previous temperature fields

    double dt;                  //!< Time step
    int nsteps = 500;           //!< Number of time steps

    int rows = 200;             //!< Field dimensions with default values
    int cols = 200;

    char input_file[64];        //!< Name of the optional input file

    int image_interval = 10;    //!< Image output interval

    parallel_data parallelization; //!< Parallelization info

    int iter;                   //!< Iteration counter

    double dx2, dy2;            //!< delta x and y squared

    double start_clock;        //!< Time stamps

    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a given file
     * Two arguments:   initial field from file and number of time steps
     * Three arguments: field dimensions (rows,cols) and number of time steps
     */

    MPI_Init(&argc, &argv);

    switch (argc) {
    case 1:
        /* Use default values */
        parallel_initialize(&parallelization, rows, cols);
        initialize_field_metadata(&current, rows, cols,
                                  &parallelization);
        initialize_field_metadata(&previous, rows, cols,
                                  &parallelization);
        initialize(&current, &previous, &parallelization);
        break;
    case 2:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_input(&current, &previous, input_file, &parallelization);
        break;
    case 3:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_input(&current, &previous, input_file, &parallelization);
        /* Number of time steps */
        nsteps = atoi(argv[2]);
        break;
    case 4:
        /* Field dimensions */
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        parallel_initialize(&parallelization, cols, rows);
        initialize_field_metadata(&current, cols, rows,
                                  &parallelization);
        initialize_field_metadata(&previous, cols, rows,
                                  &parallelization);
        initialize(&current, &previous, &parallelization);
        /* Number of time steps */
        nsteps = atoi(argv[3]);
        break;
    default:
        printf("Unsupported number of command line arguments\n");
        return -1;
    }

    /* Output the initial field */
    output(&current, 0, &parallelization);

    /* Largest stable time step */
    dx2 = current.dx * current.dx;
    dy2 = current.dy * current.dy;
    dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    /* Get the start time stamp */
    if (parallelization.rank == 0) {
        start_clock = MPI_Wtime();
    }

    /* Time evolve */
    for (iter = 1; iter <= nsteps; iter++) {
        exchange(&previous, &parallelization);
        evolve(&current, &previous, a, dt);
        /* Output once in every 10 iterations */
        if (iter % image_interval == 0) {
            output(&current, iter, &parallelization);
        }
        /* Swap current field so that it will be used
         * as previous for next iteration step */
        swap_fields(&current, &previous);
    }

    /* Determine the CPU time used for the iteration */
    if (parallelization.rank == 0) {
        printf("Iteration took %.3f seconds.\n", MPI_Wtime() - start_clock);
        printf("Reference value at 5,5: %f\n", previous.data[5][5]);
    }

    finalize(&current, &previous, &parallelization);
    MPI_Finalize();

    return 0;
}

