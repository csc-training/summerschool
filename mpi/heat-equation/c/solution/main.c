/* Heat equation solver in 2D. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "heat.h"

int main(int argc, char **argv)
{
    double a = 0.5;             //!< Diffusion constant
    field current, previous;    //!< Current and previous temperature fields

    double dt;                  //!< Time step
    int nsteps;                 //!< Number of time steps

    int image_interval = 1500;    //!< Image output interval

    parallel_data parallelization; //!< Parallelization info

    double dx2, dy2;            //!< Delta x and y squared

    double average_temp;        //!< Average temperature

    double start_clock, stop_clock;  //!< Time stamps


    MPI_Init(&argc, &argv);

    initialize(argc, argv, &current, &previous, &nsteps, &parallelization);

    /* Output the initial field */
    write_field(&current, 0, &parallelization);

    average_temp = average(&current);
    if (parallelization.rank == 0) {
        printf("Average temperature at start: %f\n", average_temp);
    }    


    /* Largest stable time step */
    dx2 = current.dx * current.dx;
    dy2 = current.dy * current.dy;
    dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    /* Get the start time stamp */
    start_clock = MPI_Wtime();

    /* Time evolve */
    for (int iter = 1; iter <= nsteps; iter++) {
        exchange(&previous, &parallelization);
        evolve(&current, &previous, a, dt);
        if (iter % image_interval == 0) {
            write_field(&current, iter, &parallelization);
        }
        /* Swap current field so that it will be used
            as previous for next iteration step */
        swap_fields(&current, &previous);
    }

    stop_clock = MPI_Wtime();

    /* Average temperature for reference */
    average_temp = average(&previous);

    /* Determine the CPU time used for the iteration */
    if (parallelization.rank == 0) {
        printf("Iteration took %.3f seconds.\n", (stop_clock - start_clock));
        printf("Average temperature: %f\n", average_temp);
        if (argc == 1) {
            printf("Reference value with default arguments: 59.281239\n");
        }
    }

    /* Output the final field */
    write_field(&previous, nsteps, &parallelization);

    finalize(&current, &previous);
    MPI_Finalize();

    return 0;
}
