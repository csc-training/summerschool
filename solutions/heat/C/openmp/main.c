/* Heat equation solver in 2D. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "heat.h"


int main(int argc, char **argv)
{
    double a = 0.5;             //!< Diffusion constant
    field current, previous;    //!< Current and previous temperature fields

    double dt;                  //!< Time step
    int nsteps;                 //!< Number of time steps

    int image_interval = 10;    //!< Image output interval

    int iter;                   //!< Iteration counter

    double dx2, dy2;            //!< delta x and y squared

    clock_t start_clock;        //!< Time stamps

#pragma omp parallel private(iter)
{
    initialize(argc, argv, &current, &previous, &nsteps);

#pragma omp single
    {
        /* Output the initial field */
        write_field(&current, 0);

        /* Largest stable time step */
        dx2 = current.dx * current.dx;
        dy2 = current.dy * current.dy;
        dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

        /* Get the start time stamp */
        start_clock = clock();
    }

    /* Time evolve */
    for (iter = 1; iter < nsteps; iter++) {
        evolve(&current, &previous, a, dt);
        if (iter % image_interval == 0) {
#pragma omp single
            write_field(&current, iter);
        }
        /* Swap current field so that it will be used
            as previous for next iteration step */
#pragma omp single
        swap_fields(&current, &previous);
    }
} /* End of parallel region */

    /* Determine the CPU time used for the iteration */
    printf("Iteration took %.3f seconds.\n", (double)(clock() - start_clock) /
        (double)CLOCKS_PER_SEC);
    printf("Reference value at 5,5: %f\n", previous.data[5][5]);

    finalize(&current, &previous);
    return 0;
}
