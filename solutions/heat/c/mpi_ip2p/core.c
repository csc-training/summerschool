/* Main solver routines for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat.h"

/* Exchange the boundary values */
void exchange(field *temperature, parallel_data *parallel)
{
    MPI_Request reqs[4];

    // Send to the up, receive from down
    MPI_Isend(temperature->data[1], temperature->ny + 2, MPI_DOUBLE,
              parallel->nup, 11, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(temperature->data[temperature->nx + 1],
              temperature->ny + 2, MPI_DOUBLE, parallel->ndown, 11,
              MPI_COMM_WORLD, &reqs[1]);
    // Send to the down, receive from up
    MPI_Isend(temperature->data[temperature->nx], temperature->ny + 2,
              MPI_DOUBLE, parallel->ndown, 12, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(temperature->data[0], temperature->ny + 2, MPI_DOUBLE,
              parallel->nup, 12, MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}


/* Update the temperature values using five-point stencil */
void evolve(field *curr, field *prev, double a, double dt)
{
    int i, j;
    double dx2, dy2;

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    dx2 = prev->dx * prev->dx;
    dy2 = prev->dy * prev->dy;
    for (i = 1; i < curr->nx + 1; i++) {
        for (j = 1; j < curr->ny + 1; j++) {
            curr->data[i][j] = prev->data[i][j] + a * dt *
                               ((prev->data[i + 1][j] -
                                 2.0 * prev->data[i][j] +
                                 prev->data[i - 1][j]) / dx2 +
                                (prev->data[i][j + 1] -
                                 2.0 * prev->data[i][j] +
                                 prev->data[i][j - 1]) / dy2);
        }
    }
}


