/* Main solver routines for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat.h"

/* Exchange the boundary values */
/* only start the communication in background */
void exchange_init(field *temperature, parallel_data *parallel)
{
    // Send to the up, receive from down
    MPI_Isend(temperature->data[1], temperature->ny + 2, MPI_DOUBLE,
              parallel->nup, 11, MPI_COMM_WORLD, &parallel->requests[0]);
    MPI_Irecv(temperature->data[temperature->nx + 1],
              temperature->ny + 2, MPI_DOUBLE, parallel->ndown, 11,
              MPI_COMM_WORLD, &parallel->requests[1]);
    // Send to the down, receive from up
    MPI_Isend(temperature->data[temperature->nx], temperature->ny + 2,
              MPI_DOUBLE, parallel->ndown, 12, MPI_COMM_WORLD,
              &parallel->requests[2]);
    MPI_Irecv(temperature->data[0], temperature->ny + 2, MPI_DOUBLE,
              parallel->nup, 12, MPI_COMM_WORLD, &parallel->requests[3]);
}


/* Update the temperature values using five-point stencil */
/* update only the border-independent region of the field */
void evolve_interior(field *curr, field *prev, double a, double dt)
{
    int i, j;
    double dx2, dy2;

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    dx2 = prev->dx * prev->dx;
    dy2 = prev->dy * prev->dy;
    for (i = 2; i < curr->nx; i++) {
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

/* complete the non-blocking communication */
void exchange_finalize(parallel_data *parallel)
{
    MPI_Waitall(4, &parallel->requests[0], MPI_STATUSES_IGNORE);
}

/* Update the temperature values using five-point stencil */
/* update only the border-dependent regions of the field */
void evolve_edges(field *curr, field *prev, double a, double dt)
{
    int i, j;
    double dx2, dy2;

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    dx2 = prev->dx * prev->dx;
    dy2 = prev->dy * prev->dy;
    i = 1;
    for (j = 1; j < curr->ny + 1; j++) {
        curr->data[i][j] = prev->data[i][j] + a * dt *
                           ((prev->data[i + 1][j] -
                             2.0 * prev->data[i][j] +
                             prev->data[i - 1][j]) / dx2 +
                            (prev->data[i][j + 1] -
                             2.0 * prev->data[i][j] +
                             prev->data[i][j - 1]) / dy2);
    }
    i = curr -> nx;
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
