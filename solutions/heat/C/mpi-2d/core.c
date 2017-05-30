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
    // Send to the up, receive from down
    MPI_Sendrecv(temperature->data[1], 1, parallel->rowtype,
                 parallel->nup, 11,
                 temperature->data[temperature->nx + 1], 1,
                 parallel->rowtype, parallel->ndown, 11, parallel->comm,
                 MPI_STATUS_IGNORE);
    // Send to the down, receive from up
    MPI_Sendrecv(temperature->data[temperature->nx], 1,
                 parallel->rowtype, parallel->ndown, 12,
                 temperature->data[0], 1, parallel->rowtype,
                 parallel->nup, 12, parallel->comm, MPI_STATUS_IGNORE);
    // Send to the left, receive from right
    MPI_Sendrecv(&temperature->data[0][1], 1, parallel->columntype,
                 parallel->nleft, 13,
                 &temperature->data[0][temperature->ny + 1], 1,
                 parallel->columntype, parallel->nright, 13,
                 parallel->comm, MPI_STATUS_IGNORE);
    // Send to the right, receive from left
    MPI_Sendrecv(&temperature->data[0][temperature->ny], 1,
                 parallel->columntype,
                 parallel->nright, 14, &temperature->data[0][0], 1,
                 parallel->columntype,
                 parallel->nleft, 14, parallel->comm, MPI_STATUS_IGNORE);

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


