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
    MPI_Sendrecv(temperature->data[1], temperature->ny + 2, MPI_DOUBLE,
                 parallel->nup, 11,
                 temperature->data[temperature->nx + 1],
                 temperature->ny + 2, MPI_DOUBLE, parallel->ndown, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Send to the down, receive from up
    MPI_Sendrecv(temperature->data[temperature->nx], temperature->ny + 2,
                 MPI_DOUBLE, parallel->ndown, 12,
                 temperature->data[0], temperature->ny + 2, MPI_DOUBLE,
                 parallel->nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


/* Update the temperature values using five-point stencil */
void evolve(field *curr, field *prev, double a, double dt)
{
    int i, j;
    double dx2, dy2;
    int nx, ny;
    double **currdata, **prevdata;

    /* HINT: to help the compiler do not access members of structures
     * within OpenACC parallel regions */
    currdata = curr->data;
    prevdata = prev->data;
    nx = curr->nx;
    ny = curr->ny;

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    dx2 = prev->dx * prev->dx;
    dy2 = prev->dy * prev->dy;
#pragma acc parallel loop private(i,j) copyin(prevdata[0:nx+2][0:ny+2]) \
    copyout(currdata[0:nx+2][0:ny+2]) collapse(2)
    for (i = 1; i < nx + 1; i++) {
        for (j = 1; j < ny + 1; j++) {
            currdata[i][j] = prevdata[i][j] + a * dt *
                             ((prevdata[i + 1][j] -
                               2.0 * prevdata[i][j] +
                               prevdata[i - 1][j]) / dx2 +
                              (prevdata[i][j + 1] -
                               2.0 * prevdata[i][j] +
                               prevdata[i][j - 1]) / dy2);
        }
    }
}


