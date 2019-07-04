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

    /* Copy the updated boundary to the device */
    update_device_boundary(temperature);
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
#pragma acc parallel loop private(i,j) \
    present(prevdata[0:nx+2][0:ny+2], currdata[0:nx+2][0:ny+2]) collapse(2)
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

    /* Copy the updated boundary to the host */
#pragma acc update host(currdata[1:1][0:ny+2], currdata[nx:1][0:ny+2])
}

/* Start a data region and copy temperature fields to the device */
void enter_data(field *curr, field *prev)
{
    int nx, ny;
    double **currdata, **prevdata;

    currdata = curr->data;
    prevdata = prev->data;
    nx = curr->nx;
    ny = curr->ny;

#pragma acc enter data \
    copyin(currdata[0:nx+2][0:ny+2], prevdata[0:nx+2][0:ny+2])
}

/* End a data region and copy temperature fields back to the host */
void exit_data(field *curr, field *prev)
{
    int nx, ny;
    double **currdata, **prevdata;

    currdata = curr->data;
    prevdata = prev->data;
    nx = curr->nx;
    ny = curr->ny;

#pragma acc exit data \
    copyout(currdata[0:nx+2][0:ny+2], prevdata[0:nx+2][0:ny+2])
}

/* Copy a temperature field from the device to the host */
void update_host(field *temperature)
{
    int nx, ny;
    double **data;

    data = temperature->data;
    nx = temperature->nx;
    ny = temperature->ny;

#pragma acc update host(data[0:nx+2][0:ny+2], data[0:nx+2][0:ny+2])
}

/* Copy the outer boundary values from the host to the device */
void update_device_boundary(field *temperature)
{
    int nx, ny;
    double **data;

    data = temperature->data;
    nx = temperature->nx;
    ny = temperature->ny;

#pragma acc update device(data[0:1][0:ny+2], data[nx+1:1][0:ny+2])
}
