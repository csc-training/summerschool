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
    double *data;
    double *sbuf_up, *sbuf_down, *rbuf_up, *rbuf_down;

    data = temperature->devdata;

    // Send to the up, receive from down
    sbuf_up = data + temperature->ny + 2; // upper data
    rbuf_down = data + (temperature->nx + 1) * (temperature->ny + 2); // lower halo

    MPI_Sendrecv(sbuf_up, temperature->ny + 2, MPI_DOUBLE,
                 parallel->nup, 11,
                 rbuf_down, temperature->ny + 2, MPI_DOUBLE,
                 parallel->ndown, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to the down, receive from up
    sbuf_down = data + temperature->nx * (temperature->ny + 2); // lower data
    rbuf_up = data; // upper halo

    MPI_Sendrecv(sbuf_down, temperature->ny + 2, MPI_DOUBLE,
                 parallel->ndown, 12,
                 rbuf_up, temperature->ny + 2, MPI_DOUBLE,
                 parallel->nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}
