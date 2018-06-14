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
    // TODO
    // Send to the down, receive from up
    // TODO
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
    /* TODO */
}

/* complete the non-blocking communication */
/* TODO */(
    void exchange_finalize(parallel_data *parallel)
{

}

/* Update the temperature values using five-point stencil */
/* update only the border-dependent regions of the field */
/* TODO */
void evolve_edges(field *curr, field *prev, double a, double dt)
{
    int i, j;
    double dx2, dy2;

}
