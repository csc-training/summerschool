/* Setup routines for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "heat.h"
#include "../../common/pngwriter.h"

#define NSTEPS 500  // Default number of iteration steps

/* Initialize the heat equation solver */
void initialize(int argc, char* argv[], field *current,
                field *previous, int *nsteps)
{
    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a given file
     * Two arguments:   initial field from file and number of time steps
     * Three arguments: field dimensions (rows,cols) and number of time steps
     */


    int rows = 200;             //!< Field dimensions with default values
    int cols = 200;

    char input_file[64];        //!< Name of the optional input file

    int read_file = 0;

    *nsteps = NSTEPS;

    switch (argc) {
    case 1:
        /* Use default values */
        break;
    case 2:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_file = 1;
        break;
    case 3:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_file = 1;

        /* Number of time steps */
        *nsteps = atoi(argv[2]);
        break;
    case 4:
        /* Field dimensions */
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        /* Number of time steps */
        *nsteps = atoi(argv[3]);
        break;
    default:
        printf("Unsupported number of command line arguments\n");
        exit(-1);
    }

    if (read_file)
        read_field(current, previous, input_file);
    else
      {
        set_field_dimensions(current, rows, cols);
        set_field_dimensions(previous, rows, cols);
        generate_field(current);
        generate_field(previous);
      }
}

/* Generate initial temperature field.  Pattern is disc with a radius
 * of nx_full / 6 in the center of the grid.
 * Boundary conditions are (different) constant temperatures outside the grid */
void generate_field(field *temperature)
{
    int i, j;
    double radius;
    int dx, dy;

    /* Allocate the temperature array, note that
     * we have to allocate also the ghost layers */
    temperature->data =
        malloc_2d(temperature->nx + 2, temperature->ny + 2);

    /* Radius of the source disc */
    radius = temperature->nx / 6.0;
    for (i = 0; i < temperature->nx + 2; i++) {
        for (j = 0; j < temperature->ny + 2; j++) {
            /* Distances of point i, j from the origin */
            dx = i - temperature->nx / 2 + 1;
            dy = j - temperature->ny / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                temperature->data[i][j] = 5.0;
            } else {
                temperature->data[i][j] = 65.0;
            }
        }
    }

    /* Boundary conditions */
    for (i = 0; i < temperature->nx + 2; i++) {
        temperature->data[i][0] = 20.0;
        temperature->data[i][temperature->ny + 1] = 70.0;
    }

    for (j = 0; j < temperature->ny + 2; j++) {
        temperature->data[0][j] = 85.0;
        temperature->data[temperature->nx + 1][j] = 5.0;
    }

}

/* Set dimensions of the field. Note that the nx is the size of the first
 * dimension and ny the second. */
void set_field_dimensions(field *temperature, int nx, int ny)
{
    temperature->dx = DX;
    temperature->dy = DY;
    temperature->nx = nx;
    temperature->ny = ny;
}

/* Deallocate the 2D arrays of temperature fields */
void finalize(field *temperature1, field *temperature2)
{
    free_2d(temperature1->data);
    free_2d(temperature2->data);
}

