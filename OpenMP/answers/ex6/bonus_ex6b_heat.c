
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "heat.h"
#include "pngwriter.h"

/* Utility routine for allocating a two dimensional array */
double **malloc_2d(int nx, int ny)
{
    double **array;
    int i;

    array = (double **) malloc(nx * sizeof(double *));
    array[0] = (double *) malloc(nx * ny * sizeof(double));

    for (i = 1; i < nx; i++) {
        array[i] = array[0] + i * ny;
    }

    return array;
}

/* Utility routine for deallocating a two dimensional array */
void free_2d(double **array)
{
    free(array[0]);
    free(array);
}

/* Initialize the metadata. Note that the nx is the size of the first
 * dimension and ny the second. */
void initialize_field_metadata(field *temperature, int nx, int ny)
{
    temperature->dx = DX;
    temperature->dy = DY;
    temperature->dx2 = DX * DX;
    temperature->dy2 = DY * DY;
    temperature->nx = nx;
    temperature->ny = ny;
}

/* Copy data on temperature1 into temperature2 */
void copy_field(field *temperature1, field *temperature2)
{
    assert(temperature1->nx == temperature2->nx);
    assert(temperature1->ny == temperature2->ny);
    memcpy(temperature2->data[0], temperature1->data[0],
           (temperature1->nx + 2) * (temperature1->ny + 2) * sizeof(double));
}

/* Swap the data of fields temperature1 and temperature2 */
void swap_fields(field *temperature1, field *temperature2)
{
    double **tmp;
    tmp = temperature1->data;
    temperature1->data = temperature2->data;
    temperature2->data = tmp;
}

/* Initialize the temperature field.  Pattern is disc with a radius
 * of nx_full / 6 in the center of the grid.
 * Boundary conditions are (different) constant temperatures outside the grid */
void initialize(field *temperature1, field *temperature2)
{
    int i, j;
    double radius;
    int dx, dy;

    /* Allocate the temperature arrays, note that
     * we have to allocate also the ghost layers */
#pragma omp single
    {
    temperature1->data =
        malloc_2d(temperature1->nx + 2, temperature1->ny + 2);
    temperature2->data =
        malloc_2d(temperature2->nx + 2, temperature2->ny + 2);
    }

    /* Radius of the source disc */
    radius = temperature1->nx / 6.0;

#pragma omp for
    for (i = 0; i < temperature1->nx + 2; i++) {
        for (j = 0; j < temperature1->ny + 2; j++) {
            /* Distances of point i, j from the origin */
            dx = i - temperature1->nx / 2 + 1;
            dy = j - temperature1->ny / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                temperature1->data[i][j] = 5.0;
            } else {
                temperature1->data[i][j] = 65.0;
            }
        }
    }

    /* Boundary conditions */
#pragma omp for
    for (i = 0; i < temperature1->nx + 2; i++) {
        temperature1->data[i][0] = 20.0;
        temperature1->data[i][temperature1->ny + 1] = 70.0;
    }

#pragma omp for
    for (j = 0; j < temperature1->ny + 2; j++) {
        temperature1->data[0][j] = 85.0;
        temperature1->data[temperature1->nx + 1][j] = 5.0;
    }

#pragma omp single    
    copy_field(temperature1, temperature2);

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

/* Deallocate the 2D arrays of temperature fields */
void finalize(field *temperature1, field *temperature2)
{
    free_2d(temperature1->data);
    free_2d(temperature2->data);
}


/* Output routine that prints out a picture of the temperature
 * distribution. */
void output(field *temperature, int iter)
{
    char filename[64];

    /* The actual write routine takes only the actual data
     * (without ghost layers) so we need array for that. */
    int height, width;
    double **full_data;

    int i;

    height = temperature->nx;
    width = temperature->ny;

    /* Copy the inner data */
    full_data = malloc_2d(height, width);
    for (i = 0; i < temperature->nx; i++)
        memcpy(full_data[i], &temperature->data[i + 1][1],
               temperature->ny * sizeof(double));

    sprintf(filename, "%s_%04d.png", "heat", iter);
    save_png(full_data[0], height, width, filename, 'c');

    free_2d(full_data);
}

/* Read the initial temperature distribution from a file and
 * initialize the temperature fields temperature1 and
 * temperature2 to the same initial state. */
void read_input(field *temperature1, field *temperature2, char *filename)
{
    FILE *fp;
    int nx, ny, i, j;
    int count;

    fp = fopen(filename, "r");
    /* Read the header */
    count = fscanf(fp, "# %d %d \n", &nx, &ny);
    if (count < 2) {
        fprintf(stderr, "Error while reading the input file!\n");
        exit(EXIT_FAILURE);
    }

    initialize_field_metadata(temperature1, nx, ny);
    initialize_field_metadata(temperature2, nx, ny);

    /* Allocate arrays (including ghost layers */
    temperature1->data = malloc_2d(nx + 2, ny + 2);
    temperature2->data = malloc_2d(nx + 2, ny + 2);

    /* Read the actual data */
    for (i = 1; i < nx + 1; i++) {
        for (j = 1; j < ny + 1; j++) {
            fscanf(fp, "%lf", &temperature1->data[i][j]);
        }
    }

    /* Set the boundary values */
    for (i = 1; i < nx + 1; i++) {
        temperature1->data[i][0] = temperature1->data[i][1];
        temperature1->data[i][ny + 1] = temperature1->data[i][ny];
    }
    for (j = 0; j < ny + 2; j++) {
        temperature1->data[0][j] = temperature1->data[1][j];
        temperature1->data[nx + 1][j] = temperature1->data[nx][j];
    }

    copy_field(temperature1, temperature2);

    fclose(fp);
}
