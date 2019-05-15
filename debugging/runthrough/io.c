/* I/O related functions for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "heat.h"
#include "pngwriter.h"

/* Output routine that prints out a picture of the temperature
 * distribution. */
void write_field(field *temperature, int iter)
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
void read_field(field *temperature1, field *temperature2, char *filename)
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

    set_field_dimensions(temperature1, nx, ny);
    set_field_dimensions(temperature2, nx, ny);

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
