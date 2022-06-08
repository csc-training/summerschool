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
    double *full_data;

    height = temperature->nx;
    width = temperature->ny;

    /* Copy the inner data */
    full_data = new double [height * width];
    for (int i = 0; i < temperature->nx; i++)
        memcpy(&full_data[i * width], &temperature->data[(i + 1) * (width + 2) + 1],
           temperature->ny * sizeof(double));

    /* Write out the data to a png file */
    sprintf(filename, "%s_%04d.png", "heat", iter);
    save_png(full_data, height, width, filename, 'c');
    delete[] full_data;
}

/* Read the initial temperature distribution from a file and
 * initialize the temperature fields temperature1 and
 * temperature2 to the same initial state. */
void read_field(field *temperature1, field *temperature2, char *filename)
{
    FILE *fp;
    int nx, ny, ind;
    double *full_data;

    int nx_local, ny_local, count;

    fp = fopen(filename, "r");
    /* Read the header */
    count = fscanf(fp, "# %d %d \n", &nx, &ny);
    if (count < 2) {
        fprintf(stderr, "Error while reading the input file!\n");
    exit(-1);
    }

    set_field_dimensions(temperature1, nx, ny);
    set_field_dimensions(temperature2, nx, ny);

    /* Allocate arrays (including ghost layers) */
    temperature1->data = new double[(temperature1->nx + 2) * (temperature1->ny + 2)];
    temperature2->data = new double[(temperature1->nx + 2) * (temperature1->ny + 2)];

    /* Full array */
    full_data = new double [nx * ny];

    /* Read the actual data */
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ind = i * ny + j;
            count = fscanf(fp, "%lf", &full_data[ind]);
        }
    }

    nx_local = temperature1->nx;
    ny_local = temperature1->ny;

    /* Copy to the array containing also boundaries */
    for (int i = 0; i < nx_local; i++)
      memcpy(&temperature1->data[(i + 1) * (ny_local + 2) + 1], &full_data[i * ny_local],
               ny * sizeof(double));

    /* Set the boundary values */
    for (int i = 1; i < nx_local + 1; i++) {
        temperature1->data[i * (ny_local + 2)] = temperature1->data[i * (ny_local + 2) + 1];
        temperature1->data[i * (ny_local + 2) + ny + 1] = temperature1->data[i * (ny_local + 2) + ny];
    }
    for (int j = 0; j < ny + 2; j++) {
        temperature1->data[j] = temperature1->data[ny_local + j];
        temperature1->data[(nx_local + 1) * (ny_local + 2) + j] =
            temperature1->data[nx_local * (ny_local + 2) + j];
    }

    copy_field(temperature1, temperature2);

    delete[] full_data;
    fclose(fp);
}
