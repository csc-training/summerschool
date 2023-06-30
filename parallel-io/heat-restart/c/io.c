/* I/O related functions for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat.h"
#include "../common/pngwriter.h"

/* Output routine that prints out a picture of the temperature
 * distribution. */
void write_field(field *temperature, int iter, parallel_data *parallel)
{
    char filename[64];

    /* The actual write routine takes only the actual data
     * (without ghost layers) so we need array for that. */
    int height, width;
    double **full_data;
    double **tmp_data;          // array for MPI sends and receives

    int i, p;

    height = temperature->nx * parallel->size;
    width = temperature->ny;

    tmp_data = malloc_2d(temperature->nx, temperature->ny);

    if (parallel->rank == 0) {
        /* Copy the inner data */
        full_data = malloc_2d(height, width);
        for (i = 0; i < temperature->nx; i++)
            memcpy(full_data[i], &temperature->data[i + 1][1],
                   temperature->ny * sizeof(double));
        /* Receive data from other ranks */
        for (p = 1; p < parallel->size; p++) {
            MPI_Recv(&tmp_data[0][0], temperature->nx * temperature->ny,
                     MPI_DOUBLE, p, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* Copy data to full array */
            memcpy(&full_data[p * temperature->nx][0], tmp_data[0],
                   temperature->nx * temperature->ny * sizeof(double));
        }
        /* Write out the data to a png file */
        sprintf(filename, "%s_%04d.png", "heat", iter);
        save_png(full_data[0], height, width, filename, 'c');
        free_2d(full_data);
    } else {
        /* Send data */
        for (i = 0; i < temperature->nx; i++)
            memcpy(tmp_data[i], &temperature->data[i + 1][1],
                   temperature->ny * sizeof(double));
        MPI_Send(&tmp_data[0][0], temperature->nx * temperature->ny,
                 MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
    }

    free_2d(tmp_data);

}

/* Read the initial temperature distribution from a file and
 * initialize the temperature fields temperature1 and
 * temperature2 to the same initial state. */
void read_field(field *temperature1, field *temperature2, char *filename,
                parallel_data *parallel)
{
    FILE *fp;
    int nx, ny, i, j;
    double **full_data;
    double **inner_data;

    int nx_local, count;

    fp = fopen(filename, "r");
    /* Read the header */
    count = fscanf(fp, "# %d %d \n", &nx, &ny);
    if (count < 2) {
        fprintf(stderr, "Error while reading the input file!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    parallel_setup(parallel, nx, ny);
    set_field_dimensions(temperature1, nx, ny, parallel);
    set_field_dimensions(temperature2, nx, ny, parallel);

    /* Allocate arrays (including ghost layers) */
    temperature1->data =
        malloc_2d(temperature1->nx + 2, temperature1->ny + 2);
    temperature2->data =
        malloc_2d(temperature2->nx + 2, temperature2->ny + 2);

    inner_data = malloc_2d(temperature1->nx, temperature1->ny);

    if (parallel->rank == 0) {
        /* Full array */
        full_data = malloc_2d(nx, ny);

        /* Read the actual data */
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                count = fscanf(fp, "%lf", &full_data[i][j]);
            }
        }
    } else {
        /* Dummy array for full data. Some MPI implementations
         * require that this array is actually allocated... */
        full_data = malloc_2d(1, 1);
    }

    nx_local = temperature1->nx;

    MPI_Scatter(full_data[0], nx_local * ny, MPI_DOUBLE, inner_data[0],
                nx_local * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Copy to the array containing also boundaries */
    for (i = 0; i < nx_local; i++)
        memcpy(&temperature1->data[i + 1][1], &inner_data[i][0],
               ny * sizeof(double));

    /* Set the boundary values */
    for (i = 1; i < nx_local + 1; i++) {
        temperature1->data[i][0] = temperature1->data[i][1];
        temperature1->data[i][ny + 1] = temperature1->data[i][ny];
    }
    for (j = 0; j < ny + 2; j++) {
        temperature1->data[0][j] = temperature1->data[1][j];
        temperature1->data[nx_local + 1][j] =
            temperature1->data[nx_local][j];
    }

    copy_field(temperature1, temperature2);

    free_2d(full_data);
    free_2d(inner_data);
    fclose(fp);
}

/* Write an HDF5 restart checkpoint file that contains the current
 * iteration number and temperature field. */
void write_restart(field *temperature, parallel_data *parallel, int iter)
{
    // TODO: create a file called CHECKPOINT (defined in heat.h)

    // TODO: define a dataspace with the dimensions of the full
    //   temperature field and create the dataset.

    // TODO: use hyperslabs to define the part of the file that
    //   each rank writes.

    // TODO: define a dataspace with the dimensions of a local
    //   domain for each rank and user hyperslabs to select
    //   the part containing the data (not including the ghost
    //   cells).

    // TODO: write data using a collective write operation.

    // TODO: write the current iteration number as an
    //   attribute to the dataset containing the temperature
    //   field.

    // TODO: close all handles.
}

/* Read an HDF restart checkpoint file that contains the current
 * iteration number and temperature field. */
void read_restart(field *temperature, parallel_data *parallel, int *iter)
{
    // TODO: open the file called CHECKPOINT (defined in heat.h)

    // TODO: open the dataset containing the temperature field.

    // TODO: read the dimensions of the dataspace and
    //   set correct dimensions to MPI metadata
    parallel_setup(parallel, dim[0], dim[1]);

    //   set local dimensions and allocate memory for the data
    set_field_dimensions(temperature, dim[0], dim[1], parallel);
    allocate_field(temperature);

    // TODO: use hyperslabs to define the part of the file that
    //   each rank reads.

    // TODO: in each rank create a dataspace to read the data into
    //   and use hyperslabs to define the part containing the
    //   data.

    // TODO: read the data using a collective read.

    // TODO: read the attribute containing the number of
    //   current iteration.

    // TODO: close all handles.
}
