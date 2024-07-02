/* I/O related functions for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat.h"
#include "pngwriter.h"

/* Output routine that prints out a picture of the temperature
 * distribution. */
void write_field(field *temperature, int iter, parallel_data *parallel)
{
    char filename[64];

    /* The actual write routine takes only the actual data
     * (without ghost layers) so we need array for that. */
    int height, width;
    double *full_data;
    double *tmp_data;          // array for MPI sends and receives

    height = temperature->nx * parallel->size;
    width = temperature->ny;

    tmp_data = new double [temperature->nx * temperature->ny];

    if (parallel->rank == 0) {
        /* Copy the inner data */
        full_data = new double [height * width];
        for (int i = 0; i < temperature->nx; i++)
	  memcpy(&full_data[i * width], &temperature->data[(i + 1) * (width + 2) + 1],
                   temperature->ny * sizeof(double));
        /* Receive data from other ranks */
        for (int p = 1; p < parallel->size; p++) {
            MPI_Recv(tmp_data, temperature->nx * temperature->ny,
                     MPI_DOUBLE, p, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* Copy data to full array */
            memcpy(&full_data[p * temperature->nx * width], tmp_data,
                   temperature->nx * temperature->ny * sizeof(double));
        }
        /* Write out the data to a png file */
        sprintf(filename, "%s_%04d.png", "heat", iter);
        save_png(full_data, height, width, filename, 'c');
        delete[] full_data;
        // free(full_data);
    } else {
        /* Send data */
        for (int i = 0; i < temperature->nx; i++)
	  memcpy(&tmp_data[i * width], &temperature->data[(i + 1) * (width + 2) + 1],
                   temperature->ny * sizeof(double));
        MPI_Send(tmp_data, temperature->nx * temperature->ny,
                 MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
    }

    delete[] tmp_data;
    // free(tmp_data);

}

/* Read the initial temperature distribution from a file and
 * initialize the temperature fields temperature1 and
 * temperature2 to the same initial state. */
void read_field(field *temperature1, field *temperature2, char *filename,
                parallel_data *parallel)
{
    FILE *fp;
    int nx, ny, ind;
    double *full_data;
    double *inner_data;

    int nx_local, ny_local, count;

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
    temperature1->data = new double[(temperature1->nx + 2) * (temperature1->ny + 2)];
    temperature2->data = new double[(temperature1->nx + 2) * (temperature1->ny + 2)];

    inner_data = new double[temperature1->nx * temperature1->ny];

    if (parallel->rank == 0) {
        /* Full array */
        full_data = new double [nx * ny];

        /* Read the actual data */
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
	        ind = i * ny + j;
                count = fscanf(fp, "%lf", &full_data[ind]);
            }
        }
    } else {
        /* Dummy array for full data. Some MPI implementations
         * require that this array is actually allocated... */
        full_data = new double[1];
    }

    nx_local = temperature1->nx;
    ny_local = temperature1->ny;

    MPI_Scatter(full_data, nx_local * ny, MPI_DOUBLE, inner_data,
                nx_local * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Copy to the array containing also boundaries */
    for (int i = 0; i < nx_local; i++)
      memcpy(&temperature1->data[(i + 1) * (ny_local + 2) + 1], &inner_data[i * ny_local],
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
    delete[] inner_data; 
    // free(full_data);
    // free(inner_data);
    fclose(fp);
}
