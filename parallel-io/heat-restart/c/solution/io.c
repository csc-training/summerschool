/* I/O related functions for heat equation solver */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <hdf5.h>

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

/* Write a restart checkpoint that contains the current
 * iteration number and temperature field. */
void write_restart(field *temperature, parallel_data *parallel, int iter)
{
    herr_t status;
    hid_t plist_id, dset_id, filespace, memspace, attrspace, file_id, attr_id;
    hsize_t size_full[2]   = {temperature->nx_full, temperature->ny_full};
    hsize_t start_full[2]  = {parallel->rank*temperature->nx, 0};
    hsize_t size_block[2]  = {temperature->nx, temperature->ny};
    hsize_t ones[2] = {1, 1};
    hsize_t size_block_ghost[2] = {temperature->nx+2, temperature->ny+2};

    /* Create the file. */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    file_id = H5Fcreate(CHECKPOINT, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

    H5Pclose(plist_id);

    /* Define the dataspace for the dataset. */

    filespace = H5Screate_simple(2, size_full, NULL);

    /* Create the dataset */
    dset_id = H5Dcreate(file_id, "Temperature", H5T_NATIVE_DOUBLE, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Select a hyperslab of the file dataspace */

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start_full, NULL, ones,
                        size_block);

    /* Define the dataspace and hyperslab that containst the data
       to be written to disk. */

    memspace = H5Screate_simple(2, size_block_ghost, NULL);
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, ones, NULL, ones,
                        size_block);

    /* Now we can write our local data to the correct position in the
       dataset. Here we use collective write, but independent writes are
       also possible. */

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace,
                      plist_id, &(temperature->data[0][0]));


    /* Create dataspace for a scalar attribute. */
    attrspace = H5Screate(H5S_SCALAR);

    /* Create and write attribute holding the iteration number. */

    attr_id = H5Acreate(dset_id, "Iteration", H5T_NATIVE_INT, attrspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, H5T_NATIVE_INT, &iter);

    /* Close all open HDF5 handles. */
    H5Sclose(attrspace);
    H5Aclose(attr_id);
    H5Dclose(dset_id);
    H5Pclose(plist_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Fclose(file_id);
}

/* Read a restart checkpoint that contains the current
 * iteration number and temperature field. */
void read_restart(field *temperature, parallel_data *parallel, int *iter)
{
    herr_t status;
    hid_t plist_id, dset_id, filespace, memspace, file_id, attr_id;
    hsize_t size_full[2], start_full[2], size_block[2], size_block_ghost[2];
    hsize_t ones[2]={1, 1};

    /* Open the file for read-only access. */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    file_id = H5Fopen(CHECKPOINT, H5F_ACC_RDONLY, plist_id);

    H5Pclose(plist_id);

    /* Open the dataset in the file. */

    dset_id = H5Dopen(file_id, "Temperature", H5P_DEFAULT);

    /* Open the dataspace and read its dimensions .*/

    filespace = H5Dget_space(dset_id);
    H5Sget_simple_extent_dims(filespace, size_full, NULL);

    /* Set local dimensions and allocate memory for the data. */

    parallel_setup(parallel, size_full[0], size_full[1]);
    set_field_dimensions(temperature, size_full[0], size_full[1], parallel);
    allocate_field(temperature);

    /* Select the hyperslab to read. */

    start_full[0] = parallel->rank*temperature->nx;
    start_full[1] = 0;
    size_block[0] = temperature->nx;
    size_block[1] = temperature->ny;

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start_full, NULL, ones,
                        size_block);

    /* Create the dataspace for the dataset in memory */;

    size_block_ghost[0] = temperature->nx+2;
    size_block_ghost[1] = temperature->ny+2;

    memspace = H5Screate_simple(2, size_block_ghost, NULL);
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, ones, NULL, ones,
                        size_block);

    /* Read the data. */

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);


    status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace,
                      plist_id, &(temperature->data[0][0]));


    /* Read the iteration number. */

    attr_id = H5Aopen(dset_id, "Iteration", H5P_DEFAULT);
    H5Aread(attr_id, H5T_NATIVE_INT, iter);

    (*iter) += 1;

    /* Close the handles. */
    H5Aclose(attr_id);
    H5Dclose(dset_id);
    H5Pclose(plist_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Fclose(file_id);
}
