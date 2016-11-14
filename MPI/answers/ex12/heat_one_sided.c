#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat_mpi.h"
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
void initialize_field_metadata(field *temperature, int nx, int ny,
                               parallel_data *parallel)
{
    int nx_local;

    nx_local = nx / parallel->size;

    temperature->dx = DX;
    temperature->dy = DY;
    temperature->nx = nx_local;
    temperature->ny = ny;
    temperature->nx_full = nx;
    temperature->ny_full = ny;
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
    MPI_Win tmp_win;
    tmp = temperature1->data;
    temperature1->data = temperature2->data;
    temperature2->data = tmp;
    /* Swap the RMA windows */
    tmp_win = temperature1->rma_window;
    temperature1->rma_window = temperature2->rma_window;
    temperature2->rma_window = tmp_win;
}

/* Allocate memory for a temperature field and initialise it to zero */
void allocate_field(field *temperature)
{
    // Allocate also ghost layers
    temperature->data =
        malloc_2d(temperature->nx + 2, temperature->ny + 2);

    // Initialize to zero
    memset(temperature->data[0], 0.0,
           (temperature->nx + 2) * (temperature->ny + 2) * sizeof(double));
}

void parallel_set_dimensions(parallel_data *parallel, int nx, int ny)
{
    int nx_local;

    nx_local = nx / parallel->size;
    if (nx_local * parallel->size != nx) {
        printf("Cannot divide grid evenly to processors\n");
        MPI_Abort(MPI_COMM_WORLD, -2);
    }
}

void parallel_initialize(parallel_data *parallel, int nx, int ny)
{
    MPI_Comm_size(MPI_COMM_WORLD, &parallel->size);
    MPI_Comm_rank(MPI_COMM_WORLD, &parallel->rank);

    parallel_set_dimensions(parallel, nx, ny);

    parallel->nup = parallel->rank - 1;
    parallel->ndown = parallel->rank + 1;

    if (parallel->nup < 0) {
        parallel->nup = MPI_PROC_NULL;
    }
    if (parallel->ndown > parallel->size - 1) {
        parallel->ndown = MPI_PROC_NULL;
    }

}

/* Initialize the temperature field.  Pattern is disc with a radius
 * of nx_full / 6 in the center of the grid.
 * Boundary conditions are (different) constant temperatures outside the grid */
void initialize(field *temperature1, field *temperature2,
                parallel_data *parallel)
{
    int i, j;
    double radius;
    int dx, dy;

    /* Allocate the temperature arrays, note that
     * we have to allocate also the ghost layers */
    temperature1->data =
        malloc_2d(temperature1->nx + 2, temperature1->ny + 2);
    temperature2->data =
        malloc_2d(temperature2->nx + 2, temperature2->ny + 2);

    // Create RMA window. In principle, only borders would be needed
    // but it is simpler to expose the whole array
    MPI_Win_create(&temperature1->data[0][0],
                   (temperature1->nx + 2) * (temperature1->ny + 2) * sizeof(double),
                   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
                   &temperature1->rma_window);
    MPI_Win_create(&temperature2->data[0][0],
                   (temperature2->nx + 2) * (temperature2->ny + 2) * sizeof(double),
                   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
                   &temperature2->rma_window);

    /* Radius of the source disc */
    radius = temperature1->nx_full / 6.0;
    for (i = 0; i < temperature1->nx + 2; i++) {
        for (j = 0; j < temperature1->ny + 2; j++) {
            /* Distance of point i, j from the origin */
            dx = i + parallel->rank * temperature1->nx -
                 temperature1->nx_full / 2 + 1;
            dy = j - temperature1->ny / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                temperature1->data[i][j] = 5.0;
            } else {
                temperature1->data[i][j] = 65.0;
            }
        }
    }

    /* Boundary conditions */
    for (i = 0; i < temperature1->nx + 2; i++) {
        temperature1->data[i][0] = 20.0;
        temperature1->data[i][temperature1->ny + 1] = 70.0;
    }

    if (parallel->rank == 0) {
        for (j = 0; j < temperature1->ny + 2; j++) {
            temperature1->data[0][j] = 85.0;
        }
    } else if (parallel->rank == parallel->size - 1) {
        for (j = 0; j < temperature1->ny + 2; j++) {
            temperature1->data[temperature1->nx + 1][j] = 5.0;
        }
    }

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

/* Exchange the boundary values */
void exchange(field *temperature, parallel_data *parallel)
{

    MPI_Win_fence(0, temperature->rma_window);
    /* Put upwards */
    MPI_Put(temperature->data[1], temperature->ny + 2, MPI_DOUBLE,
            parallel->nup, (temperature->ny + 2) * (temperature->nx + 1),
            temperature->ny + 2, MPI_DOUBLE, temperature->rma_window);
    /* Put downwards */
    MPI_Put(temperature->data[temperature->nx], temperature->ny + 2,
            MPI_DOUBLE, parallel->ndown, 0, temperature->ny + 2,
            MPI_DOUBLE, temperature->rma_window);
    MPI_Win_fence(0, temperature->rma_window);
}

/* Deallocate the 2D arrays of temperature fields */
void finalize(field *temperature1, field *temperature2,
              parallel_data *parallel)
{
    free_2d(temperature1->data);
    free_2d(temperature2->data);

    MPI_Win_free(&temperature1->rma_window);
    MPI_Win_free(&temperature2->rma_window);
}

/* Output routine that prints out a picture of the temperature
 * distribution. */
void output(field *temperature, int iter, parallel_data *parallel)
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
void read_input(field *temperature1, field *temperature2, char *filename,
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

    parallel_initialize(parallel, nx, ny);
    initialize_field_metadata(temperature1, nx, ny, parallel);
    initialize_field_metadata(temperature2, nx, ny, parallel);

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

