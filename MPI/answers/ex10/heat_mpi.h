#ifndef __HEAT_MPI_H__
#define __HEAT_MPI_H__

#include "mpi.h"

/* Datatype for temperature field */
typedef struct {
    /* nx and ny are the true dimensions of the field. The array data
     * contains also ghost layers, so it will have dimensions nx+2 x ny+2 */
    int nx;                     /* Local dimensions of the field */
    int ny;
    int nx_full;                /* Global dimensions of the field */
    int ny_full;                /* Global dimensions of the field */
    double dx;
    double dy;
    double **data;
} field;

/* Datatype for basic parallelization information */
typedef struct {
    int size;                       /* Number of MPI tasks */
    int rank;
	int nup, ndown, nleft, nright;  /* Ranks of neighbouring MPI tasks */
	MPI_Comm comm;                  /* MPI communicator */
	MPI_Datatype rowtype;           /* MPI Datatype for communication of 
									   rows */
	MPI_Datatype columntype;        /* MPI Datatype for communication of 
									   columns */
	MPI_Datatype subarraytype;      /* MPI Datatype for communication of 
									   inner region */
} parallel_data;

/* We use here fixed grid spacing */
#define DX 0.01
#define DY 0.01


/* Function prototypes */
double **malloc_2d(int nx, int ny);

void free_2d(double **array);

void initialize_field_metadata(field *temperature, int nx, int ny,
                               parallel_data *parallel);

void parallel_initialize(parallel_data *parallel, int nx, int ny);

void initialize(field *temperature1, field *temperature2,
                parallel_data *parallel);

void evolve(field *curr, field *prev, double a, double dt);

void exchange(field *temperature, parallel_data *parallel);

void output(field *temperature, int iter, parallel_data *parallel);

void read_input(field *temperature1, field *temperature2, char *filename,
                parallel_data *parallel);

void copy_field(field *temperature1, field *temperature2);

void swap_fields(field *temperature1, field *temperature2);

void finalize(field *temperature1, field *temperature2,
              parallel_data *parallel);

void write_restart(field *temperature, parallel_data *parallel, int iter);

void read_restart(field *temperature, parallel_data *parallel, int *iter);

#endif /* __HEAT_MPI_H__ */

