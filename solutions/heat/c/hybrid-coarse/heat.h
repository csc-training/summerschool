#ifndef __HEAT_SERIAL_H__
#define __HEAT_SERIAL_H__

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
    int size;                   /* Number of MPI tasks */
    int rank;
    int nup, ndown;      /* Ranks of neighbouring MPI tasks */
} parallel_data;


/* We use here fixed grid spacing */
#define DX 0.01
#define DY 0.01


/* Function prototypes */
double **malloc_2d(int nx, int ny);

void free_2d(double **array);

void set_field_dimensions(field *temperature, int nx, int ny,
                          parallel_data *parallel);

void parallel_setup(parallel_data *parallel, int nx, int ny);

void parallel_set_dimensions(parallel_data *parallel, int nx, int ny);

void initialize(int argc, char *argv[], field *temperature1,
                field *temperature2, int *nsteps, parallel_data *parallel);

void generate_field(field *temperature, parallel_data *parallel);

void exchange(field *temperature, parallel_data *parallel, int thread_id);

void evolve(field *curr, field *prev, double a, double dt);

void write_field(field *temperature, int iter, parallel_data *parallel);

void read_field(field *temperature1, field *temperature2,
                char *filename, parallel_data *parallel);

void copy_field(field *temperature1, field *temperature2);

void swap_fields(field *temperature1, field *temperature2);

void allocate_field(field *temperature);

void finalize(field *temperature1, field *temperature2);

#endif  /* __HEAT_SERIAL_H__ */

