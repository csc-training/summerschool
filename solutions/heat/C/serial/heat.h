#ifndef __HEAT_H__
#define __HEAT_H__

/* Datatype for temperature field */
typedef struct {
    /* nx and ny are the true dimensions of the field. The array data
     * contains also ghost layers, so it will have dimensions nx+2 x ny+2 */
    int nx;
    int ny;
    double dx;
    double dy;
    double **data;
} field;

/* We use here fixed grid spacing */
#define DX 0.01
#define DY 0.01


/* Function prototypes */
double **malloc_2d(int nx, int ny);

void free_2d(double **array);

void set_field_dimensions(field *temperature, int nx, int ny);

void initialize(int argc, char *argv[], field *temperature1,
                field *temperature2, int *nsteps);

void generate_field(field *temperature);

void evolve(field *curr, field *prev, double a, double dt);

void write_field(field *temperature, int iter);

void read_field(field *temperature1, field *temperature2,
                char *filename);

void copy_field(field *temperature1, field *temperature2);

void swap_fields(field *temperature1, field *temperature2);

void finalize(field *temperature1, field *temperature2);

#endif  /* __HEAT_H__ */

