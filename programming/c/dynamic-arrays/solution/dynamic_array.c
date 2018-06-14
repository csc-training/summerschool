#include <stdio.h>
#include <stdlib.h>
#include "dynamic_array.h"

// This routine initializes the values of field structure
void init_field(field *f, int nx, int ny)
{
    int i, j;

    f->nx = nx;
    f->ny = ny;
    f->dx = DX;
    f->dy = DY;
    f->dx2 = DX * DX;
    f->dy2 = DY * DY;

    // First zero out the inner part of the array
    for (i = 1; i < f->nx + 1; i++)
        for (j = 1; j < f->ny + 1; j++) {
            f->data[i][j] = 0.0;
        }

    // Initial conditions for left and right
    for (i = 0; i < f->nx + 2; i++) {
        f->data[i][0] = 20.0;
        f->data[i][f->ny + 1] = 70.0;
    }
    // and top and bottom boundaries
    for (j = 0; j < f->ny + 2; j++) {
        f->data[0][j] = 85.0;
        f->data[f->nx + 1][j] = 5.0;
    }
}

// This function saves the field values to a png file
void print_field(field *f)
{
    int error_code;

    error_code =
        save_png((double *) f->data[0], f->nx + 2, f->ny + 2,
                 "dynamic_array.png", 'c');
    if (error_code == 0) {
        printf("Wrote output file dynamic_array.png\n");
    } else {
        printf("Error while writing output file dynamic_array.png\n");
    }
}

int main(int argc, char *argv[])
{
    int nx, ny;
    field temperature;

    int i;

    // Two command line arguments required
    if (argc != 3) {
        printf("Please give two command line arguments "
	       "for x- an y-dimensions\n");
        return (-1);
    }
    // atoi function converts a string to integer
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);

    // Allocate the array
    temperature.data = (double **) malloc((nx + 2) * sizeof(double *));
    temperature.data[0] = (double *) malloc((nx + 2) * (ny + 2) * sizeof(double));

    for (i = 1; i < nx + 2; i++) {
        temperature.data[i] = temperature.data[0] + i * (ny + 2);
    }


    // Initialize field and print out the result
    init_field(&temperature, nx, ny);
    print_field(&temperature);

    // Free memory allocation
    free(temperature.data[0]);
    free(temperature.data);

    return 0;
}
