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

    // TODO: Fix the call to png writing. Note that since we now have a
    // dynamically allocated 2D array, we need to pass on a pointer to the
    // first element (=row) of the array instead of the array itself.
    error_code = save_png(..., ..., ..., "dynamic_array.png", 'c');
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

    // TODO: Allocate memory for a 2D array (nx,ny). Remember to allocate
    // space also for a ghost layer around the real data.

    // Initialize field and print out the result
    init_field(&temperature, nx, ny);
    print_field(&temperature);

    // Free memory allocation
    // TODO: Free memory allocations

    return 0;
}
