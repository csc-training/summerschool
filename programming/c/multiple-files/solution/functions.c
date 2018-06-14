#include "functions.h"
#include "pngwriter.h"

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

    // Initial conditions for top and bottom
    for (i = 0; i < f->nx + 2; i++) {
        f->data[i][0] = 30.0;
        f->data[i][f->ny + 1] = -10.0;
    }
    // and left and right boundaries
    for (j = 0; j < f->ny + 2; j++) {
        f->data[0][j] = 15.0;
        f->data[f->nx + 1][j] = -25.0;
    }
}

// This function computes the Laplacian of the field
void laplacian(field *f)
{
    // Array where we store the result
    double laplacian[NX + 2][NX + 2];
    int i, j;

    // Evaluate the Laplacian
    // *INDENT-OFF*
    for (i = 1; i < f->nx + 1; i++) {
        for (j = 1; j < f->ny + 1; j++) {
            laplacian[i][j] =
                (f->data[i-1][ j ] - 2.0 * f->data[i][j] + f->data[i+1][ j ]) / (f->dx * f->dx) +
                (f->data[ i ][j-1] - 2.0 * f->data[i][j] + f->data[ i ][j+1]) / (f->dy * f->dy);
        }
    }
    // *INDENT-ON*

    // Copy the results back to the struct
    for (i = 1; i < f->nx + 1; i++)
        for (j = 1; j < f->ny + 1; j++) {
            f->data[i][j] = laplacian[i][j];
        }
}

// This function saves the field values to a png file
void print_field(field *f)
{
    int error_code;

    error_code =
        save_png((double *) f->data, f->nx + 2, f->ny + 2, "functions.png",
                 'c');
    if (error_code == 0) {
        printf("Wrote output file functions.png\n");
    } else {
        printf("Error while writing output file functions.png\n");
    }
}
