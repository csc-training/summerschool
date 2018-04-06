#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NX_MAX 640
#define NY_MAX 640

/* This routine is used to print out the array in pgm format */
void write_pgm(double data[][NY_MAX], char *filename, int nx, int ny)
{
    FILE *fp;
    int i, j;

    if ((fp = fopen(filename, "w")) == NULL) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    /* write metadata */
    fprintf(fp, "P2\n# time step=0\n%d %d\n", nx, ny);
    fprintf(fp, "%d\n", 100);
    /* write temperature field */
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            fprintf(fp, "%d ", (int) data[i][j]);
        }
        fprintf(fp, "\n");
    }
    /* close file */
    fclose(fp);
}


void read_input(double image_buffer[][NY_MAX], char *filename)
{
    FILE *fp;
    int nx, ny, i, j;

    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    // Read the header
#error Add fscanf to read the dimensions of the array

    // Read the actual data
#error Add the loop where you read in the array into image_buffer

    // Write out an pgm file
    write_pgm(image_buffer, "bottle.pgm", nx, ny);

    fclose(fp);
}


int main(int argc, char **argv)
{
    char input_file[] = "bottle.dat";
    double image_buffer[NX_MAX][NY_MAX];

    read_input(image_buffer, input_file);
    return 0;
}
