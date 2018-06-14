#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pngwriter.h"

#define NX 258
#define NY 258

int main(void)
{
    int i, j, error_code;
    double array[NX][NY];

    // Initialize the zeros first
    memset(array, 0, NX * NY * sizeof(double));

    // Initial conditions for top and bottom
    for (i = 0; i < NX; i++) {
        array[i][0] = 20.0;
        array[i][NY - 1] = 70.0;
    }
    // left and right boundaries
    for (j = 0; j < NY; j++) {
        array[0][j] = 85.0;
        array[NX - 1][j] = 5.0;
    }

    // Call the png writer routine
#error Add here the correct call for png writing

    if (error_code == 0) {
        printf("Wrote the output file ex5.png\n");
    } else {
        printf("Error while writing output file ex5.png\n");
    }

    return 0;
}
