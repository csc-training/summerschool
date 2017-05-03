#include <stdio.h>
#include <stdlib.h>

#define NX 258
#define NY 258

int main(void)
{
    int i, j;
    double array[NX][NY];

    // Initialize first the zeros
    for (i = 1; i < NX - 1; i++) {
        for (j = 1; j < NY - 1; j++) {
            array[i][j] = 0.0;
        }
    }

    // Initial conditions for left and right
    for (i = 0; i < NX; i++) {
        array[i][0] = 20.0;
        array[i][NY - 1] = 70.0;
    }
    // and top and bottom boundaries
    for (j = 0; j < NY; j++) {
        array[0][j] = 85.0;
        array[NX - 1][j] = 5.0;
    }

    return 0;
}
