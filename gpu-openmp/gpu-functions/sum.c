#include <stdio.h>

#define NX 102400

// TODO add declaration for target usage
double my_sum(double a, double b);

// TODO end

int main(void)
{
    double vecA[NX], vecB[NX], vecC[NX];

    /* Initialization of the vectors */
    for (int i = 0; i < NX; i++) {
        vecA[i] = 1.0 / ((double) (NX - i));
        vecB[i] = vecA[i] * vecA[i];
    }

#pragma omp target teams distribute parallel for
    for (int i = 0; i < NX; i++) {
        vecC[i] = my_sum(vecA[i], vecB[i]);
    }

    double sum = 0.0;
    /* Compute the check value */
    for (int i = 0; i < NX; i++) {
        sum += vecC[i];
    }
    printf("Reduction sum: %18.16f\n", sum);

    return 0;
}
