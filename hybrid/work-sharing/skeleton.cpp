#include <cstdio>

#define NX 102400

int main(void)
{
    double vecA[NX], vecB[NX], vecC[NX];

    /* Initialization of the vectors */
    for (int i = 0; i < NX; i++) {
        vecA[i] = 1.0 / ((double)(NX - i));
        vecB[i] = vecA[i] * vecA[i];
    }

    /* TODO:
     *   Implement here a parallelized version of vector addition,
     *   vecC = vecA + vecB
     */

    int i = 0;
    #pragma omp parallel for shared(vecA, vecB, vecC) private(i)  // NX not included in the shared list because it is not a variable. Do not put reduction(+:vecC) here since each thread works on a separate chunk of the loop and accesses independent elements of vecC.
    for (i=0; i < NX; i++) {
        vecC[i] = vecA[i] + vecB[i];
    }

    double sum = 0.0;
    /* Compute the check value */
    #pragma omp parallel for shared(vecC) private(i) reduction(+:sum)
    for (int i = 0; i < NX; i++) {
        sum += vecC[i];
    }
    printf("Reduction sum: %18.16f\n", sum);

    return 0;
}
