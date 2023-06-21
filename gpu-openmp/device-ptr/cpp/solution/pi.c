#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <hiprand/hiprand.h>


float cpu_pi(int n)
{
    int inside, i;
    float *x, *y;

    x = (float *)malloc(n * sizeof(float));
    y = (float *)malloc(n * sizeof(float));

    for (i = 0; i < n; i++) {
        x[i] = (float)rand() / (float)RAND_MAX;
        y[i] = (float)rand() / (float)RAND_MAX;
    }

    inside = 0;
    for (i = 0; i < n; i++) {
        if (x[i]*x[i] + y[i]*y[i] < 1.0) {
            inside++;
        }
    }

    free(x);
    free(y);

    return 4.0 * (float)inside / (float)n;
}


float gpu_pi(size_t n)
{
    hiprandGenerator_t g;
    int istat;
    int inside;
    float *x, *y, pi;

    pi = 0;

    x = (float *)malloc(n * sizeof(float));
    y = (float *)malloc(n * sizeof(float));

    #pragma omp target enter data map(alloc:x[0:n], y[0:n])

    inside = 0;

    istat = hiprandCreateGenerator(&g, HIPRAND_RNG_PSEUDO_DEFAULT);

    #pragma omp target data use_device_ptr(x, y)
    {
        istat = hiprandGenerateUniform(g, x, n);
        if (istat != HIPRAND_STATUS_SUCCESS) printf("Error in hiprandGenerate: %d\n", istat);
        istat = hiprandGenerateUniform(g, y, n);
        if (istat != HIPRAND_STATUS_SUCCESS) printf("Error in hiprandGenerate: %d\n", istat);
    }

    #pragma omp target teams distribute parallel for reduction(+:inside)
        for (int i = 0; i < n; i++) {
            if (x[i]*x[i] + y[i]*y[i] < 1.0) {
                inside++;
            }
        }

    #pragma omp target exit data map(delete:x[0:n], y[0:n])

    free(x);
    free(y);

    istat = hiprandDestroyGenerator(g);
    if (istat != HIPRAND_STATUS_SUCCESS) {
        fprintf(stderr, "Error in hiprandDestroyGenerator: %d\n", istat);
    }

    pi = 4.0 * (float)inside / (float)n;

    return pi;
}


int main(int argc, char *argv[])
{
    int nsamples;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s N\n  where N is the number samples\n", argv[0]);
        exit(EXIT_FAILURE);
    } else {
        nsamples = atoi(argv[1]);
    }

    printf("Pi equals to %9.6f\n", cpu_pi(nsamples));
    printf("Pi equals to %9.6f\n", gpu_pi(nsamples));

    return 0;
}
