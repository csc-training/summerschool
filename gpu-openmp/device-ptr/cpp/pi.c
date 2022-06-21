#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <curand.h>


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
    curandGenerator_t g;
    int istat;
    int inside;
    float *x, *y, pi;

    pi = 0;

    x = (float *)malloc(n * sizeof(float));
    y = (float *)malloc(n * sizeof(float));

    // TODO start: allocate x and y in the device with OpenMP enter data

    // TODO end

    inside = 0;

    istat = curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);

    // TODO start: use device pointer for CUDA random generator calls

    istat = curandGenerateUniform(g, x, n);
    if (istat != CURAND_STATUS_SUCCESS) printf("Error in curandGenerate: %d\n", istat);
    istat = curandGenerateUniform(g, y, n);
    if (istat != CURAND_STATUS_SUCCESS) printf("Error in curandGenerate: %d\n", istat);

    // TODO end


    // TODO start: execute the loop in parallel in device

        for (int i = 0; i < n; i++) {
            if (x[i]*x[i] + y[i]*y[i] < 1.0) {
                inside++;
            }
        }

    // TODO end

    // TODO start: deallocate x and y in the device with OpenMP exit data

    // TODO end

    free(x);
    free(y);

    istat = curandDestroyGenerator(g);
    if (istat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "Error in curandDestroyGenerator: %d\n", istat);
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
