
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>


void copyP2P(int gpu0, int gpu1, int* dA_0, int* dA_1, int size) {

    // Do a dummy copy without timing to remove the impact of the first one
    omp_target_memcpy(dA_0, dA_1, size, 0, 0, gpu0, gpu1);

    // Do a series of timed P2P memory copies
    int N = 10;
    clock_t tStart = clock();
    for (int i = 0; i < N; ++i) {
      omp_target_memcpy(dA_0, dA_1, size, 0, 0, gpu0, gpu1);
    }
    clock_t tStop = clock();

    // Calcute time and bandwith
    double time_s = (double) (tStop - tStart) / CLOCKS_PER_SEC;
    double bandwidth = (double) size * (double) N / (double) 1e9 / time_s;
    printf("P2P, Bandwith: %.3f (GB/s), Time: %.3f s\n", bandwidth, time_s);
}


int main(int argc, char *argv[])
{
    // Check that we have at least two GPUs
    int devcount = omp_get_num_devices();
    if(devcount < 2) {
        printf("Need at least two GPUs!\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Found %d GPU devices, using GPUs 0 and 1!\n", devcount);
    }

    // Allocate memory for both GPUs
    int size = pow(2, 28);
    int gpu0 = 0, gpu1 = 1;
    int *dA_0, *dA_1, *hA_0, *hA_1;
    dA_0 = (int*)omp_target_alloc(size, gpu0);
    dA_1 = (int*)omp_target_alloc(size, gpu1);

    // Memcopy, P2P
    copyP2P(gpu0, gpu1, dA_0, dA_1, size);

    // Deallocate device memory
    omp_target_free(dA_0, gpu0);
    omp_target_free(dA_1, gpu1);
}
