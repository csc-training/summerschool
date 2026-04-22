#include <chrono>
#include <cstdio>
#include <cmath>
#include <omp.h>


void copyP2P(int gpu0, int gpu1, int* dA_0, int* dA_1, int N) {

    // Do a dummy copy without timing to remove the impact of the first one
    omp_target_memcpy(dA_0, dA_1, N, 0, 0, gpu0, gpu1);

    // Do a series of timed P2P memory copies
    int M = 10;
    auto tStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        omp_target_memcpy(dA_0, dA_1, N, 0, 0, gpu0, gpu1);
    }
    auto tStop = std::chrono::high_resolution_clock::now();

    // Calcute time and bandwith
    double time_s = std::chrono::duration_cast<std::chrono::nanoseconds>(tStop - tStart).count() / 1e9;
    double bandwidth = (double) N * sizeof(int) / (1024*1024*1024) / (time_s / M);
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
        printf("Found %d GPU devices\n", devcount);
    }

    // Allocate memory for both GPUs
    int N = pow(2, 28);
    int gpu0 = 0, gpu1 = 1;
    int *dA_0, *dA_1;
    dA_0 = (int*)omp_target_alloc(N * sizeof(int), gpu0);
    dA_1 = (int*)omp_target_alloc(N * sizeof(int), gpu1);

    // Memcopy, P2P
    copyP2P(gpu0, gpu1, dA_0, dA_1, N);

    // Deallocate device memory
    omp_target_free(dA_0, gpu0);
    omp_target_free(dA_1, gpu1);
}
