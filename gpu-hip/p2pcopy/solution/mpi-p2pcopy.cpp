#include <chrono>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include <hip/hip_runtime.h>

inline void mpiMemcpy(int rank, int* dA, int N) {
    if (rank == 0) {
        MPI_Send(dA, N, MPI_INT, 1, 123, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(dA, N, MPI_INT, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void copyP2P(int rank, int* dA, int N) {
    // Do a dummy copy without timing to remove the impact of the first one
    mpiMemcpy(rank, dA, N);
    hipDeviceSynchronize();

    // Do a series of timed P2P memory copies
    int M = 10;
    auto tStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        mpiMemcpy(rank, dA, N);
    }
    hipDeviceSynchronize();
    auto tStop = std::chrono::high_resolution_clock::now();

    // Calcute time and bandwith
    double time_s = std::chrono::duration_cast<std::chrono::nanoseconds>(tStop - tStart).count() / 1e9;
    double bandwidth = (double) N * sizeof(int) / (1024*1024*1024) / (time_s / M);

    printf("MPI - Bandwith: %.3f (GB/s), Time: %.3f s\n",
            bandwidth, time_s);
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int ntasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check that we have two ranks
    if (ntasks != 2) {
        if (rank == 0) {
            fprintf(stderr, "Run this program with 2 tasks.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Check that we have at least two GPUs
    int devcount;
    hipGetDeviceCount(&devcount);
    if(devcount < 2) {
        printf("Need at least two GPUs!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    } else {
        printf("Found %d GPU devices\n", devcount);
    }

    // Set device
    hipSetDevice(rank % devcount);

    // Allocate memory for both GPUs
    int N = pow(2, 28);
    int *dA;
    hipMalloc((void**) &dA, sizeof(int) * N);

    // Memcopy
    copyP2P(rank, dA, N);

    // Deallocate device memory
    hipFree(dA);

    MPI_Finalize();
}
