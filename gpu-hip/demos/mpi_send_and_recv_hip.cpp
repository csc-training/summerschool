#include <cstdio>
#include <mpi.h>
#include <hip/hip_runtime.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int count, device;
    hipGetDeviceCount(&count);
    hipSetDevice(rank % count);
    hipGetDevice(&device);

    printf("Hello from MPI rank %d/%d with GPU %d/%d\n", rank, size, device, count);

    // Device data
    int N = 1024*1024;
    double *data;
    hipMalloc((void **) &data, sizeof(double) * N);

    if (rank == 0) {
        // Send with rank 0
        double h_data = 42.0;
        hipMemcpy(data, &h_data, sizeof(double), hipMemcpyHostToDevice);

        MPI_Send(data, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        printf("Rank %d sent %f\n", rank, h_data);

    } else if (rank == 1) {
        // Receive with rank 1
        MPI_Recv(data, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double h_data;
        hipMemcpy(&h_data, data, sizeof(double), hipMemcpyDeviceToHost);
        printf("Rank %d received %f\n", rank, h_data);
    }

    hipFree(data);

    MPI_Finalize();

    return 0;
}
