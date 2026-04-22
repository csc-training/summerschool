#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>
void getNodeInfo(int *nodeRank, int *nodeProcs, int *devCount)
{
    MPI_Comm intranodecomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &intranodecomm);
    MPI_Comm_rank(intranodecomm, nodeRank);
    MPI_Comm_size(intranodecomm, nodeProcs);
    MPI_Comm_free(&intranodecomm);

    *devCount = omp_get_num_devices();
}

void CPUtoCPU(int rank, double *data, int N, double &timer)
{
    double start = MPI_Wtime();

    if (rank == 0) {
        MPI_Send(data, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        MPI_Recv(data, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        MPI_Recv(data, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; ++i)
            data[i] += 1.0;
        MPI_Send(data, N, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
    }

    double stop = MPI_Wtime();
    timer = stop - start;
}

/* Ping-pong test for direct GPU-to-GPU communication using GPU-aware MPI */
void GPUtoGPUdirect(int rank, double *dA, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();

    // GPU-to-GPU ping-pong that communicates directly from GPU memory
    // using HIP-aware MPI.
    if (rank == 0) {
        // Send vector to rank 1
        #pragma omp target data use_device_ptr(dA)
        {
            MPI_Send(dA, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        }
        // Receive vector from rank 1
        
        #pragma omp target data use_device_ptr(dA)
        {
            MPI_Recv(dA, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else if (rank == 1) {
        
        // Receive vector from rank 0
        #pragma omp target data use_device_ptr(dA)
        {
            MPI_Recv(dA, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Increment values on the GPU
        #pragma omp target teams distribute parallel for
        {
            for (int i = 0; i < N; i++) {
                dA[i] ++;
            }
        }

        // Send vector to rank 0
        #pragma omp target data use_device_ptr(dA)
        {
            MPI_Send(dA, N, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
        }
    }

    stop = MPI_Wtime();
    timer = stop - start;

}

/* Ping-pong test for indirect GPU-to-GPU communication via the host */
void GPUtoGPUviaHost(int rank, double *hA, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();

    // GPU-to-GPU ping-pong that communicates via the host, but uses the GPU
    // to increment the vector elements.
    if (rank == 0) {
         // Copy vector to host and send it to rank 1
        #pragma omp target update from(hA[0:N])
        MPI_Send(hA, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        // Receive vector from rank 1 and copy it to the device
        MPI_Recv(hA, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        #pragma omp target update to(hA[0:N])
    } else if (rank == 1) {
        // Receive vector from rank 0 and copy it to the device
        MPI_Recv(hA, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        #pragma omp target update to(hA[0:N])

        // Increment values on the GPU
        #pragma omp target teams distribute parallel for
        {
            for (int i = 0; i < N; i++) {
                hA[i] ++;
            }
        }
        // Copy vector to host and send it to rank 0
        #pragma omp target update from(hA[0:N])
        MPI_Send(hA, N, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
    }

    stop = MPI_Wtime();
    timer = stop - start;

}

int main(int argc, char *argv[])
{
    
    int rank, nprocs, noderank, nodenprocs, devcount;
    int N = 256*1024*1024;
    double GPUtime, CPUtime;
    double *hA;
    
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    
    getNodeInfo(&noderank, &nodenprocs, &devcount);

    if (nprocs < 2) {
        if (rank == 0) printf("Need at least 2 MPI ranks.\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    else if (devcount == 0) {
        if (rank == 0) printf("No OpenMP devices found.\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    } else {
        printf("MPI rank %d: Found %d GPU devices, using GPU %d\n",
               rank, devcount, noderank % devcount);
    }

    omp_set_default_device(noderank % devcount);
    int device = omp_get_default_device();

    // Allocate host and device data pointer
    hA = (double*)malloc(sizeof(double)*N);
    if (!hA) {
        printf("Rank %d: malloc failed\n", rank);
        MPI_Finalize();
        return 1;
    }

    // Allocate device data: use target enter data to allocate, no init copy yet
    #pragma omp target enter data map(alloc: hA[0:N])

    // Initialize host data
    for (int i = 0; i < N; ++i)
        hA[i] = 1.0;

    // Copy host to device
    #pragma omp target update to(hA[0:N])

    CPUtime = 0.0; GPUtime = 0.0;

    // CPU to CPU test
    CPUtoCPU(rank, hA, N, CPUtime);
    if (rank == 0) {
        double err = 0;
        for (int i = 0; i < N; i++) err += hA[i] - 2.0;
        printf("CPU-CPU: time %e, errorsum %f\n", CPUtime, err);
    }


    // Initialize the vectors
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;


    // Dummy transfer to remove the overhead of the first communication
    GPUtoGPUdirect(rank, hA, N, GPUtime);
    // Copy host to device
    #pragma omp target update to(hA[0:N])
    GPUtoGPUdirect(rank, hA, N, GPUtime);
    #pragma omp target update from(hA[0:N])
    if (rank == 0) {
        double err = 0;
        for (int i = 0; i < N; i++) err += hA[i] - 2.0;
        printf("GPU-GPU direct: time %e, errorsum %f\n", GPUtime, err);
    }

    // Dummy transfer to remove the overhead of the first communication
    GPUtoGPUviaHost(rank, hA, N, GPUtime);

    
    
    // Re-initialize the vectors
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;
    // Copy host to device
    #pragma omp target update to(hA[0:N])

    // GPU-to-GPU test, communication via host
    GPUtoGPUviaHost(rank, hA, N, GPUtime);
    #pragma omp target update from(hA[0:N])
    if (rank == 0) {
        double err = 0;
        for (int i = 0; i < N; i++) err += hA[i] - 2.0;
        printf("GPU-GPU via host: time %e, errorsum %f\n", GPUtime, err);
    }

    // Cleanup device memory
    #pragma omp target exit data map(delete: hA[0:N])
    free(hA);

    MPI_Finalize();
    return 0;
}
