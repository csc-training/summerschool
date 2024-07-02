#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <unistd.h>
#include <hip/hip_runtime.h>


/* HIP kernel to increment every element of a vector by one */
__global__ void add_kernel(double *in, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        in[tid]++;
}


/*
   This routine can be used to inspect the properties of a node
   Return arguments:

   nodeRank (int *)  -- My rank in the node communicator
   nodeProcs (int *) -- Total number of processes in this node
   devCount (int *)  -- Number of HIP devices available in the node
*/
void getNodeInfo(int *nodeRank, int *nodeProcs, int *devCount)
{
    MPI_Comm intranodecomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &intranodecomm);

    MPI_Comm_rank(intranodecomm, nodeRank);
    MPI_Comm_size(intranodecomm, nodeProcs);

    MPI_Comm_free(&intranodecomm);
    hipGetDeviceCount(devCount);
}


/* Ping-pong test for CPU-to-CPU communication */
void CPUtoCPU(int rank, double *data, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();

    if (rank == 0) {
        MPI_Send(data, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        MPI_Recv(data, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        MPI_Recv(data, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Increment by one on the CPU
        for (int i = 0; i < N; ++i)
            data[i] += 1.0;
        MPI_Send(data, N, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
    }

    stop = MPI_Wtime();
    timer = stop - start;
}


/* Ping-pong test for indirect GPU-to-GPU communication via the host */
void GPUtoGPUviaHost(int rank, double *hA, double *dA, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();
    int dest_rank, source_rank;
    int tag01 = 42;
    int tag10 = 43;

    // TODO: Implement a GPU-to-GPU ping-pong that communicates via the host,
    //       but uses the GPU to increment the vector elements. Copy data from
    //       device to host (and back) and use normal MPI communication on the
    //       host. Use the HIP kernel add_kernel() to increment values before
    //       sending them back to rank 0.
    if (rank == 0) {
        // TODO: Copy vector to host and send it to rank 1
        hipMemcpy(hA, dA, sizeof(double)*N, hipMemcpyDeviceToHost);
        dest_rank = 1;
        MPI_Send(hA, N, MPI_DOUBLE, dest_rank, tag01, MPI_COMM_WORLD);

        // TODO: Receive vector from rank 1 and copy it to the device
        source_rank = 1;
        MPI_Recv(hA, N, MPI_DOUBLE, source_rank, tag10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        hipMemcpy(dA, hA, sizeof(double)*N, hipMemcpyHostToDevice);

    } else if (rank == 1) {
        // TODO: Receive vector from rank 0 and copy it to the device
        source_rank = 0;
        MPI_Recv(hA, N, MPI_DOUBLE, source_rank, tag01, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        hipMemcpy(dA, hA, sizeof(double)*N, hipMemcpyHostToDevice);

        // TODO: Launch kernel to increment values on the GPU
        int n_threads = 128;  // Specifies the number of threads per block.
        int n_blocks = (N - 1 + n_threads) / n_threads;  // Calculates the number of blocks needed to cover all N elements.
                                                         // The -1 ensures that any remaining elements are covered by an 
                                                         // additional block.
        add_kernel<<<n_blocks, n_threads, 0, 0>>>(dA, N);

        // TODO: Copy vector to host and send it to rank 0
        dest_rank = 0;
        hipMemcpy(hA, dA, sizeof(double)*N, hipMemcpyDeviceToHost);  // Implicit hipDeviceSynchronize(). Will wait for all 
                                                                     // threads of the GPU to be idle before starting to copy.
        MPI_Send(hA, N, MPI_DOUBLE, dest_rank, tag10, MPI_COMM_WORLD);
    }

    stop = MPI_Wtime();
    timer = stop - start;
}


/* Ping-pong test for direct GPU-to-GPU communication using HIP-aware MPI */
void GPUtoGPUdirect(int rank, double *dA, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();
    int dest_rank, source_rank;
    int tag01 = 42;
    int tag10 = 43;

    // TODO: Implement a GPU-to-GPU ping-pong that communicates directly
    //       from GPU memory using HIP-aware MPI.
    if (rank == 0) {
        // TODO: Send vector to rank 1
        dest_rank = 1;
        MPI_Send(dA, N, MPI_DOUBLE, dest_rank, tag01, MPI_COMM_WORLD);

        // TODO: Receive vector from rank 1
        source_rank = 1;
        MPI_Recv(dA, N, MPI_DOUBLE, source_rank, tag10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else if (rank == 1) {
        // TODO: Receive vector from rank 0
        source_rank = 0;
        MPI_Recv(dA, N, MPI_DOUBLE, source_rank, tag01, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // TODO: Launch kernel to increment values on the GPU
        int n_threads = 128;  // Specifies the number of threads per block.
        int n_blocks = (N - 1 + n_threads) / n_threads;  // Calculates the number of blocks needed to cover all N elements.
        add_kernel<<<n_blocks, n_threads, 0, 0>>>(dA, N);  // Asynchronous by default.
        hipStreamSynchronize(0);  // Wait for all commands in stream 0 to complete. Without this, MPI_Send 
                                  // (executed by host) may start sending dA before add_kernel has finished.
        
        // TODO: Send vector to rank 0
        dest_rank = 0;
        MPI_Send(dA, N, MPI_DOUBLE, dest_rank, tag10, MPI_COMM_WORLD);
    }

    stop = MPI_Wtime();
    timer = stop - start;
}


int main(int argc, char *argv[])
{
    int rank, nprocs, noderank, nodenprocs, devcount;
    int N = 100;
    double GPUtime, CPUtime;
    double *dA, *hA;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    getNodeInfo(&noderank, &nodenprocs, &devcount);

    // Check that we have enough MPI tasks and GPUs
    if (nprocs < 2) {
        printf("Not enough MPI tasks! Need at least 2.\n");
        exit(EXIT_FAILURE);
    } else if (devcount == 0) {
        printf("Could not find any GPU devices.\n");
        exit(EXIT_FAILURE);
    } else {
        printf("MPI rank %d: Found %d GPU devices, using GPU %d\n",
               rank, devcount, noderank % devcount);
    }

    // Select the device according to the node rank
    hipSetDevice(noderank % devcount);

    // Allocate enough pinned host and device memory for hA and dA
    // to store N doubles
    hipHostMalloc((void **) &hA, sizeof(double) * N);
    hipMalloc((void **) &dA, sizeof(double) * N);

    // Dummy transfer to remove the overhead of the first communication
    CPUtoCPU(rank, hA, N, CPUtime);
    
    // Initialize the vectors
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;
    hipMemcpy(dA, hA, sizeof(double) * N, hipMemcpyHostToDevice);

    // CPU-to-CPU test
    CPUtoCPU(rank, hA, N, CPUtime);
    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        printf("CPU-CPU: time %e, errorsum %f\n", CPUtime, errorsum);
    }

    // Dummy transfer to remove the overhead of the first communication
    GPUtoGPUdirect(rank, dA, N, GPUtime);

    // Re-initialize the vectors

    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;
    hipMemcpy(dA, hA, sizeof(double) * N, hipMemcpyHostToDevice);

    // GPU-to-GPU test, direct communication with HIP-aware MPI
    GPUtoGPUdirect(rank, dA, N, GPUtime);
    if (rank == 0) {
        hipMemcpy(hA, dA, sizeof(double) * N, hipMemcpyDeviceToHost);
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        printf("GPU-GPU direct: time %e, errorsum %f\n", GPUtime, errorsum);
    }

    // Dummy transfer to remove the overhead of the first communication
    GPUtoGPUviaHost(rank, hA, dA, N, GPUtime);
    
    // Re-initialize the vectors
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;
    hipMemcpy(dA, hA, sizeof(double) * N, hipMemcpyHostToDevice);

    // GPU-to-GPU test, communication via host
    GPUtoGPUviaHost(rank, hA, dA, N, GPUtime);
    hipMemcpy(hA, dA, sizeof(double) * N, hipMemcpyDeviceToHost);  // Only need to be executed by rank 0, which does the checking. However, it does not matter in this case.
    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        printf("GPU-GPU via host: time %e, errorsum %f\n", GPUtime, errorsum);
    }

    // Deallocate memory
    hipHostFree(hA);
    hipFree(dA);

    MPI_Finalize();
    return 0;
}
