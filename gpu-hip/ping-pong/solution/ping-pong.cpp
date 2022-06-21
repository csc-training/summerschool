#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

/* HIP error handling macro */
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}


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
    HIP_ERRCHK(hipGetDeviceCount(devCount));
}


/* Ping-pong test for CPU-to-CPU communication */
void CPUtoCPU(int rank, double *data, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();

    if (rank == 0) {
        MPI_Send(data, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        MPI_Recv(data, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(data, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Add one*/
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

    // Implement a transfer here that uses manual memcopies from device to host
    // (and back to device). Host pointers are passed for the MPI.
    // Remember to add one as in CPU code (using the existing GPU kernel).
    if (rank == 0) { //Sender process
        HIP_ERRCHK( hipMemcpy(hA, dA, sizeof(double)*N,
                               hipMemcpyDeviceToHost) );
        /* Send data to rank 1 for addition */
        MPI_Send(hA, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        /* Receive the added data back */
        MPI_Recv(hA, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        HIP_ERRCHK( hipMemcpy(dA, hA, sizeof(double)*N,
                               hipMemcpyHostToDevice) );
    } else { // Adder process
       MPI_Recv(hA, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       HIP_ERRCHK( hipMemcpy(dA, hA, sizeof(double)*N,
                              hipMemcpyHostToDevice) );
       int blocksize = 128;
       int gridsize = (N + blocksize - 1) / blocksize;
       add_kernel<<<blocksize, gridsize>>> (dA, N);
       HIP_ERRCHK( hipMemcpy(hA, dA, sizeof(double)*N,
                              hipMemcpyDeviceToHost) );
       MPI_Send(hA, N, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
    }

    stop = MPI_Wtime();
    timer = stop - start;
}


/* Ping-pong test for direct GPU-to-GPU communication using HIP-aware MPI */
void GPUtoGPUdirect(int rank, double *dA, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();
    // Implement a transfer here that uses HIP-aware MPI to transfer the data
    // directly by passing a device pointer to MPI.
    // Remember to add one as in CPU code (using the existing GPU kernel).

    if (rank == 0) { //Sender process
        /* Send data to rank 1 for addition */
        MPI_Send(dA, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        /* Receive the added data back */
        MPI_Recv(dA, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else { // Adder process
        int blocksize = 128;
        int gridsize = (N + blocksize - 1) / blocksize;

        MPI_Recv(dA, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        add_kernel<<<blocksize, gridsize>>> (dA, N);
        MPI_Send(dA, N, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
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

    /* Due to the test, we need exactly two processes with one GPU for
       each */
    if (nprocs != 2) {
        printf("Need exactly two processes!\n");
        exit(EXIT_FAILURE);
    }
    if (devcount == 0) {
        printf("Could now find any HIP devices.\n");
        exit(EXIT_FAILURE);
    }
    if (nodenprocs > devcount) {
        printf("Not enough GPUs for all processes in the node.\n");
        exit(EXIT_FAILURE);
    }
    else{
        printf("MPI rank %d: Found %d GPU devices, using GPUs 0 and 1!\n\n",
               rank, devcount);
    }

    // Select the device according to the node rank
    HIP_ERRCHK( hipSetDevice(noderank % devcount) );

    // Allocate enough pinned host and device memory for hA and dA
    // to store N doubles
    HIP_ERRCHK( hipHostMalloc((void **)&hA, sizeof(double) * N) );
    HIP_ERRCHK( hipMalloc((void **)&dA, sizeof(double) * N) );

    // Initialize the vectors
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;
    HIP_ERRCHK( hipMemcpy(dA, hA, sizeof(double)*N, hipMemcpyHostToDevice) );

    // CPU-to-CPU test
    CPUtoCPU(rank, hA, N, CPUtime);
    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        printf("CPU-CPU: time %f, errorsum %f\n", CPUtime, errorsum);
    }

    // Re-initialize the vectors
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;
    HIP_ERRCHK( hipMemcpy(dA, hA, sizeof(double)*N, hipMemcpyHostToDevice) );

    // GPU-to-GPU test, direct communication with HIP-aware MPI
    GPUtoGPUdirect(rank, dA, N, GPUtime);
    HIP_ERRCHK( hipMemcpy(hA, dA, sizeof(double)*N, hipMemcpyDeviceToHost) );
    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        printf("GPU-GPU direct: time %f, errorsum %f\n", GPUtime, errorsum);
    }

    // Re-initialize the vectors
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;
    HIP_ERRCHK( hipMemcpy(dA, hA, sizeof(double)*N, hipMemcpyHostToDevice) );

    // GPU-to-GPU test, communication via host
    GPUtoGPUviaHost(rank, hA, dA, N, GPUtime);
    HIP_ERRCHK( hipMemcpy(hA, dA, sizeof(double)*N, hipMemcpyDeviceToHost) );
    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        printf("GPU-GPU via host: time %f, errorsum %f\n", GPUtime, errorsum);
    }

    // Deallocate memory
    HIP_ERRCHK( hipHostFree(hA) );
    HIP_ERRCHK( hipFree(dA) );

    MPI_Finalize();
    return 0;
}
