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

/* Very simple addition kernel */
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
   devCount (int *) -- Number of HIP devices available in the node
*/
void getNodeInfo(int *nodeRank, int *nodeProcs, int *devCount)
{
    MPI_Comm intranodecomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,  MPI_INFO_NULL, &intranodecomm);

    MPI_Comm_rank(intranodecomm, nodeRank);
    MPI_Comm_size(intranodecomm, nodeProcs);

    MPI_Comm_free(&intranodecomm);
    HIP_ERRCHK(hipGetDeviceCount(devCount));
}

/* Test routine for CPU-to-CPU copy */
void CPUtoCPUtest(int rank, double *data, int N, double &timer)
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

/* Test routine for GPU-CPU-to-CPU-GPU copy */
void GPUtoGPUtestManual(int rank, double *hA, double *dA, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();
    
    #error Implement a transfer here that uses manual memcopies from device to host 
    #error (and back to device). Host pointers are passed for the MPI. 
    #error Remember to add one as in CPU code (using the existing GPU kernel).

    stop = MPI_Wtime();
    timer = stop - start;
}

/* Test routine for GPU-CPU-to-CPU-GPU copy */
void GPUtoGPUtestHipAware(int rank, double *dA, int N, double &timer)
{
    double start, stop;
    start = MPI_Wtime();
    #error Implement a transfer here that uses HIP-aware MPI to transfer the data
    #error directly by passing a device pointer to MPI. 
    #error Remember to add one as in CPU code (using the existing GPU kernel).

    stop = MPI_Wtime();
    timer = stop - start;
}

/* Simple ping-pong main program */
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
        printf("MPI rank %d: Found %d GPU devices, using GPUs 0 and 1!\n\n", rank, devcount);
    }

    #error Select the device according to the node rank

    #error Allocate pinned host and device memory for hA and dA (sizeof(double) * N)

    /* Re-initialize and copy the data to the device memory to prepare for
     * MPI test */
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;    
    
    /* CPU-to-CPU test */
    CPUtoCPUtest(rank, hA, N, CPUtime);
    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;        
        printf("CPU-CPU time %f, errorsum %f\n", CPUtime, errorsum);
    }

    /* Re-initialize and copy the data to the device memory */
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;    
    HIP_ERRCHK( hipMemcpy(dA, hA, sizeof(double)*N, hipMemcpyHostToDevice) );
    
    /* GPU-to-GPU test, Hip-aware */
    GPUtoGPUtestHipAware(rank, dA, N, GPUtime);

    /*Check results, copy device array back to Host*/
    HIP_ERRCHK( hipMemcpy(hA, dA, sizeof(double)*N, hipMemcpyDeviceToHost) );
    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;        
        printf("GPU-GPU hip-aware time %f, errorsum %f\n", GPUtime, errorsum);
    }

    /* Re-initialize and copy the data to the device memory to prepare for
     * MPI test */
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;    
    HIP_ERRCHK( hipMemcpy(dA, hA, sizeof(double)*N, hipMemcpyHostToDevice) );

    /* GPU-to-GPU test, Manual option*/
    GPUtoGPUtestManual(rank, hA, dA, N, GPUtime);

    /* Check results, copy device array back to Host */
    HIP_ERRCHK( hipMemcpy(hA, dA, sizeof(double)*N, hipMemcpyDeviceToHost) );
    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        
        printf("GPU-GPU manual time %f, errorsum %f\n", GPUtime, errorsum);
    }

    #error Free pinned host and device memory for hA and dA

    MPI_Finalize();
    return 0;
}
