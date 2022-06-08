#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

/* HIP error handling macro */
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
	if (err != hipSuccess) {
		printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

/* Information of the decomposition */
struct Decomp {
    int len; // the lenght of the array for the current device
    int start; // the start index for the array on the current device
};

/* Kernel for vector summation */
__global__ void vector_add(double *C, const double *A, const double *B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not try to access past the allocated memory
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[])
{
    const int ThreadsInBlock = 128;
    double *dA[2], *dB[2], *dC[2];
    double *hA, *hB, *hC;
    int devicecount;
    int N = 100;
    hipEvent_t start, stop;
    hipStream_t strm[2];
    Decomp dec[2];

    #error Check that we have two HIP devices available     

    // Create timing events
    HIP_ERRCHK( hipSetDevice(0) );
    HIP_ERRCHK( hipEventCreate(&start) );
    HIP_ERRCHK( hipEventCreate(&stop) );

    #error Allocate pinned host memory for hA, hB, and hC (sizeof(double) * N)
    
    
    // Here we initialize the host memory values    
    for(int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    /* The decomposition */
    dec[0].len   = N / 2;
    dec[0].start = 0;
    dec[1].len   = N - N / 2;
    dec[1].start = dec[0].len;

    /* Allocate memory for the devices and per device streams */
    for (int i = 0; i < 2; ++i) {
        #error Allocate device memory for dA, dB, dC, (sizeof(double) * dec[i].len) and create streams for each device
    }

    /* Start timer */
    HIP_ERRCHK( hipSetDevice(0) );
    HIP_ERRCHK( hipEventRecord(start) );

    /* Copy the parts of the vectors on host to the devices and
       execute a kernel for each part. Note that we use asynchronous
       copies and streams. Without this the execution is serialized
       because the memory copies block the host process execution. */
    for (int i = 0; i < 2; ++i) {
        // Start by selecting the active device!
        #error Add here the memcpy-kernel-memcpy parts
    }

    //// Add here the stream synchronization calls. After both
    // streams have finished, we know that we stop the timing.
    for (int i = 0; i < 2; ++i) {
        #error Add here the synchronization calls and destroy streams
    }

    // Add here the timing event stop calls
    #error Add here timing calls

    /* Free device memories */
    for (int i = 0; i < 2; ++i) {
        #error Add here HIP deallocations
    }

    int errorsum = 0;

    for (int i = 0; i < N; i++) {
        errorsum += hC[i] - 3.0;
    }

    printf("Error sum = %i\n", errorsum);

    // Compute the elapsed time and release host memory
    float gputime;
    HIP_ERRCHK( hipSetDevice(0) );
    HIP_ERRCHK( hipEventElapsedTime(&gputime, start, stop) );
    printf("Time elapsed: %f\n", gputime / 1000.);

    HIP_ERRCHK( hipHostFree((void*)hA) );
    HIP_ERRCHK( hipHostFree((void*)hB) );
    HIP_ERRCHK( hipHostFree((void*)hC) );

    return 0;
}
