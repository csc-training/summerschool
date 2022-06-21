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


// Data structure for storing decomposition information
struct Decomp {
    int len;    // length of the array for the current device
    int start;  // start index for the array on the current device
};


/* HIP kernel for the addition of two vectors, i.e. C = A + B */
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

    // TODO: Check that we have two HIP devices available

    // Create timing events
    HIP_ERRCHK( hipSetDevice(0) );
    HIP_ERRCHK( hipEventCreate(&start) );
    HIP_ERRCHK( hipEventCreate(&stop) );

    // Allocate host memory
    // TODO: Allocate enough pinned host memory for hA, hB, and hC
    //       to store N doubles each

    // Initialize host memory
    for(int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    // Decomposition of data for each stream
    dec[0].len   = N / 2;
    dec[0].start = 0;
    dec[1].len   = N - N / 2;
    dec[1].start = dec[0].len;

    // Allocate memory for the devices and per device streams
    for (int i = 0; i < 2; ++i) {
        // TODO: Allocate enough device memory for dA, dB, dC
        //       to store dec[i].len doubles
        // TODO: Create streams for each device
    }

    // Start timing
    HIP_ERRCHK( hipSetDevice(0) );
    HIP_ERRCHK( hipEventRecord(start) );

    /* Copy each decomposed part of the vectors from host to device memory
       and execute a kernel for each part.
       Note: one needs to use streams and asynchronous calls! Without this
       the execution is serialized because the memory copies block the
       execution of the host process. */
    for (int i = 0; i < 2; ++i) {
        // TODO: Set active device
        // TODO: Copy data from host to device (hA -> dA, hB -> dB)
        // TODO: Launch kernel to calculate dC = dA + dB
        // TODO: Copy data from device to host (dC -> hC)
    }

    // Synchronize and destroy the streams
    for (int i = 0; i < 2; ++i) {
        // TODO: Add synchronization calls and destroy streams
    }

    // Stop timing
    // TODO: Add here the timing event stop calls

    // Free device memory
    for (int i = 0; i < 2; ++i) {
        // TODO: Deallocate device memory
    }

    // Check results
    int errorsum = 0;
    for (int i = 0; i < N; i++) {
        errorsum += hC[i] - 3.0;
    }
    printf("Error sum = %i\n", errorsum);

    // Calculate the elapsed time
    float gputime;
    HIP_ERRCHK( hipSetDevice(0) );
    HIP_ERRCHK( hipEventElapsedTime(&gputime, start, stop) );
    printf("Time elapsed: %f\n", gputime / 1000.);

    // Deallocate host memory
    HIP_ERRCHK( hipHostFree((void*)hA) );
    HIP_ERRCHK( hipHostFree((void*)hB) );
    HIP_ERRCHK( hipHostFree((void*)hC) );

    return 0;
}
