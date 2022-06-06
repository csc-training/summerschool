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

    // Check that we have two HIP devices available    
    HIP_ERRCHK( hipGetDeviceCount(&devicecount) );
    switch (devicecount) {
    case 0:
        printf("Could not find any HIP devices!\n");
        exit(EXIT_FAILURE);
    case 1:
        printf("Found one HIP device, this program requires two\n");
        exit(EXIT_FAILURE);
    default:
        printf("Found %d GPU devices, using GPUs 0 and 1!\n\n", devicecount);
    }

    // Create timing events
    HIP_ERRCHK( hipSetDevice(0) );
    HIP_ERRCHK( hipEventCreate(&start) );
    HIP_ERRCHK( hipEventCreate(&stop) );

    HIP_ERRCHK( hipHostMalloc((void**)&hA, sizeof(double) * N) );
    HIP_ERRCHK( hipHostMalloc((void**)&hB, sizeof(double) * N) );
    HIP_ERRCHK( hipHostMalloc((void**)&hC, sizeof(double) * N) );

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
        HIP_ERRCHK( hipSetDevice(i) );
        HIP_ERRCHK( hipMalloc((void**)&dA[i], sizeof(double) * dec[i].len) );
        HIP_ERRCHK( hipMalloc((void**)&dB[i], sizeof(double) * dec[i].len) );
        HIP_ERRCHK( hipMalloc((void**)&dC[i], sizeof(double) * dec[i].len) );
        HIP_ERRCHK( hipStreamCreate(&(strm[i])) );
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
        HIP_ERRCHK( hipSetDevice(i) );

        HIP_ERRCHK( hipMemcpyAsync(dA[i], (void *)&(hA[dec[i].start]), 
                                    sizeof(double) * dec[i].len, 
                                    hipMemcpyHostToDevice, strm[i]) );
        
        HIP_ERRCHK( hipMemcpyAsync(dB[i], (void *)&(hB[dec[i].start]),
                                    sizeof(double) * dec[i].len, 
                                    hipMemcpyHostToDevice, strm[i]) );
        
        dim3 grid, threads;
        grid.x = (dec[i].len + ThreadsInBlock - 1) / ThreadsInBlock;
        threads.x = ThreadsInBlock;
        
        vector_add<<<grid, threads, 0, strm[i]>>>(dC[i], dA[i], dB[i], 
                                                  dec[i].len);
        
        HIP_ERRCHK( hipMemcpyAsync((void *)&(hC[dec[i].start]), dC[i],
                                    sizeof(double) * dec[i].len, 
                                    hipMemcpyDeviceToHost, strm[i]) );
    }

    //// Add here the stream synchronization calls. After both
    // streams have finished, we know that we stop the timing.
    for (int i = 0; i < 2; ++i) {
        HIP_ERRCHK( hipSetDevice(i) );
        HIP_ERRCHK( hipStreamSynchronize(strm[i]) );
        HIP_ERRCHK( hipStreamDestroy(strm[i]) );
    }

    // Add here the timing event stop calls
    HIP_ERRCHK( hipSetDevice(0) );
    HIP_ERRCHK( hipEventRecord(stop) );

    /* Free device memories */
    for (int i = 0; i < 2; ++i) {
        HIP_ERRCHK( hipSetDevice(i) );
        HIP_ERRCHK( hipFree((void*)dA[i]) );
        HIP_ERRCHK( hipFree((void*)dB[i]) );
        HIP_ERRCHK( hipFree((void*)dC[i]) );
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
