#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

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

    // Check that we have two HIP devices available
    hipGetDeviceCount(&devicecount);
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
    hipSetDevice(0);
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Allocate host memory
    hipHostMalloc((void**)&hA, sizeof(double) * N);
    hipHostMalloc((void**)&hB, sizeof(double) * N);
    hipHostMalloc((void**)&hC, sizeof(double) * N);

    // Initialize host memory
    for(int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    // Decomposition of data
    dec[0].len   = N / 2;
    dec[0].start = 0;
    dec[1].len   = N - N / 2;
    dec[1].start = dec[0].len;

    // Allocate memory for the devices and per device streams
    for (int i = 0; i < 2; ++i) {
        hipSetDevice(i);
        hipMalloc((void**)&dA[i], sizeof(double) * dec[i].len);
        hipMalloc((void**)&dB[i], sizeof(double) * dec[i].len);
        hipMalloc((void**)&dC[i], sizeof(double) * dec[i].len);
        hipStreamCreate(&strm[i]);
    }

    // Start timing
    hipSetDevice(0);
    hipEventRecord(start);

    /* Copy each decomposed part of the vectors from host to device memory
       and execute a kernel for each part.
       Note: one needs to use streams and asynchronous calls! Without this
       the execution is serialized because the memory copies block the
       execution of the host process. */
    for (int i = 0; i < 2; ++i) {
        // Set active device
        hipSetDevice(i);

        // Copy data from host to device (hA -> dA, hB -> dB)
        hipMemcpyAsync(dA[i], (void *)&(hA[dec[i].start]),
                                    sizeof(double) * dec[i].len,
                                    hipMemcpyHostToDevice, strm[i]);

        hipMemcpyAsync(dB[i], (void *)&(hB[dec[i].start]),
                                    sizeof(double) * dec[i].len,
                                    hipMemcpyHostToDevice, strm[i]);

        // Launch kernel to calculate dC = dA + dB
        dim3 grid, threads;
        grid.x = (dec[i].len + ThreadsInBlock - 1) / ThreadsInBlock;
        threads.x = ThreadsInBlock;

        vector_add<<<grid, threads, 0, strm[i]>>>(dC[i], dA[i], dB[i],
                                                  dec[i].len);

        // Copy data from device to host (dC -> hC)
        hipMemcpyAsync((void *)&(hC[dec[i].start]), dC[i],
                                    sizeof(double) * dec[i].len,
                                    hipMemcpyDeviceToHost, strm[i]);
    }

    // Synchronize and destroy the streams
    for (int i = 0; i < 2; ++i) {
        hipSetDevice(i);
        hipStreamSynchronize(strm[i]);
        hipStreamDestroy(strm[i]);
    }

    // Stop timing
    hipSetDevice(0);
    hipEventRecord(stop);

    // Free device memory
    for (int i = 0; i < 2; ++i) {
        hipSetDevice(i);
        hipFree((void*)dA[i]);
        hipFree((void*)dB[i]);
        hipFree((void*)dC[i]);
    }

    // Check results
    int errorsum = 0;
    for (int i = 0; i < N; i++) {
        errorsum += hC[i] - 3.0;
    }
    printf("Error sum = %i\n", errorsum);

    // Calculate the elapsed time
    float gputime;
    hipSetDevice(0);
    hipEventElapsedTime(&gputime, start, stop);
    printf("Time elapsed: %f\n", gputime / 1000.);

    // Deallocate host memory
    hipHostFree((void*)hA);
    hipHostFree((void*)hB);
    hipHostFree((void*)hC);

    return 0;
}
