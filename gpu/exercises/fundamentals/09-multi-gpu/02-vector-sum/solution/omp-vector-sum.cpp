#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Decomposition metadata
struct Decomp {
    int len;
    int start;
};

int main()
{
    const int N = 100;
    const int ThreadsInBlock = 128;
    double *hA, *hB, *hC;
    Decomp dec[2];
    int devicecount = omp_get_num_devices();

    if (devicecount < 2) {
        printf("This program requires at least 2 OpenMP target devices. Found %d.\n", devicecount);
        exit(EXIT_FAILURE);
    } else {
        printf("Found %d OpenMP devices, using devices 0 and 1!\n\n", devicecount);
    }

    // Allocate and initialize host memory
    hA = (double *)malloc(N * sizeof(double));
    hB = (double *)malloc(N * sizeof(double));
    hC = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    // Decompose the data
    dec[0].len = N / 2;
    dec[0].start = 0;
    dec[1].len = N - N / 2;
    dec[1].start = dec[0].len;

    double start_time = omp_get_wtime();

    // Launch two asynchronous target regions, one per device
    for (int i = 0; i < 2; ++i) {
        int dev = i;
        int len = dec[i].len;
        int start = dec[i].start;

        #pragma omp target teams distribute parallel for device(dev) map(to: hA[start:len], hB[start:len]) map(from: hC[start:len]) thread_limit(ThreadsInBlock) nowait
        for (int j = 0; j < len; ++j) {
            hC[start + j] = hA[start + j] + hB[start + j];
        }
    }

    // Wait for both devices to finish
    #pragma omp taskwait

    double stop_time = omp_get_wtime();

    // Check results
    int errorsum = 0;
    for (int i = 0; i < N; ++i) {
        if (hC[i] != 3.0)
            errorsum += 1;
    }
    printf("Error count = %d\n", errorsum);
    printf("Time elapsed: %f seconds\n", stop_time - start_time);

    // Cleanup
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
