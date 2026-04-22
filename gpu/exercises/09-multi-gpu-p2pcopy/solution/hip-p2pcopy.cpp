#include <chrono>
#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>


void copyP2P(int p2p, int gpu0, int gpu1, int* dA_0, int* dA_1, int N) {

    // Enable peer access for GPUs?
    if (p2p)
    {
        hipSetDevice(gpu0);
        hipDeviceEnablePeerAccess(gpu1, 0);
        hipSetDevice(gpu1);
        hipDeviceEnablePeerAccess(gpu0, 0);
    }

    // Do a dummy copy without timing to remove the impact of the first one
    hipMemcpy(dA_0, dA_1, sizeof(int)*N, hipMemcpyDefault);
    hipDeviceSynchronize();

    // Do a series of timed P2P memory copies
    int M = 10;
    auto tStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        hipMemcpy(dA_0, dA_1, sizeof(int)*N, hipMemcpyDefault);
    }
    hipDeviceSynchronize();
    auto tStop = std::chrono::high_resolution_clock::now();

    // Calcute time and bandwith
    double time_s = std::chrono::duration_cast<std::chrono::nanoseconds>(tStop - tStart).count() / 1e9;
    double bandwidth = (double) N * sizeof(int) / (1024*1024*1024) / (time_s / M);

    // Disable peer access for GPUs?
    if (p2p) {
        hipSetDevice(gpu0);
        hipDeviceDisablePeerAccess(gpu1);
        hipSetDevice(gpu1);
        hipDeviceDisablePeerAccess(gpu0);
        printf("P2P enabled - Bandwith: %.3f (GB/s), Time: %.3f s\n",
                bandwidth, time_s);
    } else {
        printf("P2P disabled - Bandwith: %.3f (GB/s), Time: %.3f s\n",
                bandwidth, time_s);
    }
}


int main(int argc, char *argv[])
{
    // Check that we have at least two GPUs
    int devcount;
    hipGetDeviceCount(&devcount);
    if(devcount < 2) {
        printf("Need at least two GPUs!\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Found %d GPU devices\n", devcount);
    }

    // Allocate memory for both GPUs
    int N = pow(2, 28);
    int gpu0 = 0, gpu1 = 1;
    int *dA_0, *dA_1;
    hipSetDevice(gpu0);
    hipMalloc((void**) &dA_0, N * sizeof(int));
    hipSetDevice(gpu1);
    hipMalloc((void**) &dA_1, N * sizeof(int));

    // Check peer accessibility between GPUs 0 and 1
    int peerAccess01;
    int peerAccess10;
    hipDeviceCanAccessPeer(&peerAccess01, gpu0, gpu1);
    hipDeviceCanAccessPeer(&peerAccess10, gpu1, gpu0);
    printf("hipDeviceCanAccessPeer: %d (GPU %d to GPU %d)\n",
            peerAccess01, gpu0, gpu1);
    printf("hipDeviceCanAccessPeer: %d (GPU %d to GPU %d)\n",
            peerAccess10, gpu1, gpu0);

    // Memcopy, P2P enabled
    if (peerAccess01 && peerAccess10)
        copyP2P(1, gpu0, gpu1, dA_0, dA_1, N);

    // Memcopy, P2P disabled
    copyP2P(0, gpu0, gpu1, dA_0, dA_1, N);

    // Deallocate device memory
    hipFree(dA_0);
    hipFree(dA_1);
}
