#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstring>

#include "../../../error_checking.hpp"



using counter_t = unsigned long long;

// ============================================================
// kernels
// ============================================================


//#1: basic serial
__global__ void histogram_single_thread(const int* __restrict__ bins, 
                              size_t n,
                              counter_t* __restrict__ hist,
                              int num_bins)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int b = 0; b < num_bins; ++b) hist[b] = 0;
        for (size_t i = 0; i < n; ++i) {
            int bin = bins[i];
            if (0 <= bin && bin < num_bins) {
                hist[bin] += 1;
            }
        }
    }
}


//#2 parallelized on threads, one block per bin
__global__ void histogram_one_block_per_bin(const int* __restrict__ bins, 
                                size_t n,
                                counter_t* __restrict__ hist, 
                                int num_bins)
{
    // WRITE THE KERNEL HERE. 
    // HINT:
    // every thread "owns" a cell in shared memory. we "split" the array across the threads of every block, and when the value is the one that should be in the bin (remember every block "owns" one bin) then they increment the thread local counter. When all the values in the array are evaluated, we perform a reduction from the shared memory values and only the thread 0 writes in global memory

}

//#3 atomic add in memory
__global__ void histogram_intbins_global(const int* __restrict__ bins,
                              size_t n,
                              counter_t* __restrict__ global_hist,
                              int num_bins)
{
    // WRITE THE KERNEL HERE
    // HINT: 
    // the algorithm is simpler: every thread gets a value, identifies the bin, performs an atomicadd into the global memory

}


//#4 shared-memory accelerated
__global__ void histogram_intbins_shared(const int* __restrict__ bins,
                              size_t n,
                              counter_t* __restrict__ global_hist,
                              int num_bins)
{
    // WRITE THE KERNEL HERE
    // HINT:
    // similar to previous one, but instead of making an atomicadd to global memory for every value we have a shared memory slice that represents all the bins. we do our atomic adds there, and only when all values are processed, we do global atomic add from every bin from all blocks into the result array

}




// ============================================================
//   wrapper functions:
//   
//   they setup the grid/block sizes, since every kernel has a different requirement.
// ============================================================

void histogram_bins_v1(const int* d_bins, size_t n, counter_t* d_hist, int num_bins)
{
    const int block = 256;
    // Oversubscribe a bit for latency hiding
    int grid = 1;
    LAUNCH_KERNEL(histogram_single_thread, grid, block, 0, 0, d_bins, n, d_hist, num_bins);
}


void histogram_bins_v2(const int* d_bins, size_t n, counter_t* d_hist, int num_bins)
{
    const int block = 256;
    // Oversubscribe a bit for latency hiding
    int grid = num_bins; 
    LAUNCH_KERNEL(histogram_one_block_per_bin, grid, block, 0, 0, d_bins, n, d_hist, num_bins);
}


void histogram_bins_v3(const int* d_bins, size_t n, counter_t* d_hist, int num_bins)
{

    // size up the grid
    int device = 0;
    HIP_ERRCHK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_ERRCHK(hipGetDeviceProperties(&props, device));

    const int block = 256;
    // oversubscribe a bit for latency hiding
    int maxBlocks = std::max(1, props.multiProcessorCount * 20);
    int grid = std::min<int>( (int)((n + block - 1) / block), maxBlocks);

    LAUNCH_KERNEL(histogram_intbins_global, grid, block, 0, 0, d_bins, n, d_hist, num_bins);
}


void histogram_bins_v4(const int* d_bins, size_t n, counter_t* d_hist, int num_bins)
{


    // size up the grid
    int device = 0;
    HIP_ERRCHK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_ERRCHK(hipGetDeviceProperties(&props, device));

    const int block = 256;
    // oversubscribe a bit for latency hiding
    int maxBlocks = std::max(1, props.multiProcessorCount * 20);
    int grid = std::min<int>( (int)((n + block - 1) / block), maxBlocks);

    // check shared memory availability
    size_t max_shmem = 0;
    HIP_ERRCHK(hipDeviceGetAttribute((int*)&max_shmem, hipDeviceAttributeMaxSharedMemoryPerBlock, device));

    // Try shared path if it fits
    size_t needed = (size_t)num_bins * sizeof(counter_t);

    if (max_shmem > needed) {
        LAUNCH_KERNEL(histogram_intbins_shared, grid, block, needed, 0, d_bins, n, d_hist, num_bins);
    } else {
      printf("there is not enough shmem! not running the kernel");
    }
}

int main(int argc, char** argv)
{
    int algorithm = (argc > 1) ? std::atoi(argv[1]) : 1;
    int num_bins  = (argc > 2) ? std::atoi(argv[2]) : 256;
    size_t N      = (argc > 3) ? static_cast<size_t>(std::atoll(argv[3])) : (size_t)5e6;

    printf("Running HIP histogram with N=%zu, bins=%d\n", N, num_bins);

    // -------- Test A: integer pre-binned input
    {
        std::vector<int> h_bins(N);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, std::max(0, num_bins - 1));
        for (size_t i = 0; i < N; ++i) h_bins[i] = dist(rng);

        int* d_bins = nullptr;
        counter_t* d_hist = nullptr;
        HIP_ERRCHK(hipMalloc(&d_bins, N * sizeof(int)));
        HIP_ERRCHK(hipMalloc(&d_hist, num_bins * sizeof(counter_t)));
        HIP_ERRCHK(hipMemcpy(d_bins, h_bins.data(), N * sizeof(int), hipMemcpyHostToDevice));
        HIP_ERRCHK(hipMemset(d_hist, 0, num_bins * sizeof(counter_t)));

        
        switch (algorithm) {
            case 1:
                histogram_bins_v1(d_bins, N, d_hist, num_bins);
                break;
    
            case 2:
                histogram_bins_v2(d_bins, N, d_hist, num_bins);
                break;
    
            case 3:
                histogram_bins_v3(d_bins, N, d_hist, num_bins);
                break;
            
            case 4:
                histogram_bins_v4(d_bins, N, d_hist, num_bins);
                break;
    
            default:
                printf( "we don't have that many implementations, select a value from 1 to 4\n");
                break;
        }
 

        std::vector<counter_t> h_hist(num_bins);
        HIP_ERRCHK(hipMemcpy(h_hist.data(), d_hist, num_bins * sizeof(counter_t), hipMemcpyDeviceToHost));

        // Validate
        unsigned long long sum = 0ULL;
        for (int b = 0; b < num_bins; ++b) sum += h_hist[b];
        printf("[int-bins] Sum = %llu (expected %zu)\n", (unsigned long long)sum, N);

        HIP_ERRCHK(hipFree(d_bins));
        HIP_ERRCHK(hipFree(d_hist));
    }

    return 0;
}
