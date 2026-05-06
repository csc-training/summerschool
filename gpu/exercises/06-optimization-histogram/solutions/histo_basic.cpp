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
    int b = blockIdx.x;                     // one block per bin
    if (b >= num_bins) return;

    // Parallel reduction inside the block for occurrences of bin b
    int local = 0;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        int x = bins[i];
        if (x == b) local++;
    }

    // Reduce within the block (naive shared-mem sum)
    extern __shared__ counter_t smem[];
    smem[threadIdx.x] = local;
    __syncthreads();

    // standard tree reduction
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hist[b] = smem[0];  // single writer → no atomics
    }
}

//#3 atomic add in memory
__global__ void histogram_intbins_global(const int* __restrict__ bins,
                              size_t n,
                              counter_t* __restrict__ global_hist,
                              int num_bins)
{
    int grid_stride_start = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride_step  = blockDim.x * gridDim.x;

    for (auto i = grid_stride_start; i < n; i += grid_stride_step) {
        int bin = bins[i];
        if (static_cast<unsigned>(bin) < static_cast<unsigned>(num_bins)) {
            atomicAdd(&global_hist[bin], 1ULL);
        }
    }
}


//#4 shared-memory accelerated
__global__ void histogram_intbins_shared(const int* __restrict__ bins,
                              size_t n,
                              counter_t* __restrict__ global_hist,
                              int num_bins)
{
    extern __shared__ counter_t s_hist[]; // size: num_bins * sizeof(counter_t)

    // Initialize shared histogram to zero
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();
    
    int grid_stride_start = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride_step  = blockDim.x * gridDim.x;

    // Accumulate in shared memory
    for (auto i = grid_stride_start; i < n; i += grid_stride_step) {
        int bin = bins[i];
        if (static_cast<unsigned>(bin) < static_cast<unsigned>(num_bins)) {
            atomicAdd(&s_hist[bin], 1ULL);
        }
    }
    __syncthreads();

    // Flush to global
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        counter_t v = s_hist[b];
        if (v) atomicAdd(&global_hist[b], v);
    }
}




// ============================================================
//   wrapper functions
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
