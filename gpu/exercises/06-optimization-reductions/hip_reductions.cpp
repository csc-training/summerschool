#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

#include "../../../error_checking.hpp"

// ==========================================================
// 1) Naive: each thread atomically adds its grid-stride sum
//    Extremely slow baseline (max atomics).
// ==========================================================
__global__ void reduce_naive_atomic(const double* __restrict__ in,
                                    double* __restrict__ out,
                                    size_t N)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (size_t i = tid; i < N; i += stride)
        sum += in[i];

    // Every thread does one atomic add
    atomicAdd(out, sum);
}

// ==========================================================
// 2) Naive (per-block): each block reduces scalars to one
//     and atomically adds once. No shared memory tree,
//     just serial block accumulation by thread 0.
//     (Minimal code, still self-sufficient.)
// ==========================================================
__global__ void reduce_naive_blockatomic(const double* __restrict__ in,
                                         double* __restrict__ out,
                                         size_t N)
{
}

// ==========================================================
// 3) Shared memory tree: one atomic per block
//     + grid-stride, 2 elements per thread per iteration.
// ==========================================================
__global__ void reduce_shared_atomic(const double* __restrict__ in,
                                     double* __restrict__ out,
                                     size_t N)
}


// ============================
// CPU reference
// ============================
double cpu_reduce(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return (double)s;
}

// ============================
// hipCUB reference (DeviceReduce::Sum)
// ============================
double run_hipcub(const double* d_in, size_t N)
{
    // hipCUB API: two-stage call to get temp storage size
    double* d_out = nullptr;
    HIP_ERRCHK(hipMalloc(&d_out, sizeof(double)));

    void* d_temp_storage = nullptr;
    size_t temp_bytes = 0;

    HIP_ERRCHK(hipcub::DeviceReduce::Sum(
        d_temp_storage, temp_bytes, d_in, d_out, N
    ));

    HIP_ERRCHK(hipMalloc(&d_temp_storage, temp_bytes));

    HIP_ERRCHK(hipcub::DeviceReduce::Sum(
        d_temp_storage, temp_bytes, d_in, d_out, N
    ));

    HIP_ERRCHK(hipDeviceSynchronize());

    double result = 0.0;
    HIP_ERRCHK(hipMemcpy(&result, d_out, sizeof(double), hipMemcpyDeviceToHost));

    HIP_ERRCHK(hipFree(d_temp_storage));
    HIP_ERRCHK(hipFree(d_out));
    return result;
}
// ============================
// Kernel launcher
// ============================
enum Mode {
    M_NAIVE = 0,
    M_BLOCKATOMIC,
    M_SHARED,
    M_HIPCUB
};

const char* mode_name(Mode m) {
    switch (m) {
        case M_NAIVE:       return "naive";
        case M_BLOCKATOMIC: return "blockatomic";
        case M_SHARED:      return "shared";
        case M_HIPCUB:      return "hipcub";
        default:            return "unknown";
    }
}

double run_kernel(const double* d_in, size_t N,
                 Mode mode, int blocks, int threads)
{
    if (mode == M_HIPCUB) {
        // hipCUB chooses its own kernel config internally
        return run_hipcub(d_in, N);
    }

    double *d_out;
    HIP_ERRCHK(hipMalloc(&d_out, sizeof(double)));
    HIP_ERRCHK(hipMemset(d_out, 0, sizeof(double)));

    size_t smem = threads * sizeof(double);

    switch (mode) {
    case M_NAIVE:
        LAUNCH_KERNEL(reduce_naive_atomic,blocks, threads, 0,0,d_in, d_out, N);
        break;
    case M_BLOCKATOMIC:
        LAUNCH_KERNEL(reduce_naive_blockatomic,blocks, threads, 0,0,d_in, d_out, N);
        break;
    case M_SHARED:
        LAUNCH_KERNEL(reduce_shared_atomic,blocks, threads, smem,0,d_in, d_out, N);
        break;
    case M_HIPCUB:
        printf("should not be here");
        break;
    }
    HIP_ERRCHK(hipGetLastError());
    HIP_ERRCHK(hipDeviceSynchronize());

    double result;
    HIP_ERRCHK(hipMemcpy(&result, d_out, sizeof(double), hipMemcpyDeviceToHost));
    HIP_ERRCHK(hipFree(d_out));
    return result;
}

// ============================
// CLI & Main
// ============================
int main(int argc, char** argv)
{
    size_t N = 1ull << 24;
    Mode mode = M_OPTIMIZED;
    int threads = 256;
    int blocks  = 0;      // 0 = auto
    int repeat  = 1;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-N" && i+1 < argc) {
            N = std::stoull(argv[++i]);
        } else if (a == "--mode" && i+1 < argc) {
            std::string m = argv[++i];
            if      (m == "naive")       mode = M_NAIVE;
            else if (m == "blockatomic") mode = M_BLOCKATOMIC;
            else if (m == "shared")      mode = M_SHARED;
            else if (m == "hipcub")      mode = M_HIPCUB;
            else {
                std::cerr << "Unknown --mode\n"; return 1;
            }
        } else if (a == "--threads" && i+1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (a == "--blocks" && i+1 < argc) {
            blocks = std::stoi(argv[++i]);
        } else if (a == "--repeat" && i+1 < argc) {
            repeat = std::stoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage: --mode [naive|blockatomic|shared|hipcub]\n"
                "       --N <elements>\n"
                "       --threads <blockDim>\n"
                "       --blocks <gridDim> (auto if omitted)\n"
                "       --repeat <runs>\n";
            return 0;
        }
    }

    // Device info + default blocks
    hipDeviceProp_t prop{};
    HIP_ERRCHK(hipGetDeviceProperties(&prop, 0));
    int mp = prop.multiProcessorCount; // CUs per GCD (gfx90a ~110 per GCD)
    if (blocks <= 0) {
        // Good starting point: ~20x CUs for latency hiding
        blocks = std::max( (int)(2 * mp), 1 );
        // But don't exceed N coverage for 2x loading kernels
        int max_blocks = (int)((N + (size_t)threads*2 - 1) / ((size_t)threads*2));
        std::cout<<"Running with "<<blocks<< "blocks" <<std::endl;
    }

    std::cout << "Device: " << prop.name << " (MPs=" << mp << ")\n";
    std::cout << "Mode=" << mode_name(mode)
              << " N=" << N
              << " blocks=" << (mode==M_MULTIWAVE?1:blocks)
              << " threads=" << threads
              << " repeat=" << repeat
              << (HAS_NT_LOAD && mode==M_NONTEMPORAL ? " [nontemporal=ON]" : "")
              << "\n";

    // Host data
    std::vector<double> h(N, 1.0f); // simple correctness check
    double ref = cpu_reduce(h);

    // Device buffer
    double* d_in = nullptr;
    HIP_ERRCHK(hipMalloc(&d_in, N * sizeof(double)));
    HIP_ERRCHK(hipMemcpy(d_in, h.data(), N*sizeof(double), hipMemcpyHostToDevice));

    // Timing
    hipEvent_t start, stop;
    HIP_ERRCHK(hipEventCreate(&start));
    HIP_ERRCHK(hipEventCreate(&stop));

    float best_ms = std::numeric_limits<float>::max();
    double last_result = 0.0;

    for (int r = 0; r < repeat; ++r) {
        HIP_ERRCHK(hipDeviceSynchronize());
        HIP_ERRCHK(hipEventRecord(start));

        last_result = run_kernel(d_in, N, mode, blocks, threads);

        HIP_ERRCHK(hipEventRecord(stop));
        HIP_ERRCHK(hipEventSynchronize(stop));

        float ms;
        HIP_ERRCHK(hipEventElapsedTime(&ms, start, stop));
        best_ms = std::min(best_ms, ms);
    }

    std::cout << "GPU result = " << last_result << "\n";
    std::cout << "CPU result = " << ref << "\n";
    std::cout << "Abs diff   = " << std::abs(last_result - ref) << "\n";
    std::cout << "Best time  = " << best_ms << " ms\n";

    HIP_ERRCHK(hipFree(d_in));
    HIP_ERRCHK(hipEventDestroy(start));
    HIP_ERRCHK(hipEventDestroy(stop));

    return 0;
}
