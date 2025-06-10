#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>

using namespace sycl;

void copyP2P(bool p2p, queue &q0, queue &q1, int* dA_0, int* dA_1, int N) {
    // Dummy copy to warm up
    q0.memcpy(dA_0, dA_1, sizeof(int)*N).wait();

    // Timed copies
    int M = 10;
    auto tStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        if (p2p) {
            q0.memcpy(dA_0, dA_1, sizeof(int)*N);
        } else {
            // Host-mediated copy as fallback (manual P2P simulation)
            int* temp = malloc_host<int>(N, q0.get_context());
            q1.memcpy(temp, dA_1, sizeof(int)*N).wait();
            q0.memcpy(dA_0, temp, sizeof(int)*N).wait();
            free(temp, q0.get_context());
        }
    }
    q0.wait(); // Ensure all copies complete
    auto tStop = std::chrono::high_resolution_clock::now();

    double time_s = std::chrono::duration_cast<std::chrono::nanoseconds>(tStop - tStart).count() / 1e9;
    double bandwidth = static_cast<double>(N * sizeof(int)) / (1024.0 * 1024.0 * 1024.0) / (time_s / M);

    if (p2p)
        std::cout << "P2P enabled - Bandwidth: " << bandwidth << " GB/s, Time: " << time_s << " s\n";
    else
        std::cout << "P2P disabled - Bandwidth: " << bandwidth << " GB/s, Time: " << time_s << " s\n";
}

int main() {
    std::vector<device> gpus;

    for (const auto& dev : device::get_devices()) {
        if (dev.is_gpu()) {
            gpus.push_back(dev);
        }
    }

    if (gpus.size() < 2) {
        std::cout << "Need at least two GPUs!\n";
        return EXIT_FAILURE;
    } else {
        std::cout << "Found " << gpus.size() << " GPU devices\n";
    }

    int gpu0 = 0, gpu1 = 1;
    queue q0(gpus[gpu0]);
    queue q1(gpus[gpu1]);

    std::cout << "Using GPU " << gpu0 << ": " << gpus[gpu0].get_info<info::device::name>() << "\n";
    std::cout << "Using GPU " << gpu1 << ": " << gpus[gpu1].get_info<info::device::name>() << "\n";

    int N = 1 << 28;
    int* dA_0 = malloc_device<int>(N, q0);
    int* dA_1 = malloc_device<int>(N, q1);

    // Simulate P2P copy
    copyP2P(true, q0, q1, dA_0, dA_1, N);
    // Fallback non-P2P copy
    copyP2P(false, q0, q1, dA_0, dA_1, N);

    free(dA_0, q0);
    free(dA_1, q1);

    return 0;
}
