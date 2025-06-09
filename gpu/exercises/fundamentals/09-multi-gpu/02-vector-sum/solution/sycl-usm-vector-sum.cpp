#include <iostream>
#include <vector>
#include <numeric>
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

struct Decomp {
    int len;
    int start;
};

int main() {
    constexpr int ThreadsInBlock = 128;
    constexpr int N = 100;

    // Query and check that we have at least two GPUs
    auto gpus = device::get_devices(info::device_type::gpu);
    if (gpus.size() < 2) {
        std::cerr << "Error: This program requires at least 2 GPU devices\n";
        return 1;
    }

    std::cout << "Found " << gpus.size() << " GPU devices, using GPUs 0 and 1!\n\n";

    // Create two in-order queues with profiling enabled
    queue queues[2] = {
        queue(gpus[0], {property::queue::in_order(), property::queue::enable_profiling()}),
        queue(gpus[1], {property::queue::in_order(), property::queue::enable_profiling()})
    };

    // Decomposition into two parts
    Decomp dec[2];
    dec[0].len   = N / 2;
    dec[0].start = 0;
    dec[1].len   = N - N / 2;
    dec[1].start = dec[0].len;

    // Host memory (pinned)
    double *hA = malloc_host<double>(N, queues[0]);
    double *hB = malloc_host<double>(N, queues[0]);
    double *hC = malloc_host<double>(N, queues[0]);

    // Initialize inputs
    for (int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    // Allocate device memory per GPU
    double *dA[2], *dB[2], *dC[2];
    for (int i = 0; i < 2; ++i) {
        dA[i] = malloc_device<double>(dec[i].len, queues[i]);
        dB[i] = malloc_device<double>(dec[i].len, queues[i]);
        dC[i] = malloc_device<double>(dec[i].len, queues[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<event> final_events;
    std::vector<event> kernel_events;

    for (int i = 0; i < 2; ++i) {
        auto &q = queues[i];
        auto &len = dec[i].len;
        auto &start_idx = dec[i].start;

        // Async copy: hA/hB -> dA/dB
        event e1 = q.memcpy(dA[i], hA + start_idx, sizeof(double) * len);
        event e2 = q.memcpy(dB[i], hB + start_idx, sizeof(double) * len);

        // Kernel: dC = dA + dB
        event compute = q.submit([&](handler &h) {
            // No depends_on needed for in-order queue
            h.parallel_for(range<1>(len), [=](id<1> idx) {
                dC[i][idx] = dA[i][idx] + dB[i][idx];
            });
        });

        // Async copy: dC -> hC
        event e3 = q.memcpy(hC + start_idx, dC[i], sizeof(double) * len);

        final_events.push_back(e3);
        kernel_events.push_back(compute);
    }

    // Wait for all final copy events (meaning all GPU work done)
    for (auto &e : final_events)
        e.wait();

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    // Validate result
    int errorsum = 0;
    for (int i = 0; i < N; ++i) {
        errorsum += static_cast<int>(hC[i] - 3.0);
    }
    std::cout << "Error sum = " << errorsum << "\n";
    std::cout << "Total elapsed time (host wall clock): " << elapsed.count() << " seconds\n";

    // Free device memory
    for (int i = 0; i < 2; ++i) {
        free(dA[i], queues[i]);
        free(dB[i], queues[i]);
        free(dC[i], queues[i]);
    }

    // Free host memory
    free(hA, queues[0]);
    free(hB, queues[0]);
    free(hC, queues[0]);

    return 0;
}
