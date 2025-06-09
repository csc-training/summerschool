#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

struct Decomp {
    int len;
    int start;
};

int main() {
    constexpr int N = 100;

    // Query and check that we have at least two GPUs
    auto gpus = device::get_devices(info::device_type::gpu);
    if (gpus.size() < 2) {
        std::cerr << "Error: This program requires at least 2 GPU devices\n";
        return 1;
    }

    std::cout << "Found " << gpus.size() << " GPU devices, using GPUs 0 and 1!\n\n";

    // Create two in-order queues with profiling enabled
    queue queues[2];
    for (int i = 0; i < 2; ++i) {
        queues[i] = queue(gpus[i], {property::queue::in_order(), property::queue::enable_profiling()});
    }

    // Decomposition into two parts
    Decomp dec[2];
    dec[0].len   = N / 2;
    dec[0].start = 0;
    dec[1].len   = N - N / 2;
    dec[1].start = dec[0].len;

    // Allocate host memory
    std::vector<double> hA(N, 1.0);  // Initialized to 1.0
    std::vector<double> hB(N, 2.0);  // Initialized to 2.0
    std::vector<double> hC(N, 0.0);  // Will hold the result

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<event> kernel_events;

    for (int i = 0; i < 2; ++i) {
        auto &q = queues[i];
        auto len = dec[i].len;
        auto start_idx = dec[i].start;

        // Create per-device sub-buffers from host arrays
        buffer<double> a_buf(hA.data() + start_idx, range<1>(len));
        buffer<double> b_buf(hB.data() + start_idx, range<1>(len));
        buffer<double> c_buf(hC.data() + start_idx, range<1>(len));

        // Submit work to each device
        kernel_events.push_back(
            q.submit([=](handler &h) {
                accessor a(a_buf, h, read_only);
                accessor b(b_buf, h, read_only);
                accessor c(c_buf, h, write_only);

                h.parallel_for(range<1>(len), [=](id<1> idx) {
                    c[idx] = a[idx] + b[idx];
                });
            })
        );
    }

    // Wait for all kernels to finish
    for (auto &e : kernel_events) e.wait();

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    // Validate result
    int errorsum = 0;
    for (int i = 0; i < N; ++i) {
        errorsum += static_cast<int>(hC[i] - 3.0);  // 1.0 + 2.0 = 3.0
    }

    std::cout << "Error sum = " << errorsum << "\n";
    std::cout << "Total elapsed time (host wall clock): " << elapsed.count() << " seconds\n";

    return 0;
}
