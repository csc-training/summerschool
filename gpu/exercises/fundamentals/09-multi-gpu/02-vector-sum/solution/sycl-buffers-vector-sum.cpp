#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

struct Decomp {
    int len;
    int start;
};

int main() {
    namespace sycl = cl::sycl;

    const int N = 100;
    const int num_devices_required = 2;

    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    if (devices.size() < num_devices_required) {
        std::cerr << "Found " << devices.size() << " GPU device(s), but require "
                  << num_devices_required << ". Exiting.\n";
        return -1;
    }

    std::cout << "Using GPUs 0 and 1 for computation.\n";

    std::vector<double> hA(N, 1.0);
    std::vector<double> hB(N, 2.0);
    std::vector<double> hC(N, 0.0);

    Decomp dec[2];
    dec[0].len = N / 2;
    dec[0].start = 0;
    dec[1].len = N - dec[0].len;
    dec[1].start = dec[0].len;

    // Single buffers over the entire arrays
    sycl::buffer<double, 1> bufA(hA.data(), N);
    sycl::buffer<double, 1> bufB(hB.data(), N);
    sycl::buffer<double, 1> bufC(hC.data(), N);

    sycl::queue queues[2] = { sycl::queue(devices[0]), sycl::queue(devices[1]) };

    // Submit kernels using global start offsets
    auto e0 = queues[0].submit([&](sycl::handler& cgh) {
        auto A = bufA.get_access<sycl::access::mode::read>(cgh);
        auto B = bufB.get_access<sycl::access::mode::read>(cgh);
        auto C = bufC.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class vector_add_0>(sycl::range<1>(dec[0].len), [=](sycl::id<1> i) {
            size_t idx = i[0] + dec[0].start;
            C[idx] = A[idx] + B[idx];
        });
    });

    auto e1 = queues[1].submit([&](sycl::handler& cgh) {
        auto A = bufA.get_access<sycl::access::mode::read>(cgh);
        auto B = bufB.get_access<sycl::access::mode::read>(cgh);
        auto C = bufC.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class vector_add_1>(sycl::range<1>(dec[1].len), [=](sycl::id<1> i) {
            size_t idx = i[0] + dec[1].start;
            C[idx] = A[idx] + B[idx];
        });
    });

    e0.wait();
    e1.wait();

    // Verify results
    int errorsum = 0;
    for (int i = 0; i < N; i++) {
        if (hC[i] != 3.0)
            errorsum++;
    }

    std::cout << "Error count = " << errorsum << std::endl;

    return 0;
}
