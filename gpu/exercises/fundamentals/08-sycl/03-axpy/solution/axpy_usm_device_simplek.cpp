#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>
#include <algorithm>

using namespace sycl;

int main() {
  // Set up queue on any available device
  queue q;

  // Initialize input and output memory on the host
  constexpr size_t N = 25600;
  std::vector<int> x(N), y(N);
  int a = 4;
  std::fill(x.begin(), x.end(), 1);
  std::fill(y.begin(), y.end(), 2);

  // Allocate USM device memory
  int* d_x = malloc_device<int>(N, q);
  int* d_y = malloc_device<int>(N, q);

  // Copy data from host to device
  q.memcpy(d_x, x.data(), N * sizeof(int)).wait();
  q.memcpy(d_y, y.data(), N * sizeof(int)).wait();

  // Submit the kernel to the queue
  q.submit([&](handler& h) {
    h.parallel_for(range<1>{N}, [=](id<1> idx) {
      int i = idx[0];
      d_y[i] = a * d_x[i] + d_y[i];
    });
  }).wait();

  // Copy result back to host
  q.memcpy(y.data(), d_y, N * sizeof(int)).wait();

  // Free device memory
  free(d_x, q);
  free(d_y, q);

  // Check that all outputs match expected value
  bool passed = std::all_of(y.begin(), y.end(),
                            [a](int val) { return val == a * 1 + 2; });
  std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;
  return (passed) ? 0 : 1;
}
