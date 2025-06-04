#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>
#include <algorithm>

using namespace sycl;

int main() {
  // Set up queue an in-order queue on any available device
   queue q(default_selector_v, property::queue::in_order{});

  // Initialize input and output memory on the host
  constexpr size_t N = 25600;
  std::vector<int> x(N), y(N);
  int a = 4;
  std::fill(x.begin(), x.end(), 1);
  std::fill(y.begin(), y.end(), 2);

  // Allocate USM device memory
  int* d_x = malloc_device<int>(N, q);
  int* d_y = malloc_device<int>(N, q);

  // Initialize d_x on device 
  q.submit([&](handler& h) {
    h.parallel_for(range<1>(N), [=](id<1> i) {
      d_x[i] = 1;
    });
  });
  
  // Initialize d_y on device 
  q.submit([&](handler& h) {
    h.parallel_for(range<1>(N), [=](id<1> i) {
      d_y[i] = 2;
    });
  });


  // Kernel to perform y=y+a*x
  q.submit([&](handler& h) {
    h.parallel_for(range<1>{N}, [=](id<1> idx) {
      int i = idx[0];
      d_y[i] = a * d_x[i] + d_y[i];
    });
  });

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
