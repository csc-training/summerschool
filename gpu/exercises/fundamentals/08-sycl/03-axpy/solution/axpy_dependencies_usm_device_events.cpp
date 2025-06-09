#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>
#include <algorithm>

using namespace sycl;

int main() {
  // Create an out-of-order queue on any available device
  queue q(default_selector_v);

  constexpr size_t N = 25600;
  std::vector<int> x(N), y(N);
  int a = 4;

  // Allocate USM device memory
  int* d_x = malloc_device<int>(N, q);
  int* d_y = malloc_device<int>(N, q);

  // Kernel to initialize d_x 
  auto event_x = q.submit([&](handler& h) {
    h.parallel_for(range<1>{N}, [=](id<1> idx) {
      d_x[idx] = 1;
    });
  });

  // Kernel to initialize d_y
  auto event_y = q.submit([&](handler& h) {
    h.parallel_for(range<1>{N}, [=](id<1> idx) {
      d_y[idx] = 2;
    });
  });

  // Submit axpy kernel dependent on event_x and event_y
  auto event_axpy = q.submit([&](handler& h) {
    h.depends_on({event_x, event_y});
    h.parallel_for(range<1>{N}, [=](id<1> idx) {
      d_y[idx] += a * d_x[idx];
    });
  });

  // Submit memcpy dependent on event_axpy
  auto memcpy_event = q.memcpy(y.data(), d_y, N * sizeof(int), {event_axpy});

  // Wait for memcpy to complete before using y on host
  memcpy_event.wait();

  // Free device memory
  free(d_x, q);
  free(d_y, q);

  // Validate results
  bool passed = std::all_of(y.begin(), y.end(),
                            [a](int val) { return val == a * 1 + 2; });

  std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;

  return (passed) ? 0 : 1;
}
