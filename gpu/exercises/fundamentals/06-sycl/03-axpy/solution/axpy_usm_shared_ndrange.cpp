#include <iostream>
#include <sycl/sycl.hpp>
#include <algorithm>  // for std::all_of
using namespace sycl;

int main() {
  queue q;

  constexpr size_t N = 25600;
  int a = 4;

  int* x = malloc_shared<int>(N, q);
  int* y = malloc_shared<int>(N, q);

  std::fill(x, x + N, 1);
  std::fill(y, y + N, 2);

  q.submit([&](handler& h) {// Define work-group size and global size
    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    
    h.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)),[=](nd_item<1> item) {
      size_t i = item.get_global_id(0);
      if (i < N) {
        y[i] = a * x[i] + y[i];
      }
    });
  });

  q.wait();

  // Check that all outputs match expected value
  bool passed = std::all_of(y, y + N, 
                          [a](int val) { return val == a * 1 + 2; });
  std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;
  return (passed) ? 0 : 1;
  // Free device memory
  free(x, q);
  free(y, q);
}
