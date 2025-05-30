#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Set up queue on any available device
  queue q;
  
  // Initialize input and output memory on the host
  constexpr size_t N = 25600;
  int a=4;
  std::vector<int> x(N), y(N);
  
  {
    buffer<int> x_buf(x.data(), range<1>(N));
    buffer<int> y_buf(y.data(), range<1>(N));
    // Kernel to initialize x
    q.submit([&](handler& h) {
        accessor x_acc(x_buf, h, write_only, no_init);
        h.parallel_for(range<1>(N), [=](id<1> i) {
            x_acc[i] = 1;
        });
    });
    
    // Kernel to initialize y
    q.submit([&](handler& h) {
        accessor y_acc(y_buf, h, write_only, no_init);
        h.parallel_for(range<1>(N), [=](id<1> i) {
            y_acc[i] = 2;
        });
    });

    
    // Kernel to compute y = a * x + y
    q.submit([&](handler& h) {
      // Create accessors
      accessor x_acc(x_buf, h, read_only);
      accessor y_acc(y_buf, h, read_write);

      h.parallel_for(range<1>(N), [=](id<1> i) {
        y_acc[i] = a * x_acc[i] +  y_acc[i];
      });
    });
    //Checking the result inside the scope of the buffers using host_accessors
    host_accessor y_acc(y_buf, read_only);
    bool passed = std::all_of(y_acc.begin(), y_acc.end(),
                              [a](int val) { return val == a * 1 + 2; });
    std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;
    
  }

  // Check that all outputs match expected value
  bool passed = std::all_of(y.begin(), y.end(),
                            [a](int val) { return val == a * 1 + 2; });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
            << std::endl;
  return (passed) ? 0 : 1;
}
