#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Set up queue on any available device
  queue q;
  
  // Initialize input and output memory on the host
  constexpr size_t N = 25600;
  std::vector<int> x(N),y(N);
  int a=4;
  std::fill(x.begin(), x.end(), 1);
  std::fill(y.begin(), y.end(), 2);
  
  {
    // Create buffers for the host data
    buffer<int> x_buf(x.data(), range<1>(N));
    buffer<int> y_buf(y.data(), range<1>(N));
    
    // Submit the kernel to the queue
    q.submit([&](handler& h) {
      // Create accessors
      accessor x_acc(x_buf, h, read_only);
      accessor y_acc(y_buf, h, read_write);
      
      // Define work-group size and global size
      size_t local_size = 256;
      size_t global_size = ((N + local_size - 1) / local_size) * local_size;

      // Pass nd_range directly using those named variables
      h.parallel_for(
        nd_range<1>(range<1>(global_size), range<1>(local_size)),
        [=](nd_item<1> item) {
          size_t i = item.get_global_id(0);
          if (i < N) {
            y_acc[i] = a * x_acc[i] + y_acc[i];
          }
        });
    });

    //Checking the result inside the scope of the buffers using host_accessors
    host_accessor y_acc(y_buf, read_only);
    bool passed = std::all_of(y_acc.begin(), y_acc.end(),
                              [a](int val) { return val == a * 1 + 2; });
    std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;
    return passed ? 0 : 1;
  }

  // Check that all outputs match expected value
  bool passed = std::all_of(y.begin(), y.end(),
                            [a](int val) { return val == a * 1 + 2; });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
            << std::endl;
  return (passed) ? 0 : 1;
}
