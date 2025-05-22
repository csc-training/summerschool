#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Set up queue on any available device
  //TODO 

  // Initialize input and output memory on the host
  constexpr size_t N = 25600;
  std::vector<int> x(N),y(N);
  int a=4;
  std::fill(x.begin(), x.end(), 1);
  std::fill(y.begin(), y.end(), 2);

  {
   // Create buffers for the host data
   // TODO

    // Submit the kernel to the queue
    q.submit([&](handler& h) {
      // Create accessors
      //TODO

      h.parallel_for(
        //The kernel as a lambda
        //TODO
      );
    });

      //TODO after the submission works
      //Checking the result inside the scope of the buffers using host_accessors
  }

  // Check that all outputs match expected value
  bool passed = std::all_of(y.begin(), y.end(),
                            [](int i) { return (i == 1+a*2); });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
            << std::endl;
  return (passed) ? 0 : 1;
}
