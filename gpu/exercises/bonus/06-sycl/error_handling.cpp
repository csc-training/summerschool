#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  // Define an asynchronous exception handler
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };

  // Create a SYCL queue with the exception handler
  sycl::queue queue(sycl::default_selector_v, exception_handler);

  constexpr size_t N = 10000; // Array size
  constexpr size_t wg_size = 1024; 
  constexpr size_t grid_size = 512 ; 
  
  // Allocate USM memory for array Y on the device
  int* y = sycl::malloc_device<int>(N, queue);

  // Initialize array Y to zero
  queue.fill(y, 0, N).wait();

  // Submit kernel with errors:
  // - Out-of-bounds memory access (y[N + idx])
  // - Work-group size larger than grid size
  queue.submit([&](sycl::handler& cgh) {
    auto range = sycl::nd_range<1>(sycl::range<1>(grid_size), sycl::range<1>(wg_size));
    cgh.parallel_for<class erroneous_kernel>(range, [=](sycl::nd_item<1> item) {
      size_t idx = item.get_global_id(0);
      y[N + idx]++; 
    });
  });

  // Wait for kernel execution and catch any synchronous exceptions
  try {
    queue.wait_and_throw();
  } catch (sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }

  // Free device memory
  sycl::free(y, queue);

  return 0;
}
