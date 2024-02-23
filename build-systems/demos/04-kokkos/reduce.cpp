#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>

constexpr size_t N = 10000;
typedef float numtype;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    numtype sum=0;
    Kokkos::parallel_reduce(N+1, KOKKOS_LAMBDA(const int i, numtype &lsum) {
      lsum += 1.0*i;
    }, Kokkos::Sum<numtype>(sum));
    Kokkos::fence();

    std::cout  << "reduction sum is " << std::setprecision(18) 
      << sum << std::endl;
    std::cout << "error to exact is " << std::setprecision(18) 
      << (N*(1+N) - sum*2)/2 << std::endl << "... or is it?\n";
  }
  Kokkos::finalize();
}
