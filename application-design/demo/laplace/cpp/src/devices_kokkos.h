#include <Kokkos_Core.hpp>

#define DEVICE_LAMBDA KOKKOS_LAMBDA

/* This is defined in the Kokkos_Core.hpp
  struct Kokkos::InitArguments {
    int num_threads;
    int num_numa;
    int device_id;
    int ndevices;
    int skip_device;
    bool disable_warnings;
  }; 
*/

namespace devices
{
  inline static void init(int node_rank) {
   Kokkos::InitArguments args;
   args.device_id = node_rank;
   Kokkos::initialize(args);
  }

  inline static void finalize(int rank) {
    Kokkos::finalize();
    printf("Rank %d, Kokkos finalized.\n", rank);
  }

  inline static void* allocate(size_t bytes) {
    return Kokkos::kokkos_malloc(bytes);
  }

  inline static void free(void* ptr) {
    Kokkos::kokkos_free(ptr);
  }

  inline static void memcpyd2d(void *dest, void *src, size_t bytes){
    Kokkos::View<char*> dest_view((char*)dest, bytes);
    Kokkos::View<char*> src_view((char*)src, bytes);
    Kokkos::deep_copy(dest_view, src_view);
  }

  template <typename Lambda>
  inline static void parallel_for(const int nx, const int ny, Lambda loop_body) {
    using MDPolicyType_2D = typename Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<2> >;
    MDPolicyType_2D mdpolicy_2d({0, 0}, {nx, ny});
    Kokkos::parallel_for(mdpolicy_2d, loop_body);
    Kokkos::fence();
  }
}
