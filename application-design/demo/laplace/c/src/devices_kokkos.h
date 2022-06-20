#include <Kokkos_Core.hpp>

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

// All macros and functions require a C++ compiler (Kokkos API does not support C)
inline static void devices_init(int node_rank) {
  Kokkos::InitArguments args;
  args.device_id = node_rank;
  Kokkos::initialize(args);
}

inline static void devices_finalize(int rank) {
  Kokkos::finalize();
  printf("Rank %d, Kokkos finalized.\n", rank);
}

inline static void* devices_allocate(size_t bytes) {
  return Kokkos::kokkos_malloc(bytes);
}

inline static void devices_free(void* ptr) {
  Kokkos::kokkos_free(ptr);
}

inline static void devices_memcpyd2d(void *dest, void *src, size_t bytes){
  Kokkos::View<char*> dest_view((char*)dest, bytes);
  Kokkos::View<char*> src_view((char*)src, bytes);
  Kokkos::deep_copy(dest_view, src_view);
}

#define devices_parallel_for(n1, n2, inc1, inc2, i1, i2, loop_body)          \
{                                                                            \
  auto lambda_body = KOKKOS_LAMBDA(int inc1, int inc2) {                     \
    inc1 += i1;                                                              \
    inc2 += i2;                                                              \
    loop_body;                                                               \
  };                                                                         \
                                                                             \
  using MDPolicyType_2D = typename Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<2> >; \
  MDPolicyType_2D mdpolicy_2d({0, 0}, {n1, n2});                             \
  Kokkos::parallel_for(mdpolicy_2d, lambda_body);                            \
  Kokkos::fence();                                                           \
  (void)inc1;(void)inc2;                                                     \
}
