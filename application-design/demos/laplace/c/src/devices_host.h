#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// All macros and functions can be compiled with a C compiler

inline static void devices_init(int node_rank) {
  // Nothing needs to be done here
}

inline static void devices_finalize(int rank) {
  printf("Rank %d, Host finalized.\n", rank);
}

inline static void* devices_allocate(size_t bytes) {
  return malloc(bytes);
}

inline static void devices_free(void* ptr) {
  free(ptr);
}

inline static void devices_memcpyd2d(void* dst, void* src, size_t bytes){
  memcpy(dst, src, bytes);
}

#define devices_parallel_for(n1, n2, inc1, inc2, i1, i2, loop_body) \
{                                                                   \
  for(inc2 = i2; inc2 < i2 + n2; inc2++){                           \
    for(inc1 = i1; inc1 < i1 + n1; inc1++){                         \
      loop_body;                                                    \
    }                                                               \
  }                                                                 \
}
