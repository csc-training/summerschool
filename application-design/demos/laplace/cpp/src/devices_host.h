#include <cstdio>
#include <cstdlib>
#include <string.h>

#define DEVICE_LAMBDA [=]

namespace devices
{
  inline static void init(int node_rank) {
    // Nothing needs to be done here
  }

  inline static void finalize(int rank) {
    printf("Rank %d, Host finalized.\n", rank);
  }

  inline static void* allocate(size_t bytes) {
    return malloc(bytes);
  }

  inline static void free(void* ptr) {
    ::free(ptr);
  }
  
  inline static void memcpyd2d(void* dst, void* src, size_t bytes){
    memcpy(dst, src, bytes);
  }

  template <typename Lambda>
  inline static void parallel_for(const int nx, const int ny, Lambda loop_body) {
    for(int j = 0; j < ny; j++){
      for(int i = 0; i < nx; i++){
        loop_body(i, j);
      }
    }
  }
}
