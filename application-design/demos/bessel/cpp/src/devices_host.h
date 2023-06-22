#include <cstdio>
#include <cstdlib>
#include <random>
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
  
  inline static void memcpy_d2d(void* dst, void* src, size_t bytes){
    memcpy(dst, src, bytes);
  }

  template <typename Lambda>
  inline static void parallel_for(int loop_size, Lambda loop_body) {
    for(int i = 0; i < loop_size; i++){
      loop_body(i);
    }
  }

  template <typename Lambda, typename T>
  inline static void parallel_reduce(const int loop_size, Lambda loop_body, T *sum) {
    for(int i = 0; i < loop_size; i++){
      loop_body(i, *sum);
    }
  }

  template <typename T>
  inline static void atomic_add(T *array_loc, T value){
    *array_loc += value;
  }

  template <typename T>
  inline static T random_float(unsigned long long seed, unsigned long long seq, int idx, T mean, T stdev){
    
    // Re-seed the first case
    if(idx == 0){
      // Overflow is defined behavior with unsigned, and therefore ok here
      srand((unsigned int)seed + (unsigned int)seq);
    }
  
    // Use Box Muller algorithm to get a float from a normal distribution
    const float two_pi = 2.0f * M_PI;
	  float u1 = (float) rand() / RAND_MAX;
	  float u2 = (float) rand() / RAND_MAX;
	  float factor = stdev * sqrtf (-2.0f * logf (u1));
	  float trig_arg = two_pi * u2;
	  
    // Box Muller algorithm produces two random normally distributed floats, z0 and z1
    float z0 = factor * cosf (trig_arg) + mean; // Need only one
	  // float z1 = factor * sinf (trig_arg) + mean; 
    return z0;
  }
}
