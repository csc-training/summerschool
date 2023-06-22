#include <math.h>
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

inline static void devices_memcpy_d2d(void* dst, void* src, size_t bytes){
  memcpy(dst, src, bytes);
}

inline static void devices_atomic_add(float *array_loc, float value){
  *array_loc += value;
}

inline static float devices_random_float(unsigned long long seed, unsigned long long seq, int idx, float mean, float stdev){
  
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

#define devices_parallel_for(loop_size, inc, loop_body)             \
{                                                                   \
  for(inc = 0; inc < loop_size; inc++){                             \
    loop_body;                                                      \
  }                                                                 \
}
