#include <stdio.h>
#include <omp.h>
#include <iostream>

static size_t N = 1<<20;

int main() {
  float *a, *b, *c;
  size_t N_bytes = N*sizeof(float);

  a = (float*) malloc(N_bytes);
  b = (float*) malloc(N_bytes);
  c = (float*) malloc(N_bytes);

  for (int k=0;k < N; k++) {
    a[k] = k;
    b[k] = k/2.0;
  }

#pragma omp target enter data map(to: a[:N], b[:N]) map(alloc: c[:N])


// TODO: translate this hip function to equivalent OpenMP region
/*
 __global__ void sum_vecs(float* a, float* b, float* c, size_t n) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t k = tid; k < n; k += stride) 
      c[k] = a[k] +  b[k];
}
*/
// SOLUTION:
#pragma omp target teams parallel
    {
    const int tid = omp_get_team_num() * omp_get_num_threads() +
      omp_get_thread_num();
    const int stride = omp_get_num_teams()*omp_get_num_threads();
    for (size_t k = tid; k<N; k += stride) {
      c[k] = a[k] + b[k];
    }
    }

#pragma omp target update from(c[:N])

  for (size_t k = 0; k < 10; ++k) std::cout << c[k] << " ";
  std::cout << std::endl;

  float *d_c = (float *)omp_get_mapped_ptr(c, omp_get_default_device());

  float total = 0.0f;

#pragma omp target map(tofrom:total) is_device_ptr(d_c)
#pragma omp teams distribute parallel for reduction(+:total)
    for (size_t k=0; k<N; ++k) total += d_c[k];

    std::cout << "sum: " << total << std::endl;

#pragma omp target exit data map(delete: c[:N])
}
