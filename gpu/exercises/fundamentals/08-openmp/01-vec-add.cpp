#include <stdio.h>
#include <omp.h>
#include <iostream>

static size_t N = 1<<20;

// Declare interface
void hip_sum(float* a, float* b, float* c, size_t n);

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

// TODO: offload this loop
printf("a: %016zx\n", (size_t)a);
#pragma omp target is_device_ptr(a)
#pragma omp teams
  {
/* #pragma omp parallel */
  /* if(omp_get_team_num() == 0) */
  printf("a: %016zx, %05i, %03i, %2.4f\n", (size_t)a, omp_get_team_num(), omp_get_thread_num(), a[omp_get_thread_num()]);
/* #pragma omp distribute parallel for */
/*   for (size_t k = 0; k < N; ++k) */
/*   c[k] = a[k] + b[k]; */
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

    free(a); free(b); free(c);
}
