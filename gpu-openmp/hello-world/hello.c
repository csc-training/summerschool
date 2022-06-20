#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() 
{
  int num_devices = omp_get_num_devices();
  printf("Number of available devices %d\n", num_devices);

#pragma omp target 
  {
    if (omp_is_initial_device()) {
      printf("Running on host\n");    
    } else {
      printf("Running on device\n");
    }
  }

}
