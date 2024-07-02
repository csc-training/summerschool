#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int main()
{
  int num_devices = omp_get_num_devices();  // Number of devices available.
  printf("Number of available devices %d\n", num_devices);

#pragma omp target
  {
    if (omp_is_initial_device()) {
      printf("Running on host.\n");  // Running on host device, i.e. a CPU node.
    } else {
      printf("Running on device.\n");  // Running on GPU device.
    }
  }

}

// When compiled with the modules loaded using `source ../modules-cpu`, a CPU version is compiled.
// When compiled with the modules loaded using `source ../modules-gpu`, a GPU version is compiled.