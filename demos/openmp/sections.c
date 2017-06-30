#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel
#pragma omp sections
  {
#pragma omp section
   {
    printf("Thread %d in first section\n", omp_get_thread_num());
   }
#pragma omp section
   { 
    printf("Thread %d in second section\n", omp_get_thread_num());
   }
  }
}
