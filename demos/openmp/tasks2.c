#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel
  {
  int i=0;
#pragma omp master
    while ( i < 6 ) {
      #pragma omp task shared(i)
      if (omp_get_thread_num() == 1) i=100;

      i++;
    }
#pragma omp barrier
    if (omp_get_thread_num() == 0) printf("i is %d\n", i);
  }
}
