#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel
  {
  int i = 0;
#pragma omp master
    while ( i < 6 ) {
      #pragma omp task
      {
        int thread_id = omp_get_thread_num();
        printf("Task %d, thread %d\n", i, thread_id);
      }
      i++;
    }
  }
}
