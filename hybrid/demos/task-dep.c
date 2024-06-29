
#include <stdio.h>
#include <omp.h>
#include <unistd.h>


int main(int argc, char *argv[])
{

  int data = 1;
  #pragma omp parallel
  #pragma omp single
  {
    for(int i=0;i<5; i++)
    {
      #pragma omp task depend(in: data) // in: depend on all previous out/inout tasks
      {
        printf("task A %d\n", i);
        sleep(1);
      }
      #pragma omp task depend(inout: data) // inout: depend on all previous in/inout/out tasks
      {
        printf("task B %d\n", i);
      }
    }
    #pragma omp task depend(out: data) // out: depend on all previous in/inout/out tasks (same as out)
    {
      printf("task C\n");
    }
  }
  return 0;
}
