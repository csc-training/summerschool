#include <stdio.h>
#include <omp.h>

int main(void)
{
    int array[4] = {0, 0, 0, 0};
    int tid;

    printf("Array at the beginning: ");
    for (int i=0; i < 4; i++) {
         printf("%d ", array[i]);
    }
    printf("\n");

#pragma omp parallel private(tid)
    #pragma omp single
    {
       tid = omp_get_thread_num();
       printf("Tasks created by %d\n", tid);
 
    for (int i=0; i < 4; i++) {
        #pragma omp task if (i < 3)
        {
           tid = omp_get_thread_num();
           printf("Task %d executed by thread %d\n", i, tid);
           array[i] += tid;
        }
    }
    }


    printf("Array at the end: ");
    for (int i=0; i < 4; i++) {
         printf("%d ", array[i]);
    }
    printf("\n");
 
    return 0;
}
