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

    // TODO: launch threads and create tasks so that there 
    // one task per loop iteration 
    for (int i=0; i < 4; i++) {
           tid = omp_get_thread_num();
           printf("Task %d executed by thread %d\n", i, tid);
           array[i] += tid;
    }

    // TODO end
       
    printf("Array at the end: ");
    for (int i=0; i < 4; i++) {
         printf("%d ", array[i]);
    }
    printf("\n");
 
    return 0;
}
