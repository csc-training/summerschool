#include <cstdio>
#include <cmath>
#include <omp.h>

int main(void)
{
    int total = 0;

    /* TODO:
     *   Test the effect of different data sharing clauses here
     */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // Calculate factorial of the thread id
        int value = 1;
        for (int i = 1; i <= tid; i++) {
            value *= i;
        }

        // Add the factorial to the total
        total += value;
        printf("Thread %03d added value %d. Total is now %d.\n", tid, value, total);
    }
    printf("Total value is %d\n", total);

    return 0;
}
