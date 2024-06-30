#include <cstdio>

int main(void)
{
    int var1 = 1, var2 = 2;

    /* TODO:
     *   Test the effect of different data sharing clauses here
     */
    #pragma omp parallel firstprivate(var1, var2)
    {
        printf("Region 1: var1=%i, var2=%i\n", var1, var2);
        var1++;
        var2++;
        printf("Region 1 after incrementation: var1=%i, var2=%i\n", var1, var2);
    }
    printf("After region 1: var1=%i, var2=%i\n\n", var1, var2);
    
    // When set to shared(var1, var2): var1 and var2 are both incremented by the number of threads.
    // When set to private(var1, var2): undefined initial values in the parallel region. Back to initially defined values after parallel region.
    // When set to firstprivate(var1, var2): var1 and var2 have the initialised values inside the parallel region. The threads do not commuicate in this case, as var1 and var2 are private to each thread, but they get incremented by individual threads. However, the changes do not survive outside the parallel region.
    return 0;
}
