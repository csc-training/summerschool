#include <stdio.h>

#define NX 102400

int main(void)
{
    long vecA[NX];
    long sum, psum, sumex;
    int i;

    /* Initialization of the vectors */
    for (i = 0; i < NX; i++) {
        vecA[i] = (long) i+1;
    }

    sum = 0.0;
    /* TODO: Parallelize computation */
    for (i = 0; i < NX; i++) {
        sum += vecA[i];
    }
    printf("Sum: %ld\n",sum);

    return 0;
}
