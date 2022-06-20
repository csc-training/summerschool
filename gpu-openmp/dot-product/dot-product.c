#include <stdio.h>

#define NX 102400

int main(void)
{
    double vecA[NX], vecB[NX];

    // Initialization of the vectors 
    for (int i = 0; i < NX; i++) {
        vecA[i] = 1.0 / ((double) (NX - i));
        vecB[i] = vecA[i] * vecA[i];
    }

    // TODO start: offload and parallelize the computation

    double res = 0.0;
    for (int i = 0; i < NX; i++) {
        res += vecA[i] * vecB[i];
    }

    // TODO end

    printf("Dot product: %18.16f\n", res);

    return 0;
}
