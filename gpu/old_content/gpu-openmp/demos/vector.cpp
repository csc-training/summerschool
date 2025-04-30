#include <stdio.h>
#include <stdlib.h>

class Vector {
    public:
    Vector(int n) : len(n) {
        v = new double[len];
        #pragma omp target enter data map(alloc:v[0:len])
    }
    ~Vector() {
        #pragma omp target exit data map(delete:v[0:len])
        delete[] v;
    }
    double *v;
    int len;
};

int main(void)
{
    int N=1000;
    int i;
    Vector vec = Vector(N);
    double *v = vec.v;

    for (i=0; i<N; i++) {
        vec.v[i] = 0.5;
    }
#pragma omp target update to(vec.v[0:N])

#pragma omp target teams
#pragma omp distribute parallel for
    for (i=0; i<N; i++) {
        v[i] = v[i] * v[i];
    }

#pragma omp target update from(vec.v[0:N])
    printf("%f %f %f\n", vec.v[0], vec.v[1], vec.v[2]);

    return 0;
}
