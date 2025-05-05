#include "axpy.h"
#include "common.h"

int main(int, char **) {
    run_and_measure<Axpy<float>>(2.22f);

    return 0;
}
