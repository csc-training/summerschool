#include "common.h"

// #include "axpy.h"
#include "taylor.h"

int main(int, char **) {
    // run_and_measure<Axpy<float>>(2.216f);
    run_and_measure<Taylor<float>>(-10.0f, 10.0);

    return 0;
}
