#include "axpy.h"
#include "common.h"

int main(int argc, char **argv) {
    run_and_measure<Axpy<float>>(2.216f);

    return 0;
}
