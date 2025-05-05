#include "axpy.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::printf("Give one floating point argument: a\n");
        return 1;
    }

    run_and_measure<Axpy<float>>(std::atof(argv[1]));

    return 0;
}
