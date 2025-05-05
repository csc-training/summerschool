#include "axpy.h"
#include "common.h"
#include "host_loop.h"

int main(int, char **) {
    run_and_measure(Loop<Axpy<float>>(), malloc, free, 2.22f);

    return 0;
}
