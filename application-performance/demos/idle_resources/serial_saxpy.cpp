#include "common.h"
#include <cstddef>

int main() {
    run(malloc, free, init<float>,
        [](auto n, auto a, auto *x, auto *y, auto *r) -> auto {
            for (size_t i = 0; i < n; i++) {
                saxpy(i, a, x, y, r);
            }
        });
}
