#include "common.h"
#include <chrono>

int main() {
    run([](auto n, auto a, auto &x, auto &y) -> auto {
        const auto c_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            saxpy(i, a, x.data(), y.data());
        }

        const auto c_end = std::chrono::high_resolution_clock::now();
        return c_end - c_start;
    });
}
