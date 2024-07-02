#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <ratio>
#include <vector>

template <typename T> constexpr void saxpy(size_t i, T a, T *x, T *y, T *r) {
    r[i] = a * x[i] + y[i];
}

template <typename T> constexpr void init_x(size_t i, T *x) {
    x[i] = (T)2.3 * sin(i);
}

template <typename T> constexpr void init_y(size_t i, T *y) {
    y[i] = (T)1.1 * cos(i);
}

template <typename T> void init(size_t n, T *x, T *y) {
    for (size_t i = 0; i < n; i++) {
        init_x(i, x);
        init_y(i, y);
    }
}

template <typename Allocate, typename Deallocate, typename Init, typename Func>
void run(Allocate allocate, Deallocate deallocate, Init init, Func func) {
    constexpr std::array ns{1 << 6,  1 << 9,  1 << 12, 1 << 15, 1 << 18,
                            1 << 21, 1 << 24, 1 << 27, 1 << 30};
    constexpr size_t max_n = *std::max_element(ns.begin(), ns.end());
    constexpr size_t num_bytes = sizeof(float) * max_n;

    float *const x = static_cast<float *>(allocate(num_bytes));
    float *const y = static_cast<float *>(allocate(num_bytes));
    float *const r = static_cast<float *>(allocate(num_bytes));
    init(max_n, x, y);

    for (size_t n : ns) {
        constexpr auto n_iter = 20;
        size_t avg = 0;
        for (auto iteration = 0; iteration < n_iter; iteration++) {
            constexpr float a = 3.4f;
            const auto start = std::chrono::high_resolution_clock::now();
            func(n, a, x, y, r);
            const auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::nano> dur = end - start;
            avg += iteration == 0 ? 0 : dur.count();
        }

        std::fprintf(stderr, "%f\n", r[n - 1]);
        std::printf("%ld, %ld\n", n, avg / (n_iter - 1));
    }

    deallocate(x);
    deallocate(y);
    deallocate(r);
}
