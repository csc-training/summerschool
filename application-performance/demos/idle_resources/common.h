#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <ratio>
#include <vector>

template <typename T> constexpr void saxpy(size_t i, T a, T *x, T *y) {
    y[i] += a * x[i];
}

template <typename Func> void run(Func f) {
    constexpr std::array ns{1 << 6,  1 << 9,  1 << 12, 1 << 15,
                            1 << 18, 1 << 21, 1 << 24, 1 << 27};
    constexpr auto max_n = *std::max_element(ns.begin(), ns.end());
    std::vector<float> x(max_n);
    std::vector<float> y(max_n);
    constexpr float a = 2.3f;
    constexpr float b = 1.1f;
    constexpr float c = 3.4f;

    for (size_t i = 0; i < max_n; i++) {
        x[i] = a * sin(i);
    }

    for (size_t n : ns) {
        for (size_t i = 0; i < n; i++) {
            y[i] = b * cos(i);
        }

        constexpr auto n_iter = 20;
        size_t avg = 0;
        for (uint32_t iteration = 0; iteration < n_iter; iteration++) {
            const auto default_duration = f(n, c, x, y);
            const std::chrono::duration<double, std::nano> dur =
                default_duration;
            avg += iteration == 0 ? 0 : dur.count();
        }

        printf("%ld, %ld\n", n, avg / (n_iter - 1));
    }
}
