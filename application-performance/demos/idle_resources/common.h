#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <ratio>

template <typename T, template <typename S> typename Loop, typename... Args>
void run_and_measure(Loop<T> loop, void *(*allocate)(size_t),
                     void (*deallocate)(void *), Args... args) {
    constexpr std::array ns{1 << 6,  1 << 9,  1 << 12, 1 << 15, 1 << 18,
                            1 << 21, 1 << 24, 1 << 27, 1 << 30};
    constexpr size_t max_n = *std::max_element(ns.begin(), ns.end());
    T t(max_n, allocate, deallocate, args...);
    loop(max_n, t, &T::init);

    for (size_t n : ns) {
        constexpr auto n_iter = 20;
        size_t avg = 0;
        for (auto iteration = 0; iteration < n_iter; iteration++) {
            const auto start = std::chrono::high_resolution_clock::now();
            loop(n, t, &T::compute);
            const auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::nano> dur = end - start;
            avg += iteration == 0 ? 0 : dur.count();
        }

        std::printf("%ld, %ld\n", n, avg / (n_iter - 1));
    }
}
