#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <ratio>

#include "definitions.h"

/* 
 * The Loop functor
 * calls a function f
 * from an object of type T
 * n times
 * with iteration index i as the argument to f
 *
 * If the code is compiled on the device,
 * it'll be called from a kernel running on the device.
 * Otherwise, it'll be called from a loop that can be
 * parallellized with OpenMP.
 * */

#if defined(RUN_ON_THE_DEVICE)
#include <hip/hip_runtime.h>

template <typename T, typename Lambda>
__global__ void kernel(size_t n, T t, Lambda lambda) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        lambda(t, i);
    }
}

#endif

template <typename T, typename Lambda>
void loop(size_t n, T &t, Lambda lambda) {
#if defined(RUN_ON_THE_DEVICE)
    static constexpr dim3 blocks(1024);
    static constexpr dim3 threads(1024);
    kernel<<<threads, blocks>>>(n, t, lambda);
    [[maybe_unused]] const auto result = hipDeviceSynchronize();
#else
    // clang-format off
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        lambda(t, i);
    }
    // clang-format on
#endif
}

inline void *allocate(size_t bytes) {
    void *p = nullptr;

#if defined(RUN_ON_THE_DEVICE)
    [[maybe_unused]] const auto result = hipMalloc(&p, bytes);
#else
    p = std::malloc(bytes);
#endif

    return p;
}

inline void deallocate(void *p) {
#if defined(RUN_ON_THE_DEVICE)
    [[maybe_unused]] const auto result = hipFree(p);
#else
    std::free(p);
#endif
}

template <typename T, typename... Args> void run_and_measure(Args... args) {
    constexpr std::array ns{1 << 6,  1 << 9,  1 << 12, 1 << 15, 1 << 18,
                            1 << 21, 1 << 24, 1 << 27, 1 << 30};
    constexpr size_t max_n = *std::max_element(ns.begin(), ns.end());

    T t(max_n, allocate, deallocate, args...);
    loop(max_n, t, [](T &t, size_t i) { T::init(t, i); });

    for (size_t n : ns) {
        constexpr auto n_iter = 20;
        size_t avg = 0;
        for (auto iteration = 0; iteration < n_iter; iteration++) {
            const auto start = std::chrono::high_resolution_clock::now();
            loop(max_n, t, [](T &t, size_t i) { T::compute(t, i); });
            const auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::nano> dur = end - start;
            avg += iteration == 0 ? 0 : dur.count();
        }

        std::printf("%ld, %ld\n", n, avg / (n_iter - 1));
    }
}
