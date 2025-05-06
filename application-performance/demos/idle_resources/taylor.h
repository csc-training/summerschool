#pragma once

#include "definitions.h"

#include <cassert>
#include <cstddef>

template <typename T> struct Taylor {
    T *x = nullptr;
    T *r = nullptr;
    void (*deallocate)(void *);
    const T minimum;
    const T maximum;
    const size_t size;

    Taylor(size_t size, void *(*allocate)(size_t), void (*deallocate)(void *),
           T minimum, T maximum)
        : x(static_cast<T *>(allocate(sizeof(T) * size))),
          r(static_cast<T *>(allocate(sizeof(T) * size))),
          deallocate(deallocate), minimum(minimum), maximum(maximum),
          size(size) {
        assert(maximum > minimum);
    }

    ~Taylor() {
        deallocate(static_cast<void *>(x));
        deallocate(static_cast<void *>(r));
    }

    DEVICE void init(size_t i) {
        const T width = maximum - minimum;
        x[i] = i / size * width;
        r[i] = 0.0;
    }

    DEVICE void compute(size_t i) {
        const T xi = x[i];
        // clang-format off
        r[i] = 1.0 +
               xi +
               xi * xi / 2.0 +
               xi * xi * xi / 6.0 +
               xi * xi * xi * xi / 24.0 +
               xi * xi * xi * xi * xi / 120.0 +
               xi * xi * xi * xi * xi * xi / 720.0;
        // clang-format on
    }
};
