#pragma once

#include "definitions.h"

#include <cassert>
#include <cstddef>

template <typename T> struct Taylor {
    // This computes the Taylor's expansion for e^x,
    // with x in [minimum, maximum]

    T *x = nullptr;
    T *r = nullptr;
    void (*deallocate)(void *);
    const T minimum;
    const T maximum;
    const size_t size;
    const size_t num_iters;

    Taylor(size_t size, void *(*allocate)(size_t), void (*deallocate)(void *),
           T minimum, T maximum, size_t num_iters)
        : x(static_cast<T *>(allocate(sizeof(T) * size))),
          r(static_cast<T *>(allocate(sizeof(T) * size))),
          deallocate(deallocate), minimum(minimum), maximum(maximum),
          size(size), num_iters(num_iters) {
        assert(maximum > minimum);
    }

    ~Taylor() {
        deallocate(static_cast<void *>(x));
        deallocate(static_cast<void *>(r));
    }

    DEVICE void init(size_t i) {
        const T width = maximum - minimum;
        x[i] = minimum + i / size * width;
        r[i] = 0.0;
    }

    DEVICE void compute(size_t i) {
        const T xi = x[i];
        T sum = 1.0;
        T xe = 1.0;
        T factorial = 1.0;

        for (size_t j = 1; j < num_iters; j++) {
            xe *= xi;
            factorial *= static_cast<T>(j);
            sum += xe / factorial;
        }
        r[i] = sum;
    }
};
