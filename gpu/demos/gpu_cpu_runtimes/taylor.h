#pragma once

#include "definitions.h"

#include <cassert>
#include <cstddef>

template <typename T> struct Taylor {
    // This computes the Taylor's expansion for e^x,
    // with x in [minimum, maximum]

    T *x = nullptr;
    T *y = nullptr;
    void (*deallocate)(void *);
    const T minimum;
    const T maximum;
    const size_t size;
    const size_t N;

    Taylor(size_t size, void *(*allocate)(size_t), void (*deallocate)(void *),
           T minimum, T maximum, size_t N)
        : x(static_cast<T *>(allocate(sizeof(T) * size))),
          y(static_cast<T *>(allocate(sizeof(T) * size))),
          deallocate(deallocate), minimum(minimum), maximum(maximum),
          size(size), N(N) {
        assert(maximum > minimum);
    }

    ~Taylor() {
        deallocate(static_cast<void *>(x));
        deallocate(static_cast<void *>(y));
    }

    DEVICE void init(size_t i) {
        const T width = maximum - minimum;
        x[i] = minimum + i * width / size;
        y[i] = 0.0;
    }

    DEVICE void compute(size_t i) {
        const T xi = x[i];
        T sum = 0.0;
        T xn = 1.0 / xi;
        T factorial = 1.0;

        for (size_t n = 0; n <= N; n++) {
            xn *= xi;
            factorial *= std::max(static_cast<T>(n), static_cast<T>(1.0));
            sum += xn / factorial;
        }
        y[i] = sum;
    }
};
