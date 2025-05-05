#include "definitions.h"

#include <cmath>
#include <cstddef>

template <typename T> struct Axpy {
    T *x = nullptr;
    T *y = nullptr;
    T *r = nullptr;
    void (*deallocate)(void *);
    T a;

    Axpy(size_t size, void *(*allocate)(size_t), void (*deallocate)(void *),
         T a)
        : x(static_cast<T *>(allocate(sizeof(T) * size))),
          y(static_cast<T *>(allocate(sizeof(T) * size))),
          r(static_cast<T *>(allocate(sizeof(T) * size))),
          deallocate(deallocate), a(a) {}

    ~Axpy() {
        deallocate(static_cast<void *>(x));
        deallocate(static_cast<void *>(y));
        deallocate(static_cast<void *>(r));
    }

    DEVICE void init(size_t i) {
        x[i] = sin(static_cast<T>(i));
        y[i] = cos(static_cast<T>(i));
        r[i] = 0.0;
    }

    DEVICE void compute(size_t i) { r[i] = a * x[i] + y[i]; }
};
