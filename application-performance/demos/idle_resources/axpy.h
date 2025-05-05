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

    DEVICE static void init(Axpy<T> &axpy, size_t i) {
        axpy.x[i] = sin(static_cast<T>(i));
        axpy.y[i] = cos(static_cast<T>(i));
        axpy.r[i] = 0.0;
    }

    DEVICE static void compute(Axpy<T> &axpy, size_t i) {
        axpy.r[i] = axpy.a * axpy.x[i] + axpy.y[i];
    }
};
