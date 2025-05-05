#pragma once

// clang-format off

/* This functor
* runs on the CPU
* can be compiled with -fopenmp
* calls a function f
* from an object of type T
* n times
* with iteration index i as the argument to f
*/

template <typename T>
struct Loop {
    void operator()(size_t n, T &t, void (T::*f)(size_t)) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            (t.*f)(i);
        }
    }
};
// clang-format on
