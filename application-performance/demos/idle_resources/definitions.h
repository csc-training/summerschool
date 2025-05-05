#pragma once

// clang-format off
#if defined(__NVCC__) || \
    (defined(__clang__) && defined(__CUDA__)) || \
    defined(__HIPCC__)
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST
    #define DEVICE
#endif
// clang-format on
