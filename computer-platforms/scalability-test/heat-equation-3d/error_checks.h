#pragma once
// This header provides two helper macros for error checking

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define GPU_CHECK(errarg)   __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrMsgFunc(errstr, __FILE__, __LINE__)

inline void __checkErrorFunc(hipError_t errarg, const char* file, 
                             const int line)
{
    if(errarg != hipSuccess) {
        fprintf(stderr, "Error at %s(%i): %s)\n", file, line, hipGetErrorString(errarg));
        exit(EXIT_FAILURE);
    }
}


inline void __checkErrMsgFunc(const char* errstr, const char* file, 
                              const int line)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess) {
        fprintf(stderr, "Error: %s at %s(%i): %s\n", 
                errstr, file, line, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

