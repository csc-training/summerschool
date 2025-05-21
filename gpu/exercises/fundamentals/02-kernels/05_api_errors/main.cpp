#include <hip/hip_runtime.h>

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file,
                              int32_t line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    // TODO:
    // Wrap any API calls with the HIP_ERRCHK macro
    // Then, compile and run the program and fix any errors you encounter.
    // Repeat, until you've fixed all the errors.
    // You can consult the HIP API documentation for the correct way to call
    // each function:
    // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules.html#modules-reference

    // Hint:
    // Instead of doing
    // result = hipApiCall(args);
    // do
    // HIP_ERRCHK(hipApiCall(args));

    int count = 0;
    auto result = hipGetDeviceCount(&count);

    int device = 0;
    result = hipGetDevice(&device);

    result = hipSetDevice(2);

    void *ptr = nullptr;
    result = hipMalloc(nullptr, 999999999);

    result = hipMemset(ptr, 0, 8);

    result = hipFree(ptr);

    return 0;
}
