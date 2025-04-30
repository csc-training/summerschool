# Error checking with HIP

Your task is to find a bug by wrapping all HIP calls found in `error-checking.cpp` into an error checking macro. It is a good practice to always check for errors when using the HIP API. This can make debugging significantly easier with a large code base. The error checking macro is given by:

```
/* HIP error handling macro */
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
```