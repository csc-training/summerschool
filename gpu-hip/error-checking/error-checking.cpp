#include <hip/hip_runtime.h>
#include <stdio.h>

/* HIP error handling macro */
// __FILE__ will be substituted by the compiler with the filename of the file being compiler. __LINE__ will be substituted with
// the line number where an error occurs.
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ )) 
static inline void hip_errchk(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    int count, device;


    HIP_ERRCHK( hipGetDeviceCount(&count) );
    HIP_ERRCHK( hipSetDevice(count) );
    HIP_ERRCHK( hipGetDevice(&device) );

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    

    return 0;
}
