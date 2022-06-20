#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>

// All macros and functions require a C++ compiler (HIP API does not support C)
#define HIP_ERR(err) (hip_error(err, __FILE__, __LINE__))
inline static void hip_error(hipError_t err, const char *file, int line) {
	if (err != hipSuccess) {
		printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
		exit(1);
	}
}

  __forceinline__ static void devices_init(int node_rank) {
    int num_devices = 0;
    HIP_ERR(hipGetDeviceCount(&num_devices));
    HIP_ERR(hipSetDevice(node_rank % num_devices));
  }

  __forceinline__ static void devices_finalize(int rank) {
    printf("Rank %d, HIP finalized.\n", rank);
  }

  __forceinline__ static void* devices_allocate(size_t bytes) {
    void* ptr;
    HIP_ERR(hipMallocManaged(&ptr, bytes));
    return ptr;
  }

  __forceinline__ static void devices_free(void* ptr) {
    HIP_ERR(hipFree(ptr));
  }

  __forceinline__ static void devices_memcpy_d2d(void* dst, void* src, size_t bytes){
    HIP_ERR(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
  }

  template <typename T>
  __host__ __device__ __forceinline__ static void devices_atomic_add(T *array_loc, T value){
    // Define this function depending on whether it runs on GPU or CPU
#if __HIP_DEVICE_COMPILE__
    atomicAdd(array_loc, value);
#else
    *array_loc += value;
#endif
  }

  template <typename T>
  __host__ __device__ static T devices_random_float(unsigned long long seed, unsigned long long seq, int idx, T mean, T stdev){    
    
    T var = 0;
#if __HIP_DEVICE_COMPILE__
    hiprandStatePhilox4_32_10_t state;

    // hiprand_init() reproduces the same random number with the same seed and seq
    hiprand_init(seed, seq, 0, &state);

    // hiprand_normal() gives a random float from a normal distribution with mean = 0 and stdev = 1
    var = stdev * hiprand_normal(&state) + mean;
#endif
    return var;
  }

  template <typename LambdaBody> 
  __global__ static void hipKernel(LambdaBody lambda, const int loop_size)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < loop_size)
    {
      lambda(i);
    }
  }

  #define devices_parallel_for(loop_size, inc, loop_body)           \
  {                                                                 \
    const int blocksize = 64;                                       \
    const int gridsize = (loop_size - 1 + blocksize) / blocksize;   \
    auto lambda_body = [=] __host__ __device__ (int inc) loop_body; \
    hipKernel<<<gridsize, blocksize>>>(lambda_body, loop_size);     \
    HIP_ERR(hipStreamSynchronize(0));                               \
    (void)inc;                                                      \
  }
