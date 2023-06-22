#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>

#define HIP_ERR(err) (hip_error(err, __FILE__, __LINE__))
inline static void hip_error(hipError_t err, const char *file, int line) {
	if (err != hipSuccess) {
		printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
		exit(1);
	}
}

#define DEVICE_LAMBDA [=] __host__ __device__

namespace devices
{
  __forceinline__ static void init(int node_rank) {
    int num_devices = 0;
    HIP_ERR(hipGetDeviceCount(&num_devices));
    HIP_ERR(hipSetDevice(node_rank % num_devices));
  }

  __forceinline__ static void finalize(int rank) {
    printf("Rank %d, HIP finalized.\n", rank);
  }

  __forceinline__ static void* allocate(size_t bytes) {
    void* ptr;
    HIP_ERR(hipMallocManaged(&ptr, bytes));
    return ptr;
  }

  __forceinline__ static void free(void* ptr) {
    HIP_ERR(hipFree(ptr));
  }

  __forceinline__ static void memcpyd2d(void* dst, void* src, size_t bytes){
    HIP_ERR(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
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

  template <typename T>
  __forceinline__ static void parallel_for(int loop_size, T loop_body) {
    const int blocksize = 64;
    const int gridsize = (loop_size - 1 + blocksize) / blocksize;
    hipKernel<<<gridsize, blocksize>>>(loop_body, loop_size);
    HIP_ERR(hipStreamSynchronize(0));
  }

  template <typename Lambda, typename T>
  __forceinline__ static void parallel_reduce(const int loop_size, Lambda loop_body, T *sum) {
    const int blocksize = 64;
    const int gridsize = (loop_size - 1 + blocksize) / blocksize;

    T* buf;
    HIP_ERR(hipMalloc(&buf, sizeof(T)));
    HIP_ERR(hipMemcpy(buf, sum, sizeof(T), hipMemcpyHostToDevice));

    auto lambda_outer = 
      DEVICE_LAMBDA(const int i)
      {
        T lsum = 0;
        loop_body(i, lsum);
        atomicAdd(buf, lsum);
      };

    hipKernel<<<gridsize, blocksize>>>(lambda_outer, loop_size);
    HIP_ERR(hipStreamSynchronize(0));

    HIP_ERR(hipMemcpy(sum, buf, sizeof(T), hipMemcpyDeviceToHost));
    HIP_ERR(hipFree(buf));
  }

  template <typename T>
  __host__ __device__ __forceinline__ static void atomic_add(T *array_loc, T value){
    // Define this function depending on whether it runs on GPU or CPU
#if __HIP_DEVICE_COMPILE__
    atomicAdd(array_loc, value);
#else
    *array_loc += value;
#endif
  }

  template <typename T>
  __host__ __device__ static T random_float(unsigned long long seed, unsigned long long seq, int idx, T mean, T stdev){    
    
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
}
