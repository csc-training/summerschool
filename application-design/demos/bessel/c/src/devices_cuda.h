#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C"{
    
  // The macros and functions that can be compiled with a C compiler
  #define CUDA_ERR(err) (cuda_error(err, __FILE__, __LINE__))
  inline static void cuda_error(cudaError_t err, const char *file, int line) {
  	if (err != cudaSuccess) {
  		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
  		exit(1);
  	}
  }
  
  __forceinline__ static void devices_init(int node_rank) {
    int num_devices = 0;
    CUDA_ERR(cudaGetDeviceCount(&num_devices));
    CUDA_ERR(cudaSetDevice(node_rank % num_devices));
  }

  __forceinline__ static void devices_finalize(int rank) {
    printf("Rank %d, CUDA finalized.\n", rank);
  }

  __forceinline__ static void* devices_allocate(size_t bytes) {
    void* ptr;
    CUDA_ERR(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  __forceinline__ static void devices_free(void* ptr) {
    CUDA_ERR(cudaFree(ptr));
  }

  __forceinline__ static void devices_memcpy_d2d(void* dst, void* src, size_t bytes){
    CUDA_ERR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  }

  // The macros and functions that require C++ compiler (eg, hipcc)
  extern "C++"{

    template <typename T>
    __host__ __device__ __forceinline__ static void devices_atomic_add(T *array_loc, T value){
      // Define this function depending on whether it runs on GPU or CPU
  #ifdef __CUDA_ARCH__
      atomicAdd(array_loc, value);
  #else
      *array_loc += value;
  #endif
    }
  
    template <typename T>
    __host__ __device__ static T devices_random_float(unsigned long long seed, unsigned long long seq, int idx, T mean, T stdev){    
      
      T var = 0;
  #ifdef __CUDA_ARCH__
      curandStatePhilox4_32_10_t state;
  
      // curand_init() reproduces the same random number with the same seed and seq
      curand_init(seed, seq, 0, &state);
  
      // curand_normal() gives a random float from a normal distribution with mean = 0 and stdev = 1
      var = stdev * curand_normal(&state) + mean;
  #endif
      return var;
    }

    template <typename LambdaBody> 
    __global__ static void cudaKernel(LambdaBody lambda, const int loop_size)
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
      cudaKernel<<<gridsize, blocksize>>>(lambda_body, loop_size);    \
      CUDA_ERR(cudaStreamSynchronize(0));                             \
      (void)inc;                                                      \
    }
  }
}
  