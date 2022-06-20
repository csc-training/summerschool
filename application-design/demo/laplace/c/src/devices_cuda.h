#include <cuda.h>
#include <cuda_runtime.h>

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

  __forceinline__ static void devices_memcpyd2d(void* dst, void* src, size_t bytes){
    CUDA_ERR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  }

  // The macros and functions that require C++ compiler (eg, hipcc)
  extern "C++"{

    template <typename LambdaBody> 
    __global__ static void cudaKernel(LambdaBody lambda, const int nx, const int ny)
    {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int i = idx % nx;
      const int j = idx / nx;
  
      if(i < nx && j < ny)
      {
        lambda(i, j);
      }
    }

    #define devices_parallel_for(n1, n2, inc1, inc2, i1, i2, loop_body)          \
    {                                                                            \
      const int blocksize = 64;                                                  \
      const int gridsize = (n1 * n2 - 1 + blocksize) / blocksize;                \
                                                                                 \
      auto lambda_body = [=] __host__ __device__ (int inc1, int inc2) {          \
        inc1 += i1;                                                              \
        inc2 += i2;                                                              \
        loop_body;                                                               \
      };                                                                         \
                                                                                 \
      cudaKernel<<<gridsize, blocksize>>>(lambda_body, n1, n2);                  \
      CUDA_ERR(cudaPeekAtLastError());                                           \
      CUDA_ERR(cudaStreamSynchronize(0));                                        \
      (void)inc1;(void)inc2;                                                     \
    }
  }
}
