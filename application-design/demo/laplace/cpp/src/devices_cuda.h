#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_ERR(err) (cuda_error(err, __FILE__, __LINE__))
inline static void cuda_error(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

#define DEVICE_LAMBDA [=] __host__ __device__

namespace devices
{
  __forceinline__ static void init(int node_rank) {
    int num_devices = 0;
    CUDA_ERR(cudaGetDeviceCount(&num_devices));
    CUDA_ERR(cudaSetDevice(node_rank % num_devices));
  }

  __forceinline__ static void finalize(int rank) {
    printf("Rank %d, CUDA finalized.\n", rank);
  }

  __forceinline__ static void* allocate(size_t bytes) {
    void* ptr;
    CUDA_ERR(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  __forceinline__ static void free(void* ptr) {
    CUDA_ERR(cudaFree(ptr));
  }

  __forceinline__ static void memcpyd2d(void* dst, void* src, size_t bytes){
    CUDA_ERR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  }

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

  template <typename T>
  __forceinline__ static void parallel_for(const int nx, const int ny, T loop_body) {
    const int blocksize = 64;
    const int gridsize = (nx * ny - 1 + blocksize) / blocksize;
    cudaKernel<<<gridsize, blocksize>>>(loop_body, nx, ny);
    CUDA_ERR(cudaPeekAtLastError());
    CUDA_ERR(cudaStreamSynchronize(0));
  }
}
