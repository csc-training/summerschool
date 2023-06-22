#include <hip/hip_runtime.h>

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
  __global__ static void hipKernel(LambdaBody lambda, const int nx, const int ny)
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
    hipLaunchKernelGGL(hipKernel,
                gridsize,
                blocksize,
                0, 0,
                loop_body, nx, ny);
    HIP_ERR(hipPeekAtLastError());
    HIP_ERR(hipStreamSynchronize(0));
  }
}
