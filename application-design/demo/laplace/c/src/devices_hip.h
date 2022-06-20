#include <hip/hip_runtime.h>

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

__forceinline__ static void devices_memcpyd2d(void* dst, void* src, size_t bytes){
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
  hipKernel<<<gridsize, blocksize>>>(lambda_body, n1, n2);                   \
  HIP_ERR(hipPeekAtLastError());                                             \
  HIP_ERR(hipStreamSynchronize(0));                                          \
  (void)inc1;(void)inc2;                                                     \
}
