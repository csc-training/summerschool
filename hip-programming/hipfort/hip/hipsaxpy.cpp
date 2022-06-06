#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void saxpy(float *y, float *x, float a, int n)
{
    size_t i = blockDim.x * blockIdx.x  + threadIdx.x;
    if (i < n) y[i] = y[i] + a*x[i];
}


extern "C"
{
  void launch(float **dout, float **da, float db, int N)
  {

     dim3 tBlock(256,1,1);
     dim3 grid(ceil((float)N/tBlock.x),1,1);
    
    hipLaunchKernelGGL((saxpy), grid, tBlock, 0, 0, *dout, *da, db, N);
  }
}
