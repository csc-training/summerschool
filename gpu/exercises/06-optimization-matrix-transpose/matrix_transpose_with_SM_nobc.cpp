#include <hip/hip_runtime.h>

#include <cstdlib>
#include <vector>

const static int width = 4096;
const static int height = 4096;
const static int tile_dim = 16;

__global__ void transpose_SM_nobc_kernel(float *in, float *out, int width,
                                     int height) {
  __shared__ float tile[tile_dim][tile_dim+1];

  int x_tile_index = blockIdx.x * tile_dim;
  int y_tile_index = blockIdx.y * tile_dim;

  int in_index =
      (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
  int out_index =
      (x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

  tile[threadIdx.y][threadIdx.x] = in[in_index];

  __syncthreads();

  out[out_index] = tile[threadIdx.x][threadIdx.y];
}


int main() {
  std::vector<float> matrix_in;
  std::vector<float> matrix_out;

  matrix_in.resize(width * height);
  matrix_out.resize(width * height);

  for (int i = 0; i < width * height; i++) {
    matrix_in[i] = (float)rand() / (float)RAND_MAX;
  }



  float *d_in;
  float *d_out;

  hipMalloc((void **)&d_in, width * height * sizeof(float));
  hipMalloc((void **)&d_out, width * height * sizeof(float));

  hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float),
            hipMemcpyHostToDevice);

  printf("Setup complete. Launching kernel \n");
  int block_x = width / tile_dim;
  int block_y = height / tile_dim;

  // Create events
  hipEvent_t start_kernel_event;
  hipEventCreate(&start_kernel_event);
  hipEvent_t end_kernel_event;
  hipEventCreate(&end_kernel_event);

  printf("Warm up the gpu!\n");


  for(int i=1;i<=10;i++){
    hipLaunchKernelGGL(transpose_SM_nobc_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height);}


  hipEventRecord(start_kernel_event, 0);

   for(int i=1;i<=10;i++){
    hipLaunchKernelGGL(transpose_SM_nobc_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height);}
  
  
  hipEventRecord(end_kernel_event, 0);
  hipEventSynchronize(end_kernel_event);

  float time_kernel;
  hipEventElapsedTime(&time_kernel, start_kernel_event, end_kernel_event);

   printf("Kernel execution complete \n");
  printf("Event timings:\n");
  printf("  %.6f ms - shared memory with no bank conflicts \n  Bandwidth %.6f GB/s\n", time_kernel/10, 2.0*10000*(((double)(width)*(double)height)*sizeof(float))/(time_kernel*1024*1024*1024));
 
   hipMemcpy(matrix_out.data(), d_out, width * height * sizeof(float),
            hipMemcpyDeviceToHost);
  
  hipEventDestroy(start_kernel_event);
  hipEventDestroy(end_kernel_event);

  return 0;
}

