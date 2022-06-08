#include <hip/hip_runtime.h>

#include <cstdlib>
#include <vector>

const static int width = 4096;
const static int height = 4096;
const static int tile_dim = 32;

__global__ void transpose_lds_kernel(float *in, float *out, int width,
                                     int height) {
  __shared__ float tile[tile_dim][tile_dim];

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

  hipLaunchKernelGGL(transpose_lds_kernel, dim3(block_x, block_y),
                      dim3(tile_dim, tile_dim), 0, 0, d_in, d_out, width,
                      height);

   hipDeviceSynchronize();

   printf("Kernel execution complete \n");
 
   hipMemcpy(matrix_out.data(), d_out, width * height * sizeof(float),
            hipMemcpyDeviceToHost);


  return 0;
}

