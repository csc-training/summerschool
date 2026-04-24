#include <hip/hip_runtime.h>

#include <cstdlib>
#include <vector>
#include <utility>

#include <chrono>
#include <iostream>

#define LOG2SIZE 12
const static int width = 1<<LOG2SIZE;
const static int height = 1<<LOG2SIZE;

const static int tile_dim_x = 16;
const static int tile_dim_y = 16;



//templated version of the loop to ease the reading in the profile.
template< int stride>
__global__ void copy_kernel(float *in, float *out, size_t width, size_t height) {
  size_t x_index = blockIdx.x * tile_dim_x + threadIdx.x;
  size_t y_index = blockIdx.y * tile_dim_y + threadIdx.y;

  size_t index_in = (y_index * width + stride*x_index) % (width*height);

  // Logical output position (same for all kernels)
  size_t index_out = y_index * width + x_index;

  out[index_out] = in[index_in];
}

template< int stride>
void launch_kernel(dim3 grid, dim3 block, float *in, float *out, size_t width, size_t height)
{
    printf("launch kern: %d, \n",stride);
    constexpr int kern_val = 1<<stride;

        // Start timer (CPU)
    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(copy_kernel<kern_val>, grid,block, 0, 0, in, out, width, height) ;
    hipDeviceSynchronize();    // Stop timer (CPU)
    auto stop = std::chrono::high_resolution_clock::now();

    // Convert to microseconds (or ms)
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    std::cout << "Kernel stride=" << kern_val
              << " execution time: " << us << " us" << std::endl;
}


template<int... Xs>
void call_all(std::integer_sequence<int, Xs...>,dim3 block, dim3 grid, float *in, float *out, size_t width, size_t height ) {
    (launch_kernel<Xs>(block, grid, in, out, width, height), ...);   // fold expression
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

  hipMalloc((void **)&d_in, (width * height) * sizeof(float));
  hipMalloc((void **)&d_out, (width * height) * sizeof(float));

  hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float),
            hipMemcpyHostToDevice);

  printf("Setup complete. Launching kernel \n");
  int block_x = width / tile_dim_x;
  int block_y = height / tile_dim_y;
  

  // Create events

   printf("Warm up the gpu!\n"); 
   for(int i=1;i<=10;i++){ 
     hipLaunchKernelGGL(copy_kernel<1>, dim3(block_x, block_y), 
                       dim3(tile_dim_x, tile_dim_y), 0, 0, d_in, d_out, width, 
                       height);} 


    hipDeviceSynchronize();    // Stop timer (CPU)

  call_all(std::make_integer_sequence<int, 20>{}, dim3(block_x, block_y),dim3(tile_dim_x, tile_dim_y), d_in, d_out, width,height); // generates foo<0>()..foo<4>()



//  #pragma unroll
//  for(int i=1;i<=21;i++){
//    hipLaunchKernelGGL(copy_kernel<(1<<i)-1>, dim3(block_x, block_y),dim3(tile_dim_x, tile_dim_y), 0, 0, d_in, d_out, width,height) ;}
//  

  hipDeviceSynchronize();

  printf("Done!\n");
  hipMemcpy(matrix_out.data(), d_out, width * height * sizeof(float),
            hipMemcpyDeviceToHost);


  return 0;
}
