#include <iostream>
#include <omp.h>
#include "constants.hpp"
#include "pngwriter.h"

using namespace std;

// TODO make mandelbrot device function
#pragma omp declare target
int kernel(int xi, int yi);
#pragma omp end declare target

int main() {
  
  int *image = new int[width * height];
  int num_blocks = 8; 
  int block_size = (height / num_blocks) * width;
  int y_block_size = height / num_blocks;

  double st = omp_get_wtime();

  // TODO start: offload the calculation according to assignment

#pragma omp target data map(alloc:image[width*height])
  for(int block = 0; block < num_blocks; block++ ) {
    int y_start = block * y_block_size;
    int y_end = y_start + y_block_size;
  
// #pragma omp target loop depend(out:image[y_start]) nowait
#pragma omp target teams distribute parallel for collapse(2) depend(out:image[y_start]) nowait
    for (int y = y_start; y < y_end; y++) {
      for (int x = 0; x < width; x++) {
        int ind = y * width + x;
        image[ind] = kernel(x, y);
      }
    }

#pragma omp target update from(image[block*block_size:block_size]) \
            depend(in:image[y_start]) nowait
 
  }

  #pragma omp taskwait
  // TODO end

  double et = omp_get_wtime();

  cout << "Time: " << (et - st) << " seconds" << endl; 
  save_png(image, width, height, "mandelbrot.png");

  delete [] image;
}
