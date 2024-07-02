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

  int* dummy_ptr;

  // TODO start: offload the calculation according to assignment

  // Create a data region around the outer loop over blocks 
  // Upon, entering allocate image in the device
  // With dynamically allocated arrays, one needs to specify the number of elements to be copied.
  #pragma omp target data map(alloc:image[0:width*height])  
  {
    for(int block = 0; block < num_blocks; block++ ) {
      int y_start = block * y_block_size;
      int y_end = y_start + y_block_size;
      
      // Use OpenMP offloading 
      #pragma omp target teams distribute parallel for collapse(2) depend(out:dummy_ptr[y_start]) nowait  // Implicit barrier at the end of #pragma omp parallel. Override with nowait to enable asynchronous offloading
      //#pragma omp target teams loop nowait
      for (int y = y_start; y < y_end; y++) {
        for (int x = 0; x < width; x++) {
          int ind = y * width + x;
          image[ind] = kernel(x, y);
        }
      }
      // Copy the data to the host
      #pragma omp target update from(image[block*block_size : block_size]) depend(in:dummy_ptr[y_start]) nowait  // Implicit barrier at the end of #pragma omp update. 
                                                      // Override with nowait to enable asynchronous offloading.
                                                      // Enables the target update construct to execute 
                                                      // asynchronously with respect to the encountering thread. 
                                                      // By default, the encountering thread must wait for the 
                                                      // completion of the construct.

      // For the out and inout task-dependence-types, if the storage location of at least one of the list
      // items matches the storage location of a list item appearing in a depend clause with an in, out,
      // inout, mutexinoutset, or inoutset task-dependence-type on a construct from which a sibling task 
      // was previously generated, then the generated task will be a dependent task of that sibling task.
      // In this case, we use dummy_ptr[y_start] to create matching storage locations, but which changes 
      // between every iteration between blocks. This way, we create dependencies inside the same block 
      // iteration, but not between different iterations.

                                                      
      
    
    }
  }  // end data region
  // TODO end
  
  #pragma omp taskwait  // Synchronisation point
  double et = omp_get_wtime();

  cout << "Time: " << (et - st) << " seconds" << endl;
  save_png(image, height, width, "mandelbrot.png");

  delete [] image;
}
