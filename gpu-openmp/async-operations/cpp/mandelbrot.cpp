#include <array>
#include <vector>
#include <iostream>
#include <omp.h>
#include "constants.hpp"
#include "pngwriter.h"

using namespace std;

// TODO make mandelbrot device function
int kernel(int xi, int yi);

int main() {

  std::vector<int> image(width * height);
  int num_blocks = 8;
  int block_size = (height / num_blocks) * width;
  int y_block_size = height / num_blocks;

  double st = omp_get_wtime();

  // TODO start: offload the calculation according to assignment

  for(int block = 0; block < num_blocks; block++ ) {
    int y_start = block * y_block_size;
    int y_end = y_start + y_block_size;

    for (int y = y_start; y < y_end; y++) {
      for (int x = 0; x < width; x++) {
        int ind = y * width + x;
        image[ind] = kernel(x, y);
      }
    }

  }

  // TODO end

  double et = omp_get_wtime();

  cout << "Time: " << (et - st) << " seconds" << endl;
  int *image_ptr = image.data();
  save_png(image_ptr, width, height, "mandelbrot.png");
}
