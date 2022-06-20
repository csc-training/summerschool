#include <cstdio>
#include <cstdlib>
#include <fstream>
#include "constants.h"

using namespace std;

#pragma omp declare target
unsigned char mandelbrot(int Px, int Py) {
  double x0 = xmin + Px * dx;
  double y0 = ymin + Py * dy;
  double x = 0.0;
  double y = 0.0;
  int i;
  for(i= 0; x*x + y*y < 4.0 && i < MAX_ITERS; i++) {
    double xtemp = x*x - y*y + x0;
    y = 2 * x * y + y0;
    x = xtemp;
  }
  return (double)MAX_COLOR * i/MAX_ITERS;
}
#pragma omp end declare target
