#include <cstdio>
#include <omp.h>
#include <vector>

#define NX 4096
#define NY 4096

int main()
{

  constexpr int niters = 50;

  int nx = NX;
  int ny = NY;

  // Use static to ensure allocation from heap; allocation from stack can segfault
  static double A[NX][NY];
  static double L[NX][NY];

  double dx = 1.0 / double(nx);
  double dy = 1.0 / double(ny);

  // Initialize arrays
  double x = 0.0;
  double y;
  for (int i = 0; i < nx; i++)
    {
      y = 0.0; 
      for (int j = 0; j < ny; j++)
        {
          A[i][j] = x*x + y*y;
          L[i][j] = 0.0;
          y += dy;
        }
      x += dx;
    }

  double t0 = omp_get_wtime();
  // Compute Laplacian
  #pragma nounroll
  for (int iter = 0; iter < niters; iter++)
    for (int j = 1; j < ny-1; j++)
      for (int i = 1; i < nx-1; i++)
        L[i][j] = (A[i-1][j] - 2.0*A[i][j] + A[i+1][j]) / (dx*dx) +
	           (A[i][j-1] - 2.0*A[i][j] + A[i][j+1]) / (dy*dy);

  double t1 = omp_get_wtime();

  // Check the result
  double meanL = 0.0;
  for (int i = 1; i < nx-1; i++)
    for (int j = 1; j < ny-1; j++)
      meanL += L[i][j];

  meanL /= ((nx - 1) * (ny - 1));

  printf("Numerical solution %6.4f\n", meanL);
  printf("Analytical solution %6.4f\n", 4.0);

  printf("Time %7.4f\n", t1 - t0);

  return 0;
}
