#include <cstdio>
#include <cmath>

constexpr int n = 840;

int main(int argc, char** argv)
{

  printf("Computing approximation to pi with N=%d\n", n);

  int istart = 1;
  int istop = n;

  double pi = 0.0;
  for (int i=istart; i <= istop; i++) {
    double x = (i - 0.5) / n;
    pi += 1.0 / (1.0 + x*x);
  }

  pi *= 4.0 / n;
  printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);

}


