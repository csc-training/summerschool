## Bonus: Using OpenMP tasks for dynamic parallelization ##

Mandelbrot set of complex numbers can be presented as two dimensional
fractal image. The computation of the pixel values is relatively time
consuming, and the computational cost varies for each pixel. Thus,
simply dividing the two dimensional grid evenly to threads leads into
load imbalance and suboptimal performance.

Files [tasks/mandelbrot.c](tasks/mandelbrot.c) and
[tasks/mandelbrot.F90](tasks/mandelbrot.F90) contain recursive
implementation for calculating the Mandelbrot fractal, which can be
parallelized dynamically with OpenMP tasks. Insert missing pragmas for
parallelizing the code (look for TODOs in the source code), and
investigate the scalability with varying number of threads.
