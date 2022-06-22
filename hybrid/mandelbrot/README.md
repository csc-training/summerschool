## Using OpenMP tasks for dynamic parallelization 

A Mandelbrot set of complex numbers can be presented as a two dimensional
fractal image. The computation of the pixel values is relatively time
consuming, and the computational cost varies for each pixel. Thus,
simply dividing the two dimensional grid evenly to threads leads into
load imbalance and suboptimal performance.

Files [tasks/cpp/mandelbrot.cpp](cpp/mandelbrot.cpp) and
[tasks/fortran/mandelbrot.F90](fortran/mandelbrot.F90) contain a recursive
implementation for calculating the Mandelbrot fractal, which can be
parallelized dynamically with OpenMP tasks. Insert missing directives for
parallelizing the code (look for TODOs in the source code), and
investigate the scalability with varying number of threads.
