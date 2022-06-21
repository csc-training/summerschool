## Asynchronous operations

We have here a code that calculates the Mandelbrot fractal. The
loop over y-dimension is blocked, i.e. there is an outer loop over
chosen number of blocks, and the inner double calculates then each block.

Use OpenMP offloading for accelerating the code, and investigate performance in
each step.

1. Make the `mandelbrot()` a device function, and offload the inner double loop

2. Create a data region around the outer loop over blocks as follows:
     - Upon, entering allocate `image` in the device
     - After each block, copy the data to the host
3. Do the offloading and copying to host asynchronously

Code can be built with the provided [Makefile](makefile).
