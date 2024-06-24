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

Compile the program with the provided Makefiles. On **Lumi**, simply type
```
make
```
and on **Mahti**, use instead
```
make MAHTI=1
```

NOTE! It is important to note that the OpenMP array index range notation is different between C/CPP and Fortran, ie, the following are equivalent:

```
#pragma omp target update from(array[5:3])
```
```
!$omp target update from(array(5:7))
```
In the former case (C/CPP), the integer `3` denotes the number of elements, whereas in the latter (Fortran), the integer `7` denotes the last element. Ie, both notations refer to the elements `array[5]`, `array[6]`, and `array[7]`.
