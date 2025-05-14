# Kernel: copy2d

Write a device kernel that performs the double precision BLAS operation
**dcopy**, i.e. `y = x` using GPU threads in a 2D grid.

- Assume that the vectors `x` and `y` are used to store a 400x600 matrix (in row-major format)
- Initialise the matrix `x` with some values on the CPU
- Allocate memory for `x` and `y` on the device
- Copy the host `x` to the device `x`
- Perform the operation on the device using a 2D kernel
    - Hint: it's easier to implement the kernel first without looping. i.e. such that each thread copies a single element from input to output.
    - Once you get that working, you can implement a double for loop in the kernel so it works for arbitrary sized 2D arrays
- Copy device `y` to host `y`
- Compare host `x` to host `y`

Are the values of `x` and `y` equal?

You may start from a skeleton code provided in [main.cpp](main.cpp).
