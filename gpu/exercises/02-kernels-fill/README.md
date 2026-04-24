## Exercise: Kernel for filling a 1D array with a value

So far we've mostly concerned ourselves with checking for errors. Can we finally do something more interesting?
Yes!

Many times in simulation loops we want to reset some arrays to a specific value before other computation.
So let's do that! You'll implement a kernel that fills an array with a single value.

To do that we need a few things:
- an array of values on the GPU memory
- a value which the elements of the array will be set to
- a kernel that does that

We also need to launch the kernel with enough threads to go over the entire array.
But we've learned that the maximum number of threads per block is 1024. Yes, indeed,
but the limit on the maximum number of blocks per grid is much higher! So we should be able to
easily launch enough *threads per grid* to fill the entire array.

Your job is to fill an array with a constant value on the GPU using a kernel.

You may begin with the [provided skeleton](main.cpp) and fill in the TODOs.

You can consult the earlier exercises where we launched kernels and the lecture slides.
The slides will be helpful with allocating memory, copying it and computing the indices in the kernel.
