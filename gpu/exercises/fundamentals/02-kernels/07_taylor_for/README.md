Your job is to fill in the kernel `taylor_for_strided` found in `main.cpp`.
The [lecture slides](https://csc-training.github.io/summerschool/html/gpu/02-kernels.html#/kernel-example-axpy-revisited) have an example of a kernel with a strided for loop, check that for help.

The other code in the file is used for measuring the running times and comparing the three different styles to each other.

Run the code with different number of Taylor iterations, vector sizes and block sizes.
Try values between 1 and 32 for the iterations, values between 1M and 100M for the vector sizes and values up to 1024
for block sizes.

When is the 'consecutive' (or CPU-style) loop especially slow?
Are there situations when it's not that bad?
