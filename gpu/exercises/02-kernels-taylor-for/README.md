## Exercise: Re-use threads in a 1D kernel with a for loop

Finally we did something other than just check for errors!

In the previous exercise we discussed the limits on the number of threads per block and blocks per grid.
The strategy used in the previous exercise was to couple the size of data to the launch parameters of the kernel.
It can be a good strategy for some situations, but other times it's better to reuse the threads and process
multiple values per thread.

In this exercise we'll be computing the Taylor expansion of
$$\vec{y} = e^{\vec{x}} \approx \sum_{n = 0}^{N} \frac{\vec{x}^n}{n!}$$
for a vector of values x and for different values of $N$.

If you're familiar with multithreaded CPU code, you might do something like:
- Split the data of size $M$ equally between $T$ threads (assuming $M$ divides by $T$)
- Each thread processes $M/T$ consecutive values:
    - thread 0 processes `data[0], data[1], ..., data[M/T - 1]`
    - thread 1 processes `data[M/T], data[M/T + 1], ..., data[2 * M/T - 1]`
    - and so on

As a kernel, it would look like this:
```cpp
__global__ void taylor_for_consecutive(float *x, float *y, size_t num_values,
                                       size_t num_iters) {
    // Global thread id, i.e. over the entire grid
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // How many threads in total in the entire grid
    const size_t num_threads = blockDim.x * gridDim.x;

    // How many elements per thread
    const size_t num_per_thread = num_values / num_threads;

    // Process num_per_thread consecutive elements
    for (size_t i = 0; i < num_per_thread; i++) {
        // tid      elems
        //   0      [0, num_per_thread - 1]
        //   1      [num_per_thread, 2 * num_per_thread - 1]
        //   2      [2 * num_per_thread, 3 * num_per_thread - 1]
        //   and so on...
        const size_t j = tid * num_per_thread + i;
        y[j] = taylor(x[j], num_iters);
    }

    // How many are left over
    const size_t left_over = num_values - num_per_thread * num_threads;

    // The first threads will process one more, so the left over values
    // are also processed
    if (tid < left_over) {
        // tid      elem
        //   0      num_per_thread * num_threads
        //   1      num_per_thread * num_threads + 1
        //   2      num_per_thread * num_threads + 2
        //   and so on...
        const size_t j = num_per_thread * num_threads + tid;
        y[j] = taylor(x[j], num_iters);
    }
}
```

In CPU code this is a valid strategy, but for GPUs it can be tens of times slower
than just launching a large enough grid such that each thread processes a single element.
The reason is related to how GPUs access memory. We will learn much more about that later.

A significantly more GPU friendly way of doing the same is to use a 'strided loop':
- Consecutive threads process consecutive elements
- Then all threads jump forward a constant amount (usually the total number of threads participating in the loop)
- Repeat

This way the threads of a warp always access memory close to each other: the memory needs of the warp can be
served with fewer memory accesses.

Your job is to fill in the kernel `taylor_for_strided` found in `main.cpp`.
The [lecture slides](https://csc-training.github.io/summerschool/html/gpu/02-kernels.html#/kernel-example-axpy-revisited) have an example of a kernel with a strided for loop, check that for help.

The other code in the file is used for measuring the running times and comparing the three different styles to each other.

Run the code with different number of Taylor iterations, vector sizes and block sizes.
Try values between 1 and 32 for the iterations, values between 1M and 100M for the vector sizes and values up to 1024
for block sizes.

When is the 'consecutive' (or CPU-style) loop especially slow?
Are there situations when it's not that bad?
