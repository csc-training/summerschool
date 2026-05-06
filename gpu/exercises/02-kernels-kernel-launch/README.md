## Exercise: Launching a kernel

Now you know how to compile HIP code. Great!

Next, let's launch a kernel running on the GPU.

The kernel in this exercise prints some values and does assertions.
If you're not familiar with the concept of 'assert', it's a function that takes in a boolean value
and aborts the program if the value is false. On the CPU it's similar to

```cpp
void assert(bool value) {
    if (!value) {
        abort();
    }
}
```
Assertions are useful for catching programmer errors in debug builds: if a value is not something you expect,
it immediately aborts the program.

Ok, enough about assertions.


After compiling the code, run it by giving it two arguments:
- number of blocks
- number of threads

Try running it with the following arguments:
- 1 1: one block, one thread
- 1 2: one block, two threads
- 2 1: two blocks, one thread
- 1 10: one block, ten threads
- 10 1: ten blocks, one thread
- 10 10: ten blocks, ten threads

Finally, try running it with 10 blocks and 1025 threads. What happens? Is the output as expected?
