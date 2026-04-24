
## Exercise

The Code `copy.cpp` will read and write memory array of `4096*4096` float32
entries and various strides (`(1<<n)-1, n=1...21`) (`copy_kernel`).

In this exercise you will inspect how the impact of coalescing memory accesses affects the
performances of a kernel by looking at memory requests using the `rocprof` profiler.
The read request are "strided", while the writes are kept coalesced in this exercise.

### Q1
- What do you think will be the best kernel? How do you expect the execution time to change when 
the stride increases, and why?
 it is obviously the first one, without any stride. In that case, coalescing will be effective and 
can be easily seen by the TCP_TOTAL_READ_sum counter: as soon as we break coalesced access we get 4x 
read requests!



###Q2
- How many L2 read requests are issued?
TCP_TCC_READ_REQ_sum is the counter to look for this one. As you can see, the best one is with the
first kernel, while it increases when we break coalescing.

###Q3
- How many device global memory read requests are issued?
This is a bit trickier, as it looks like that when this one decreases, performance go down, which is counterintuitive
as the opposite should happen. However, this decrease explain why the performance improves a bit when the stride is 
4096. However, the amount of read requests is still higher than the fully coalesced kernel, so even if we
 mitigate the global memory access effect, we do not improve global performances!

###Q4
- The number of Global memory read requests drop when the stride is around 4096. Why?
    *Hint*: Print out the values of `index` for some block in a 16x16 matrix.
    Are some of those indices adjacent within a warp?

some threads require the same index, so they access the same memory pages. this means that they are better cached
and less global memory accesses are needed. However, as stated before, those accesses are more than when the kernel
is properly coalesced, so even if they are faster cause they only go to L2 instead of GMem, the kernel is still slower.

- *Hint*: load and open the `metrics.csv` file with libreoffice or some other
  spreadsheet editor for quick manual analyses
