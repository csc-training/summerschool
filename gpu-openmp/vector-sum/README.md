## Work Sharing: vector sum

Calculate the sum of two vectors (`C = A + B`) in parallel using OpenMP.

A skeleton code is provided in `sum(.c|.F90)`. Fill in the computational part
and calculate it in parallel in GPU using OpenMP offloading. Try both `teams`,
`parallel`, `distribute`, `for` / `do` constructs as well as `loop` construct.

Use compiler diagnostics to investigate differences between the two versions:
* C: `cc -fopenmp -fsave-loopmark sum.c`
* Fortran: `ftn -hmsgs -hlist=m -fopenmp sum.F90`

Try also setting `CRAY_ACC_DEBUG` environment variable to emit runtime debug messages for offload activity:
`export CRAY_ACC_DEBUG=3` (value 1, 2, or 3; the higher value increases the verbosity of the messages).

Try to run the code also in CPU only nodes, do you get the same results both
with GPU and CPU?
