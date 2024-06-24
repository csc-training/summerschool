# Heat equation solver with HIP

Create a parallel version of a heat equation solver using HIP.

Starting from a [serial heat equation solver](serial) or the heat equation solvers from earlier exercises,
port the code to GPUs using HIP. Main computational routine is the time
evolution loop in the `core.cpp` file.

Alternatively, you may start from a [CUDA+MPI version](cuda) and hipify the
code to jump start the work.

