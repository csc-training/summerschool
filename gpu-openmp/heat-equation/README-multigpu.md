## Heat equation with multiple GPUs

Your task is now to combine the MPI parallelization as
[described for CPU-only code](../heat-cpu/code-description.mdn) with the
OpenMP offloading.

You can base your work on [hybrid MPI + OpenMP code](../heat-cpu/) and the
previous work on offloading with single GPU.

In order to achieve a working multi-GPU code, you should:

1. Assign MPI tasks to devices
2. Alternatively, either
    - a) Copy the data between host and device before and after the MPI
      communication, or
    - b) Pass device pointer to MPI routines
3. Use OpenMP offload constructs in the `evolve` routine
