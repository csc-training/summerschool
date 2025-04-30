## Heat equation with multiple GPUs

Your task is now to combine the MPI parallelization as
[described for the CPU-only code](heat-cpu/code-description.md) with the
OpenMP offloading.

You can base your work on the [hybrid MPI + OpenMP code](heat-cpu/) and
the previous work on offloading with a single GPU.

In order to achieve a working multi-GPU code, you should:

1. Assign MPI tasks to devices
2. Alternatively, either
    - a) Copy the data between host and device before and after the MPI
      communication, or
    - b) Pass device pointer to MPI routines
3. Use OpenMP offload constructs in the `evolve` routine

Note: If opt for 2b then 
- Set environment variable `export MPICH_GPU_SUPPORT_ENABLED=1` prior to compiling abd running.
- Run the program with `srun -n <ntasks> --gpus <ntasks> <rest of the flags> ./heat_offload`.
