# Laplace multi-architecture example

This example uses MPI to create up to 4 competing processes, which execute the computational loop of a laplace example in different ways and prints out the timings. The loop is executed on the CPU or on the device (typically GPU) with either a more or a less optimal nested loop order. 

The cases 'cpu_slow' and 'cpu_fast' in [laplace_multiarch.c](src/laplace_multiarch.c) use normal C style for loops. The cases 'gpu_slow' and 'gpu_fast' use a special 'parallel_for' construct for the loops. In the C example, this is based on a preprocessor macro, whereas the C++ example is based on a lambda function, an approach similar to some accelerator frameworks such SYCL, Kokkos, RAJA, etc. Either option allows conditional compilation of the loops for multiple architectures while keeping the source code clean and readable.

The code can be conditionally compiled for either CUDA, HIP, KOKKOS, or HOST execution with or without MPI. The correct definitions for each accelerator backend option are selected in [comms.h](src/comms.h) by choosing the respective header file. The compilation instructions are shown below:

```
// Compile to run parallel on GPU with CUDA
make MPI=1

// Compile to run parallel on GPU with HIP
make HIP=CUDA MPI=1

// Compile to run parallel on GPU with KOKKOS
git clone https://github.com/kokkos/kokkos.git
make KOKKOS=CUDA MPI=1

// Compiler to run sequentially on CPU
make HOST=1 MPI=1

```
Note that when compiling to run sequentially on the CPU, it would at first appear that the 'gpu_slow' and 'gpu_fast' cases become equivalent to the 'cpu_slow' and 'cpu_fast'. However, this is actually only the case for the C example which uses preprocessor macros. In the C++ example, the 'parallel_for' loops that use lambda functions become less optimized by the compiler and are thus slightly slower. 

On the other hand, the preprocessor macro approach used in the C example can obfuscate code from a debugger (the backtrace only points to the beginning of the loop, not to the exact line of the issue), making debugging slightly less productive.

The executable can be run with 4 MPI processes using a sbatch script: 

```
#!/bin/bash -x
#SBATCH --account=xxx
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:15:00

srun laplace 4
```
