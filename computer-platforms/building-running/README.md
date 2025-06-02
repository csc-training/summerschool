# Makefiles, modules, and batch job system

In this exercise we get practice on editing files, loading modules, compiling code, and submitting jobs.

## Building and running CPU code

0. Load the environment for CPU codes (run this in a fresh terminal):

       module load LUMI/24.03
       module load partition/C

1. Build the code in [cpu](cpu) directory with `make`:

       cd cpu
       make

   and observe the output:

       CC -c prog.cpp
       CC -c util.cpp
       CC -o prog prog.o util.o

   These are the commands that make executed in order to compile and link the code.
2. Modify the file `prog.cpp` by replacing `Hello world` with `Hello Nuuksio` on line 13.
   You can edit the file using, e.g., `nano` or any other editor of your choice.
3. Run make again and observe the output:

       CC -c util.cpp
       CC -o prog prog.o util.o

   Note that file `prog.cpp` was not compiled again, because it hadn't changed since the previous compilation.
   This is the crux of make and it can save a lot of time when developing larger code bases.
4. Run the resulting executable `prog` via batch system by using an example batch job script `job_cpu.sh`:

       sbatch job_cpu.sh

5. Run the executable with different number of nodes and tasks per node. You can do this by editing `job_cpu.sh`, but here is a protip:
   You can override the sbatch parameters through equivalent command line options. For example:

       sbatch --nodes=2 --ntasks-per-node=4 job_cpu.sh

6. For testing short jobs, it is often convenient to submit and launch the program directly from command line:

       srun --account=project_462000956 --nodes=2 --ntasks-per-node=4 --partition=small --time=0:05:00 ./prog

   The parameters like `--nodes=...` are the same in all different ways of executing.
   Feel free to use the way that is most convenient for you during the summer school.


## Building and running GPU code

0. Load the environment for GPU codes (run this in a fresh terminal):

       module load LUMI/24.03
       module load partition/G
       module load rocm/6.0.3

1. Build the code in the [gpu](gpu) directory with `make`.
2. Run the resulting executable `prog` via batch system by using an example batch job script `job_gpu.sh`.
   Note the main differences between `job_cpu.sh` and `job_gpu.sh`: the partition names are different and
   for GPU job we define `--gpus-per-node=...` to request a number GPUs per node.
