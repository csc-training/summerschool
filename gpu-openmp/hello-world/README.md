# Hello world with OpenMP offloading

Compile and run a simple OpenMP test program, provided as `hello(.c|.F90)`.

1. Compile the program first without offloading support (with only `-mp`
   option), and try to run the code in a GPU node.

2. Next, compile the code with offloading support (with `-mp=gpu -gpu=cc70`).
   Try to run both in a GPU node and in a CPU node. For CPU only run remove
   `--gres` and `--reservation` from batch job script and use the `test`
   partition, *e.g.*

```
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --account=<project>
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

srun hello
```
