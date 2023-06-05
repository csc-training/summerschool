# Hello world with OpenMP offloading

Compile and run a simple OpenMP test program, provided as `hello(.c|.F90)`.

1. Compile the program first without offloading support. you will need 

```
#!/bin/bash
module load LUMI/23.03 partition/C cce/15.0.1
```

  Try to run the code in a CPU and GPU node.

2. Next, compile the code with offloading support.

```
#!/bin/bash
module load LUMI/23.03 partition/G cce/15.0.1 rocm
```

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
