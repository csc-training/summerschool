# Generic instructions for the exercises

For most of the exercises, skeleton codes are provided to serve as a starting
point. Some may have sections marked with TODO to indicate a place in the
code where something is missing or needs to be changed.

In addition, all of the exercises have example solutions in a `solution`
subdirectory. Note that these are seldom the only or even the best way to
solve the problem.

All of the exercise materials can be downloaded with the command

```shell
git clone https://github.com/csc-training/hip-programming.git
```

If you have a GitHub account you can also **Fork** this repository and clone
then your fork.


## Computing servers

### AMD Accelerator Cloud

To get started, log in and load the `rocmmod4.5.0` module:
```shell
ssh 205.234.16.68 -p 50023 -l username
module load rocmmod4.5.0
```

#### Disk areas

* Create your directory here: `/global/scratch/hip-csc/`
* Always work under the `/global/scratch` directory and not in `$HOME`.


#### Compilation

* Use hipcc to compile, for example: `hipcc hello.cpp -o hello`
* If compiling on the login node, you should also set the target GPU
  architecture: `hipcc hello.cpp -o hello --offload-arch=gfx908`
* If Makefile exists, use make


#### Running

* To use for example the 4th GPU of a node, run or include in your submission script:
`source /global/scratch/hip-csc/setup.sh 4`

* Run with srun: `srun --reservation=Lumi --time=00:01:00 -p MI100 --nodes 1 ./hello`

* To create `submit.sh` file, add the following contents (check the node and GPU number), and hit `sbatch submit.sh`:

```shell
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --time=00:05:00
#SBATCH --partition=MI100
#SBATCH --nodes=1
#SBATCH -w ixt-sjc2-XX
#SBATCH --reservation=Lumi

source /global/scratch/hip-csc/setup.sh X
srun -n 1 ./hello
```


### Puhti

As a backup option, we have also provided you with access to CSC's Puhti
system that has NVIDIA's V100 GPUs, but has a working HIP installation to
support code porting activities.

To get started with Puhti, you should log in to Puhti and load the `hip`
module to get a HIP compiler.

```shell
ssh -Y trainingXXX@puhti.csc.fi
module load hip
```

For more detailed instructions, please refer to the system documentation at
[Docs CSC](https://docs.csc.fi/).

#### Compiling

In order to compile code with the `hipcc` on Puhti, one needs to add an
additional flag (`-x cu`) to indicate that it should target NVIDIA hardware.
Also, the target architecture should be set with `--gpu-architecture=sm_70`:

```shell
hipcc hello.cpp -o hello -x cu --gpu-architecture=sm_70
```


#### Running

Puhti uses SLURM for batch jobs. Please see [Docs CSC](https://docs.csc.fi/)
for more details. If you are using CSC training accounts, you should use the
following project as your account: `--account=project_2000745`.

We have also reserved some GPU nodes for the course. In order to use these
dedicated nodes, you need to run your job with the option
`--reservation=hip2021`, such as

```shell
srun --reservation=hip2021 -n1 -p gpu --gres=gpu:v100:1 --account=project_2000745 ./my_program
```

Please note that the normal GPU partition (`-p gpu`) needs to be used with
the reservation. Otherwise you may use the `gputest` partition for rapid fire
testing.
