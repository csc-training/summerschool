<!--
SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Using multiple GPUs with MPI

In this exercise you will assign GPUs to different MPI ranks in a simple
"hello, world" program.

The provided [hello.cpp](hello.cpp) code is an MPI program that prints out
total number of MPI tasks and the rank of each task, selected device id and
number of available devices.

# Tasks

## Part A

1. Fill in the TODO tasks (part a) in `hello.cpp`. Compile the code, run the
   code with 2 nodes, 2 tasks per node and 2 gpus per node.
2. Create a bash script `select_gpu` that limits the visibility of GPUs on a
   node to `SLURM_LOCALID`. (*Hint*: `ROCR_VISIBLE_DEVICES`)
3. Run the code again. If you see the number of devices available to each task
   go to 1, this part was successful.

## Part B

Now, don't use `select_gpu`, but instead, select the GPUs by splitting the
`MPI_COMM_WORLD` communicator.

Expand the code as follows:

1. Create a node local communicator by utilizing `MPI_Comm_split_type` and `MPI_COMM_TYPE_SHARED`.
   Query also node local rank from this communicator
2. Query the number of GPUs in each node
3. Assign a GPU to each rank. In the most common situation single GPU is used
   by single MPI task, but in some cases (*e.g.* single MPI task does not have
   enough parallelism) it can be useful to assign multiple GPUs to same GPU? 
   Can you figure out how to achieve that?

Run the program with various combinations of `--nodes`, `--ntasks-per-node` and `--gpus-per-node`, 
e.g. two nodes with two tasks and two GPUs on each node i.e.
```
...
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
...

srun ./hello
```

