<!--
SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

## Exercise: Errors IN kernel

As we have already mentioned, kernels are executed in asynchronous way. Which means that the trick we just defined will run while the kernel is running, and it will NOT catch errors happening INSIDE the kernel itself.
Debugging GPU kernels is unfortunately more painful than debugging "normal" code. This happens because of the GPU architecture: it is designed to run many parallel threads, organized in blocks/grids which maps to different compute units(or streaming multiprocessors).
And in parallel from the host code too. So we have many levels of parallelism to take into account when debugging GPU code.

In CUDA, and future rocm version (from 7.0 on), One first helper when something goes wrong is to force GPU synchronization after every kernel, and check the error code. This can be done by adding this functionality to the macro we defined previously. 
NOTE: it is important to be sure that this part of the macro is executed ONLY WHEN DEBUGGING! synchronization is an expensive operation and removes some of the advantages of using a separate device.
This helps us in identifying the kernel that is misbehaving. It may not be needed if the kernel triggers some segfault or similar "deadly" error, because you already know which one is the "offending" kernel.

Once you know that, there are two approaches. either you use a debugger (e.g. rocgdb) or you can insert prints in your device code to see what's happening on the device. 

If you go for this second approach, remember that threads are executed in parallel, and out of order.
For this reason it is useful to add some "identifiers" to all the prints done, so that you can know which thread is printing what.
For example you can use: 
`printf("Block is (%d,%d,%d) thread is (%d,%d,%d): MESSAGE\n",blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z)`
Another useful piece of information is that prints on GPU are buffered and only appear in the stdout/err AFTER the end of the kernel. so if you trigger a segfault, you will not see any prints. A trick I used in the past is to find the instruction that triggers the segfault and comment the kernel from there on to get the prints and analyze them. (TODO: CHECK THIS IS STILL TRUE)
Finally, the buffer has a limited space. So keep the prints "small" and delete them as long as you proceed with the debugging. And be sure to remove them for the releases!

In this exercise we will fix a mistake in our kernel.

We are writing a simple 8\*16 matrix, where the values are the row index multiplied by 100 + the column index. But in the provided file we made a mistake.

In the [solution directory](solution/) there is the solution.

## Task: compile and execute

We inserted some prints to help us locate the error. can you spot it? HINT: look at the thread ids and the idx...
