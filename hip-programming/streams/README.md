# Investigating streams and events

The file called async_serial.cpp includes an example to transfer asynchronously data from host to device, do the computation, and transfer the result from the host to device asynchronously.

We can observe in the code that we record the startEvent on stream 0, copy the data to the device, launch the kernel and copy the data back to the host. Then, we record the stopEvent event and we calculate the elapsed time with the HIP call hipEventElapsedTime.

## Compile 

`make async_serial`

## Execute:

`sbatch async_serial.sh`

## Results

Check the SLURM output file

## Pinned memory
The code has already pinned memory hipHostMalloc. Try to uncomment the line 51 and comment line 52, you will not use pinned memory and the performance will be lower.

## Tips

For the following exercises, you would need:

* hipMemcpyAsync
* hipStreamCreate
* Use the defined `nStreams` to calculate the `streamSize` based on `n` and define the `streamBytes`which is the total bytes per stream

## Exercises

### Case 1

Copy the async_serial.cpp file to async_case1.cpp and edit after the sequential transfer by adding the code to do the following:

1) Create 4 streams and calculate the size of bytes per stream
2) Record a new event (use same variables where possible)
3) Use same blocking data transfers
4) For the number of streams, do:
	4.1) Execute the kernels in different stream with the appropriate workload per stream

#### Compile and execute

```
make async_case1
sbatch async_case1.sh
```

### Results

Check the new output slurm file

### Case 2

Copy the async_case1.cpp file to async_case2.cpp and edit after the code fo the previous exercise by adding the code to do the following:

1) Create 4 streams and calculate the size of bytes per stream
2) Record a new event (use same variables where possible)
3) For the number of streams, do:
	3.1) Copy the data from host to device through the streams and using hipMemcpyAsync (be careful how much data you transfer per stream and try to create an offset per stream)
	3.2) Launch the kernel for each stream
	3.3) Copy the data from the device to host through hipMemcpyAsync and streams
	3.4) Synchronize the events (hipEventSynchronize)
	3.5) Record a stop event and calculate the new Elapsed time.

#### Compile and execute

```
make async_case2
sbatch async_case2.sh
```

### Results

Check the new output slurm file

Is your code faster? Why?

### Case 3

Copy the async_case2.cpp to async_case3.cpp and edit below the previous code (before the deallocation of the memory)

1) Now, make a loop and send the data from host to device through streams
2) Similar for the kernsl
3) Also for the data copy from device to host.
4) The difference in this exercise is that we have a loop per each of the previous steps and not all the steps inside a loop

#### Compile and execute

```
make async_case3
sbatch async_case3.sh
```

### Results

Check the new output slurm file


Is your code faster? Why?

## Solutions

The solutions are in the corresponding directory

## Questions

* What if we do not synchronize the events? 
