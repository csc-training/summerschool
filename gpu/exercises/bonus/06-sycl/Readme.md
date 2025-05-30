# Other SYCL Features

## X. Basic Profiling using events
Start from the solution of [Task IX](../../exercises/fundamentals/06-sycl/03-axpy/solution/axpy_dependencies_usm_device_events.cpp). First modify the **queue** definition and enable profiling
```cpp
queue q{property::queue::enable_profiling{}};
```
Next set-up `sycl::event` object the same way is done in the task VIII. Compute the execution time of the kernel by taking the difference between the end of the execution of the kernel and the start of the execution.
```
e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>();
```
**Note** Before computing the time you will have first to synchronize the host and the device (`e.wait()`)!

## XI. Error Handling

In this exercise you are given a code with errors. SYCL provides mechanisms to catch both synchonous and asynchronous expections. However the behaiviour dependends a lot on the hardware capabilities, runtime and, drivers. 

lready has in place the constructs to catch both synchonous and asynchronous expections. The task is to fix the code using the error messages given by the code.
