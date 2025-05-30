# AXPY with SYCL using essentials

In this exercise, you will solve the `axpy` problem (`Y=Y+a*X`). This exercise will will be used to exemplify all the SYCL concepts presented in the lecture.

**Structure of the Code**:
  1. define a SYCL  queue (choose device, specify the options)
  1. declare  the variables
  1. initialize the input variables (on CPU or on device using a separate kernels for each array)
  1. transfer the necesary data from CPU to device (not needed when the problem is initialized on the device)
  1. do the final `axpy` computation in a separate kernel 
  1. copy data to host to check the results


## I. Memory management using Buffer and Accesors and Basic Kernel Launching

Use the skeleton provided in [`axpy.cpp`](axpy.cpp). Look for the **//TODO** lines.

### Step 1: Define a Queue
Start by defining a **queue**  and selecting the appropriate device selector. SYCL provides predefined selectors, such as: default, gpu, cpu, accelerator:

- `queue q(default_selector_v);`  or `queue q;` targets the best device
-  `queue q(cpu_selector_v);` targets the best CPU
-  `queue q(gpu_selector_v);` targets the best GPU
-  `queue q(accelerator_selector_v);` targets the best accelerator
    
Alternatively it is possible to use the procedure from the [previous exercise](../01-info/enumerate_device.cpp). This the recommended way when the application can detect than one GPU and needs to assign specific devices accordingly to the MPI rank or (CPU) OpenMP thread index.


### Step 2: Create Buffers
Next, create buffers to encapsulate the data. For a one-dimensional array of integers of length `N`, with pointer `P`, a buffer can be constructed as follows:

```cpp
    sycl::buffer<int, 1> x_buf(P, sycl::range<1>(N));
```
Use the appropriate data type. 


### Step 3: Create Accessors
Accessors provide a mechanism to access data inside the buffers. Accessors on the device must be created within command groups. There are two ways to create accessors. Using the `sycl::accessor` class constructor

```cpp
   sycl::accessor x_acc{x_buf, h, read_write};
```
or  
```cpp
   auto a_acc = sycl::accessor{a_buf, h, sycl::read_write};
```
**Important**  Use appropriate access modes for your data:
 - **Input Buffers:** Use `sycl::read_only` / `sycl::access::mode::read` to avoid unnecessary device-to-host data transfers.
 - **Output Buffers:** Use `sycl::write_only`/ `sycl::access::mode::write` to avoid unnecessary host-to-device data transfers.
 - **Input/Ouput Buffers:** Use `sycl::read_write` / `sycl::access::mode::read_write` for the variables that are input, but they also get modified during the computations.


### Step 4: Submit the Task using Basic Submission
Once accessors are ready, submit the task to the device using the `.parallel_for()` member function. The basic submission:

```cpp
   h.parallel_for(sycl::range{N}, [=](sycl::id<1> idx) {
        y_acc[idx] = y_acc[idx] + a*x_acc[idx];
      });
```
Here: 
 - `sycl::range{N}` specifies number of work-items be launched 
 - `sycl::id<1>` represents the index used within the kernel.
**Optional**: use **item** class instead of **id**. Modify the lambda function to use the  **sycl::item** class instead of the **sycl::id** class. In this case the index `idx` is obtained from the `.get_id()` member.

### Step 5: Check the results using `host_accessor`
When a buffer is destroyed the host can access again the data to which the buffer encapsulates. However sometimes there might be a need to check the results of some operations and still the buffer for further calculations. In this case host accessors can be used:
```
   // Use host_accessor to read back the results from Ybuff
      {
          host_accessor h_accY(Ybuff, sycl::read_only); // Read back data after kernel execution
          std::cout << "First few elements of Y after operation:" << std::endl;
          for (size_t i = 0; i < 10; ++i) {
            std::cout << "Y[" << i << "] = " << h_accY[i] << std::endl;
          }
      }
```
As long a host accessor is valid the data can not be accessed by other means. When they are destroyed the program can proceed with further calculations on host or devices.

## II. Memory management using Buffer and Accesors and `nd_range" Launching
In the previous task a basic, simple way was used to launch kernels. This could be enough for many applications, but the `range` class is quite limited. It does not allow to use lower level features, like local share memory, in-work-grgoup  synchronizations or use the in-work-rgoup local index. In many cases (like matrix-matrix multiplication) more control is needed.  

The axpy calculation does not need notions of locality within the kernel, but for its simplicity is a good exercise to familiarize with the syntax.


Starting from the [solution of the task I](solution/axpy_buffer_simplek.cpp) change the way the kernel is launched following:

```cpp
   h.parallel_for(sycl::nd_range<1>(sycl::range<1>(((N+local_size-1)/local_size)*local_size), sycl::range<1>(local_size)), [=](sycl::nd_item<1> item) {
        auto idx=item.get_global_id(0);
        if(idx<N){ //to avoid out of bounds access
          y_acc[idx] = y_acc[idx] + a*x_acc[idx];
        }
      });
```
In the launching the programmer can define not only the number of work-items to execute the kernel, but also the size of the work-group. Both global and local coordinates of the work-item can be now obtained from the nd_item object (via `get_global_id()`, `get_global_linear_id()` and `get_local_id()`, `get_local_linear_id` methods).

**Note** that nd_range requires that the total number of work-items to be divisible by the size of the work-group. Asumming `local_size` for the work-group, `(N+local_size-1)/local_size)*local_size)` need work-items to be created. When `N`is not divisible by `local_size` the number of work-items is larger than `N`.

## III. Memory management with Unified Shared Memory (`malloc_device()`) and Simple Launching
The task is to write a code performing an `AXPY` operation, but this time the memory management is done using the Unified Shared Memory (USM). The structure of the code is practically the same. 

Start from the skeleton code  [`axpy.cpp`](axpy.cpp). Look for the **//TODO** lines.


### Step 1: Define a Queue
Same as in task I

### Step 2: Allocate device memory using USM
The memory on the device is allocate C style using `malloc_device` function:

```cpp
    int* x_dev = sycl::malloc_device<int>(N, q);
```
**Note** how the device pointers are associated with a specific queue.


### Step 3: Copy data from host to device
The transfer of the data is submitted to a specific queue:
```cpp
   q.memcpy(x_dev, x.data(), sizeof(int) * N).wait();
```
The `memcpy()` method is needed to transfer data from host to device. The first argument is the destination pointer, the second the source pointer, and finally the size of the data in bytes. This operation is asynchronous. Hence `.wait()`. This method pauses the program execution until the operation is completed. Otherwise a subsequent operation submitted to the same queue might start before the data is transfered to the GPU.

### Step 4: Submit the Task using Basic Submission
This is done similarly to taks I
```cpp
    q.submit([&](handler& h) {

       h.parallel_for(sycl::range{N}, [=](sycl::id<1> idx) {
        y_dev[idx] = y_dev[idx] + a*x_dev[idx];
      });
    }).wait();
``` 
In this case `.wait()`  method pauses the prorgam until the the operation is 
### Step 5: Access the results on the host
When using USM and `malloc_device` the transfer from device to host needs to be explicitely coded. The same method `memcopy()` is used:

```cpp
   q.memcpy(x.data(), x_dev, sizeof(int) * N).wait();
```
Now the destination is the host pointer (first argument), while the source (second argument) is the device pointer.
Again the `.wait()` method is needed to pause the program execution. This way it is guaranteed that the next step is not executed before all data from device is transfered.

## IV. Memory management with Unified Shared Memory (`malloc_device()`) and `nd_range` Launching

Starting from the [solution of the previous task](solution/axpy_usm_device_simplek.cpp) change the kernel launching to use `nd_range`, similar to task II.

## V. Memory management with Unified Shared Memory (`malloc_shared()`) and `simple` Launching
The `malloc_device()` function allocates memory on the devices which is pinned to the device. There can be no migration to host. Furthermore the host can not access in any direct this data. In order to simplify the programmer's life SYCL provides also mechanisms (similar to `hipMallocManaged()`)  to allocate memory which can be migrated between host and device as needed. When a block of code running on the host is encountered the memory is migrated automatically to the host, while if a kernel is executed the data will be on the device. Between two subsequent kernels the data stays on the device and similarly on the host, between two operations subsequent using the data no migration occurs.

Start from the solution of task III or task IV. Remove the host pointers `x(N),y(N);` and the device pointers `x_dev,y_dev`. Remove as all the lines code used to transfer data from host to device and from device to host.

Declare new pointers using `malloc_shared()` such as:
```cpp
    int* x_shared = sycl::malloc_device<int>(N, q);
```
Then initialize the `x_shared` and `y_shared` using host. **Note** the STL functions will not work. 

The data is first touched on the host which means that the pointers will reside on the host memory. When the kernel is executed a page fault will occured and it will trigger a migration to the device allowing ot execute the kernel on the device.  In this case the `.wait()`  is still needed to be sure that the host code wich will access the results at the end will now be executed before the kernel is completed.

# SYCL Dependencies with AXPY
Many application have independent parts that can be executed in any order. An application in which all operations are exececuted in the order of submission will always give the correct results, but it can result in an ineficient use of the resources. If some kernels can be executed in any order the runtime can schedule them in such a way that the devices are occupied as much as possible. 

The AXPY operation can be use as an test case to implement some basic dedendencies. First `X` and `Y` arrays can be initialized independentely of each other. Then the operation can be performed. Dependencies can be enforce in many ways depending on the memory model used. 

When using buffer and accessor model the dependencies are automatically satisified, because data in the buffers can be accessed only via accessors which block any other access until they are destroyed. W

When using USM the programmer needs to specify the dependencies explicitely. The kernels launch is asynchronous and there is no warranty that they are executed in the order of submission. The worst way to program is to use an out-of-order queue and use `.wait()` for each operation. It is the worst becuase the program will pause and wait for each operation before conitnued to the next submission. Lots of synchronization between host and device are costly and also results in  lots of idle time in the device. A slighly better approach is to enforce an order of execution to be the the order of submission. This means that the `.wait()` can be removed from almost all submissions to the queue. It is possible to use in-order queues .  But this would be inefficient in some cases becuase it does not allow for concurrency. Finally we can have dependencies specified via sycl events. Each submission to a queue returns a variable event. If operation B dependends on operation A, it is possible to specify a dependency off the operation B of the event returned by operation A. 

## VI. Dependencies via Buffers
Start from the solution of task I or II. It is ok to remove the initialization of the variables `X`and `Y`. Write separate kernel initializing each one separately. Keep in mind that in this case the data from the host is not needed.
## VII. Dependencies when Using USM  and `in-order` queues
Start from the solution of task III, IV, or V. Similarly to task VI remove the initialization on the host and write separate kernel initializing for each variablre separately. Chane ge also the queue definition. An in-orde queue is defined using:

```
sycl::queue queue(sycl::default_selector{}, sycl::property::queue::in_order{});
```
## VIII. Dependencies when Using USM  and events
Start from task VIII. Change the definition the queue to make it out-of-order again. **Out-of-order** queues can use `sycl::events` to explicitly set the order of execution. Each kernel submission returns an event, which can be used to ensure that subsequent tasks wait for the completion of preceding tasks. 

Initialize arrays `X` and `Y` on the device using two separate kernels. Capture the events from these kernel submissions:
```cpp
auto event_x = queue.submit([&](sycl::handler &h) {
    h.parallel_for(range{N}, [=](id<1> idx) { X[idx] = 1; });
});
auto event_b = queue.submit([&](sycl::handler &h) {
    h.parallel_for(range{N}, [=](id<1> idx) { Y[idx] = 2; });
});
```
Next submit the `axpy` kernel with an explicit dependency on the two initialization events
 ```cpp
 queue.submit([&](sycl::handler &h) {
    h.depends_on({event_x, event_y});
    h.parallel_for(range{N}, [=](id<1> idx) { Y[idx] += a * X[idx]; });
});
```
or 

 ```cpp
 queue.
    h.parallel_for(range{N},{event_x, event_y}, [=](id<1> idx) { Y[idx] += a * X[idx]; });
```
As an exercise you can synchrhonize the host with the event `sycl::event::wait({event_a, event_b});`. Finally copy the result back to the host for validation.


