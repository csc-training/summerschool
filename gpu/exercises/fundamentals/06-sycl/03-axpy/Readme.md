# AXPY with SYCL

In this exercise, you will solve the `axpy` problem (`Y=Y+a*X`). This exercise will will be used to exemplify all the SYCL concepts presented in the lecture.


**Structure of the Code**:
  1. define a SYCL  queue (choose device, specify the options)
  1. declare  the variables
  1. initialize the input variables (on CPU or on device using a separate kernels for each array)
  1. transfer the necesary data from CPU to device (not needed when the problem is initialized on the device)
  1. do the final `axpy` computation in a separate kernel 
  1. copy data to host to check the results

## I. Memory management using Buffer and Accesors and Basic Kernel Launching

Use the skeleton provided in `saxpy_buffer_simple.cpp`. Look for the **//TODO** lines.

### Step 1: Define a Queue
Start by defining a **queue**  and selecting the appropriate device selector. SYCL provides predefined selectors, such as: default, gpu, cpu, accelerator:

- `queue q(default_selector_v);` targets the best device
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
 - `sycl::range{N}` or `sycl::range(N)` specify number of work-items be launched 
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


Starting from the solution of the task I change the way the kernel is launched following:

```cpp
   h.parallel_for(sycl::nd_range<1>(sycl::range<1>(((N+local_size-1)/local_size)*local_size), sycl::range<1>(local_size)), [=](sycl::nd_item<1> item) {
        auto idx=item.get_global_id(0);
        if(idx<N){ //to avoid out of bounds access
          y_acc[idx] = y_acc[idx] + a*x_acc[idx];
        }
      });
```
In the launching the programmer can define not only the number of work-items to execute the kernel, but also the size of the work-group. Both global and local coordinates of the work-item can be now obtained from the nd_item object (via `get_global_id()`, `get_global_linear_id()` and `get_local_id()`, `get_local_linear_id` methods).

**Note** that nd_range requires that the total number of work-items to be divisible by the size of the work-group. Asumming `local_size` for the work-group, `(N+local_size-1)/local_size)*local_size)` need work-items to be created. This number is larger than `N` if `N`is not divisible by `local_size`.

      
