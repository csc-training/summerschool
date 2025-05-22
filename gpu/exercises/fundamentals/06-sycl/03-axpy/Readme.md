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
Start by defining a **queue**  and selecting the appropriate device selector. SYCL provides predefined selectors, such as: default, gpu, cpu, accelerator or you can use the procedure from the [previous exercise](../01-info/enumerate_device.cpp).



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

**Optional**: use **item** class instead of **id**. Modify the lambda function to use the  **sycl::item** class instead of the **sycl::::id** class. In this case the index `idx` is obtained from the `.get_id()` member.

