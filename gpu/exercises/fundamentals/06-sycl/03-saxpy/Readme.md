# SAXPY with SYCL

In this exercise, you will solve the `axpy` problem (`Y=Y+a*X`). This exercise will will be used to exemplify all the SYCL concepts presented in the lecture.


**Structure of the Code**:
  1. define a SYCL  queue (choose device, specify the options)
  1. declare  the variables
  1. initialize the input variables(on cpu or on device using a separate kernels for each array)
  1. transfer the necesary data from CPU to GPU (not needed when the problem is initialized on the device)
  1. do the final `axpy` computation in another kernel 
  1. copy data to host to check the results

     
