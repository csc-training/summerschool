# SAXPY with SYCL

In this exercise, you will solve the `axpy` problem (`Y=Y+a*X`). This exercise will will be used to exemplify all the SYCL concepts presented in the lecture.


**Structure of the Code**:
The structure of the SYCL code is:
  1. define a SYCL  queue
  1. declare  the variables
  1. initiate the input variables(on cpu or on device using a separate kernels for each array)
  1. do the final `axpy` computation in another kernel 
  1. copy data to host to check the results

     
