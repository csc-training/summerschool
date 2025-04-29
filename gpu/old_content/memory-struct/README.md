# Unified memory and structs

The purpose of this exercise is to run a loop accessing a struct from host and
device using different memory management strategies.

The function `runHost()` demonstrates the execution on host and is already complete. 

The task is to fill the functions `runDeviceUnifiedMem()` and `runDeviceExplicitMem()` to do
the same thing parallel on the device. The latter function also requires explicitly specifying how the struct is copied to the GPU memory, which is not always trivial. Therefore, you must also fill the GPU struct allocation and deallocation functions `createDeviceExample()` and `freeDeviceExample()`.
