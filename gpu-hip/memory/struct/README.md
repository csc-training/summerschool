# Unified memory and structs

The purpose of this exercise is to run a loop accessing a struct from host and
device using different memory management strategies.

The function `runHost()` demonstrates the execution on host. The task is to
fill the functions `runDeviceUnifiedMem()` and `runDeviceExplicitMem()` to do
the same thing parallel on the device. The latter function also requires
further filling struct allocation and deallocation functions
`createDeviceExample()` and `freeDeviceExample()`.
