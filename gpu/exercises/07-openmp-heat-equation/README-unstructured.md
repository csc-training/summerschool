## Optimizing heat equation solver

Try to improve the performance of the heat equation solver by optimizing data
movements. During the time evolution iteration, the data can be mostly kept
within the device.

1. Implement routines `enter_data()` and `exit_data()` (e.g. in
   [cpp/core.cpp](cpp/core.cpp) or [fortran/core.F90](fortran/core.F90)). The
   routines should take `current` and `previous` fields as arguments, and
   create unstructured data region where `enter_data()` copies the data into
   the device and `exit_data()` out from the device. Implement also routine
   `update_host()` which takes a single field as argument and copies its data
   from device to host.

2. Call the newly created routines at appropriate places in the main routine.
   Modify also the offload constructs in `evolve` accordingly.
