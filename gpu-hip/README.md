# GPU programming with HIP

Course material for the CSC course "GPU programming with HIP". The course is
part of the PRACE Training Center (PTC) activities at CSC.

## Exercises

[General instructions](exercise-instructions.md)

### Basics

- [Hello world](hello-world)
- [Kernel: saxpy](kernel-saxpy)
- [Kernel: copy2d](kernel-copy2d)

### Synchronisation and streams

- [Investigating streams and events](streams)

### Memory management

- [Memory management strategies](memory/prefetch)
- [Unified memory and structs](memory/struct)

### Fortran and HIP

- [hipfort](hipfort)

### Optimisation

- [Matrix Transpose](matrix_transpose)
- [Nbody](nbody)

### Multi-GPU programming

- [Multi-GPU exercise without MPI](multi-gpu/multigpu)
- [Multi-GPU exercise with MPI](multi-gpu/mpi)
- [Peer to peer device access](multi-gpu/p2pcopy)
- [Bonus: Heat equation with HIP](heat-equation)
