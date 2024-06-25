# MPI section

## Learning objectives

At the end of the MPI section, the student should be able to

- Explain differences in communicating between processes/threads in a shared memory system vs a distributed memory system
- Describe deadlocking communication patterns and approaches to avoid deadlocks
- Contrast blocking and non-blocking communication
- Write MPI programs in C, C++, or Fortran for:
  - Communicating data between processes
  - Using collective communication calls over a subset of processes
- Compile and run MPI programs in supercomputers
- Start exploring some advanced MPI features relevant for their use case


## Demo codes

See [demos](demos/) for demo codes referred to in the slides.

## Wednesday exercises

### Introduction to MPI

- [Hello world](hello-world/)

### Point-to-point communication

- [Message exchange](message-exchange/)
- [Parallel pi with two processes](parallel-pi-two-procs/)

### MPI programming practices

- [Message chain](message-chain/)
- [Heat equation solver](heat-equation/): Tasks 1-2
- [(Bonus) Parallel pi with any number of processes](parallel-pi-general/)
- [(Bonus) Broadcast and scatter](broadcast-scatter/)

### Collective operations

- [Collective operations](collectives/)


## Thursday exercises

### Debugging

- [Debugging](debugging/)

### Collective reductions

- [Heat equation solver](heat-equation/): Task 3

### Non-blocking communication

- [Non-blocking message chain](message-chain-non-blocking/)
- [Heat equation solver](heat-equation/): Task 4

### User-defined communicators

- [Communicators and collectives](communicator/)

### Further MPI topics

- Cartesian topology
  - [Cartesian grid process topology](cartesian-grid/)
  - [Message chain with Cartesian communicator](message-chain-cartesian/)
  - [Heat equation solver](heat-equation/): Task 5
- User-defined datatypes
  - [User-defined datatypes](datatypes/)
  - [Modifying extent](datatypes-extent/)
  - [Communicating struct](datatypes-struct/)
- Persistent communication
  - [Message chain with persistent communication](message-chain-persistent/)

### Bonus

- [Heat equation solver](heat-equation/): Remaining tasks

