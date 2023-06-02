## Simple scalability test

1. Build three dimensional heat equation solver. Modify [Makefile](Makefile) as needed
2. Run the program `heat_mpi` with different number of GPUs (1, 2, 4, 8, 16) and 
   investigate scalability. **Note:** for 16 GPUs you need two nodes, other cases 
   should be run wiht single node
