## Simple scalability test

1. Build three dimensional heat equation solver under [heat-equation-3d](heat-equation-3d)  with the provided [Makefile](Makefile).
2. Run the program `heat_hip` with different number of GPUs (1, 2, 4, 8, 16) and 
   investigate scalability. **Note:** for 16 GPUs you need two nodes, other cases 
   should be run wiht single node
