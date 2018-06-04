## Heat equation solver hybridized with MPI+OpenMP

Refer back to the two OpenMP-parallelized implementations of the 
[heat equation solver](../../openmp/heat), and combine the loop-level 
parallelization with some previous MPI implementation of the solver, e.g. 
[heat-p2p](../../mpi/heat-p2p). This is the MPI_THREAD_FUNNELED mode.
