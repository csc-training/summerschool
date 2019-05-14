## Heat equation solver parallelized with OpenMP ##

1. Parallelize the serial heat equation solver with OpenMP by
parallelizing the loops for data initialization and time evolution.

2. Improve the OpenMP parallelization so that the parallel region
is opened and closed only once during the program execution.
	
Example solutions can be found in
[../../heat/openmp-loops/](../../heat/openmp-loops/) and
[../../heat/openmp/](../../heat/openmp/), respectively.
