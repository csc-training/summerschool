# Simple tasking

Explore tasking and data sharing with tasks

Starting from the skeleton code ([tasks.c](tasks.c) or
[tasks.F90](tasks.F90)), add an OpenMP parallel region to the code,
and start **tasks** so that each loop iteration is executed by a task.
At the end, the contents of `array` should be consistent with the
printouts from the tasks (note that the output is generally different
between different runs of the code). 

Play around with different data sharing clauses (both for the parallel
region and for the tasks), and investigate how they affect the results.
What kind of clauses are needed for obtaining the results described above?
