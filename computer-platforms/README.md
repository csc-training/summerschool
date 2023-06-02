## Makefiles, modules, and batch job system

a) Clone your own summerschool repository also to Puhti. The system default **git** is pretty old, so first load a module for more recent **git** with `module load git`. In Puhti, go to the folder `unix-version-control/` and try to compile the `prog.cpp` file using the make command (`make prog`).

b) See the batch job script `job_script.sh` and add the missing `srun` command to execute the program that you just compiled. Run the program using 24 cores! Send your job to the queue with the command `sbatch job_script.sh` and check the results.
