## Makefiles, modules, and batch job system

a) Clone your own repository also to Sisu. The system default **git** is pretty old, so first load a module for more recent **git** with `module load git`. In Sisu, go to the folder `unix-version-control/` and try to compile the `prog.c` using the make command (`make prog`). Compilation should fail with an error message telling that you should use gnu compiler.

b) Issue the correct module command to switch the development environment from `PrgEnv-cray` to `PrgEnv-gnu` and try again to compile the program.

c) See the batch job script `job_script.sh` and add the missing `aprun` command to execute the program that you just compiled. The program should be run using 24 cores! Send your job to the queue and check the results.
