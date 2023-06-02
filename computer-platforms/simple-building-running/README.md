## Makefiles, modules, and batch job system

1. Build the provided code by issuing `make`
2. "Modify" one of the source files by `touch util.c`, run `make` again and observe 
   the behaviour
3. Try to run the resulting executable `prog` via batch system. A template batch job 
   scripts are provided in [../](../)
4. For simple testing, it is sometimes convenient to launch the program directly 
   from command line with `srun`
```
srun --account=<my_account> --nodes=<nodes> --ntasks-per-node=<tasks-per-node> --partition=<partition> ./prog
```
   Try to run the program also this way.
