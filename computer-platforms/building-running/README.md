## Makefiles, modules, and batch job system

### Building and running CPU code

1. Build the code in [cpu](cpu) with `make`. In LUMI the correct compiler command is
  `cc`, modify [Makefile](cpu/Makefile) as needed.
2. "Modify" one of the source files by `touch util.c`, run `make` again and observe 
   the behaviour
3. Try to run the resulting executable `prog` via batch system. A template batch job 
   scripts are provided in [../](../). Try different number of nodes and tasks per node.
4. For simple testing, it is sometimes convenient to launch the program directly 
   from command line with `srun`
```
srun --account=<my_account> --nodes=<nodes> --ntasks-per-node=<tasks-per-node> --partition=<partition> ./prog
```
   Try to run the program also this way.

### Building and running GPU code

1. Build the code in [gpu](gpu) with `make`. Modify [Makefile](gpu/Makefile) as needed.
   **Note:** You need to load proper GPU modules.
2. Try to run the resulting executable `prog` via batch system. A template batch job 
   scripts are provided in [../](../). Try different number of nodes and gpus per node.
3. Try to run the program also directly from command line with `srun`
