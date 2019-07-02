## Compilation

First load the modules
```
module load cuda/10.0 pgi/19.1 openmpi/3.1.4 libpng/1.6
```

Then compile with
```
mpicc -acc hello.c
```
or
```
mpif90 -acc hello.F90
```


## Running
You can execute your code in Taito (`taito-gpu`) with a command

```
srun -N1 -n1 -pgpu --gres=gpu:p100:1 --reservation=Summerschool ./a.out
```

Alternatively you can submit this information with a jobscript file `job-gpu.sh`.


