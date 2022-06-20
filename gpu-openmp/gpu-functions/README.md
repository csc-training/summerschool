## Compiling functions for target

Functions cannot be called within `target` regions unless the also a "target"
version of the function is compiled. The main programs `sum(.c|.F90)` calculate
a sum of two vectors using a function `my_sum` which is defined externally in 
`my_sum.c` / `my_mod.F90`. As such, the program does not compile correctly.

Add appropriate `declare target` constructs to the code. You can build the whole program as
```bash
nvc -o sum sum.c my_sum.c -mp=gpu -gpu=cc80
```
or
```bash
nvfortran -o sum my_mod.F90 sum.F90 -mp=gpu -gpu=cc80
```

