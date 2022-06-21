# Loop interchange

The file `laplacian.cpp` / `laplacian.F90` calculates Laplacian of a two dimensional field. The main computational part contains a simple loop order performance bug that has a large performance impact. Compile with 

```
g++ -g -O2 -fopenmp -o laplacian laplacian.cpp
```

and analyze the program with Intel Vtune to see where the hotspot is. This can be done on Puhti by 

```
source /appl/opt/testing/intel-oneapi/setvars.sh --force
srun amplxe-cl -r vtune_hotspots -collect hotspots -- ./laplacian
vtune-gui vtune_hotspots
```

Try to compile the code with different compilers and check whether they can automatically fix the performance. You can use `-fopt-info-loop` (gcc), `-Rpass=loop*` (clang), or `-qopt-report` (Intel) for optimization reports. Do you get similar performance with all compilers? You can check also the effect of optimization level (`O1`, `O2`, `O3`).

The file `laplacian_matrix.cpp` / `laplacian_type.F90` contains a version where the fields are implemented in a more complex data structures? Does this data structure affect compiler optimizations?

Try to fix the performance bug in second version manually, and compare the performance to the version with simple arrays.
