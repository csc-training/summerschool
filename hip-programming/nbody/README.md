# Nbody

This code is part of the repository https://github.com/themathgeek13/N-Body-Simulations-CUDA

## Compile and execute the code

Edit the nbody.sh to select the appropriate node and GPU

```
cd nbody
make
sbatch noby.sh
```

## Check the output file and time

* It should be close to 100 seconds, why the execution time on AMD MI100 is slower than the NVIDIA V100?
