# Monte Carlo multi-architecture example

This example uses the Monte Carlo method to simulate the value of Bessel's correction that minimizes the mean squared error in the calculation of the sample standard deviation for the chosen sample and population sizes. The sample standard deviation is typically calculated as $$s = \sqrt{\frac{1}{N - \beta}\sum_{i=1}^{N}(x_i - \bar{x})^2}$$ where $$\beta = 1.$$ The simulation calculates the mean squared error for different values of $\beta$.

The implementation uses a special construct for the parallel loops in [bessel.cpp](src/bessel.cpp). In the C example, this is based on a preprocessor macro, whereas the C++ example is based on a lambda function, an approach similar to some accelerator frameworks such as SYCL, Kokkos, RAJA, etc. Either option allows conditional compilation of the loops for multiple architectures while keeping the source code clean and readable. An example of the usage of curand and hiprand random number generation libraries inside a GPU kernel are given in [devices_cuda.h](src/devices_cuda.h) and [devices_hip.h](src/devices_hip.h).

The code can be conditionally compiled for either CUDA, HIP, or HOST execution with or without MPI. The correct definitions for each accelerator backend option are selected in [comms.h](src/comms.h) by choosing the respective header file. The compilation instructions are shown below:

```
// Compile to run parallel on GPU with CUDA
make

// Compile to run parallel on GPU with CUDA and MPI
make MPI=1

// Compile to run parallel on GPU with HIP and MPI
make HIP=CUDA MPI=1

// Compile to run sequentially on CPU with MPI
make HOST=1 MPI=1

```

The executable can be run with 4 MPI processes using a sbatch script: 
```
#!/bin/bash -x
#SBATCH --account=xxx
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:15:00

srun bessel 4
```