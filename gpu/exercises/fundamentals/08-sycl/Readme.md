# SYCL Exercises
SYCL codes can be compiled using one of the two implementations, **OneAPI** or **AdaptiveCPP**. 

A different rocm module is used, **rocm/6.2.2**. 

SYCL applications are executed the same way as a usual GPU or CPU applications depending on the desire target device. 

## OneAPI + AMD Plug-in
### Set-up the modules and paths
Run these commands:
```
source /projappl/project_462000956/apps/intel/oneapi/setvars.sh --include-intel-llvm
module load LUMI/24.03
module load partition/G
module load rocm/6.2.2
export  HSA_XNACK=1 # enables managed memory
export MPICH_GPU_SUPPORT_ENABLED=1                                # Needed for using GPU-aware MPI
```
The intel initialization is done before loading the other modules to avoid overwriting the environment variables.
### Compile
```
icpx -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a <sycl_code>.cpp
```
## AdaptiveCPP
### Set-up the modules and paths
Run these commands:
```
module load LUMI/24.03
module load partition/G
module load rocm/6.2.2
export PATH=/projappl/project_462000956/apps/ACPP/bin/:$PATH
export LD_LIBRARY_PATH=/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeGNU-24.03/lib64/:$LD_LIBRARY_PATH
export LD_PRELOAD=/appl/lumi/SW/LUMI-24.03/G/EB/rocm/6.2.2/llvm/lib/libomp.so
export  HSA_XNACK=1 # enables managed memory
export MPICH_GPU_SUPPORT_ENABLED=1                                # Needed for using GPU-aware MPI
``` 
### Compile
```
acpp -O3 --acpp-targets="omp.accelerated;hip:gfx90a" <sycl_code>.cpp
```

