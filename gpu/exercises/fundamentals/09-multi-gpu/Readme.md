# Exercises Multi-GPU Programing
## SYCL+MPI compilation 

#### Intel OneAPI
```bash
icpx -fuse-ld=lld -std=c++20 -O3 -fsycl -fsycl-targets=amdgcn-amd-amdhsa,spir64_x86_64 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a `CC --cray-print-opts=cflags` <sycl_mpi_code>.cpp `CC --cray-print-opts=libs`
```

#### AdaptiveCPP
```
acpp -O3 --acpp-targets="omp.accelerated;hip:gfx90a" `CC --cray-print-opts=cflags` <sycl_mpi_code>.cpp `CC --cray-print-opts=libs`
```
