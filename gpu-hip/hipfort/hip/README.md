# Compilation

## AMD

If the Makefile.hipfort had no issues:

```
export HIPFORT_ARCHGPU=amdgcn-gfx908
make
```

However, execute:

```
export PATH=/opt/rocm-4.5.0/hipfort/bin:$PATH
hipfc --offload-arch=gfx908 hipsaxpy.cpp main.f03
```


## NVIDIA

If the Makefile.hipfort had no issues:

```
export HIPFORT_ARCHGPU=nvptx-sm_70
make
```

Use:

```
export PATH=/opt/rocm/hipfort/bin:$PATH
hipfc -x cu --gpu-architecture=sm_70 hipsaxpy.cpp main.f03
```
