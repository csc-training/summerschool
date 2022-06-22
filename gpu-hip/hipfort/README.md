# Hipfort: saxpy

Test hipfort by compiling and running a simple Fortran code that uses a HIP
kernel to calculate saxpy on the GPU.

Before compilation, one needs to set up the environment for hipfort. Easiest
way to do this is to include a Makefile provided by hipfort in your own
Makefile. For it to work, one needs to first set two environment variables:
`HIPFORT_ARCHGPU` to set the target architecture and `HIPFORT_HOME` to point
to the path where hipfort is installed. It also makes some assumptions about
other environment variables, so depending on the system one may need to do
further adjustments. Please see the included [Makefile](Makefile) for an
example of how to do this on Puhti.

Once you are familiar with the files, you can simply compile the code with:
```
module load hip
make
```

Once compiled, you can run the code and see if it produces the correct
result (` Max error:    0.00000000`).


## Optional: use hipfc script

Instead of using the Makefile, one can also use a wrapper script provided by
hipfort called `hipfc`. The script calls a Fortran compiler with hipfort
options for Fortran files and `hipcc` for non-Fortran files. It could be used
to compile the code in one go, but unfortunately the current ROCm installation
on Puhti doesn't work with the script.

File [hipfc-puhti.patch](hipfc-puhti.patch) contains a patch that can be
applied to get a slightly modified version of the script that works also on
Puhti. You can try it out with:
```
patch -o hipfc $ROCM_PATH/hipfort/bin/hipfc hipfc-puhti.patch
./hipfc hipsaxpy.cpp main.f03 -o saxpy
```


## Original CUDA Fortran code

For reference, file [cuda-fortran/main.cuf](cuda-fortran/main.cuf) contains
the original CUDA Fortran code that was manually ported to HIP.
