# Hello world with OpenACC

Compile and run a simple OpenACC test program, provided as `hello(.c|.F90)`.

In order to compile the program on taito-gpu, you'll need to first load the
following modules:
```bash
module load cuda/10.0 pgi/19.1 openmpi/3.1.4 libpng/1.6
```

After this, you can compile the program using the normal compiler wrapper
`mpicc`. Just remember to turn on support for OpenACC (`-acc` flag with PGI).
