Compile the code first with `-DNDEBUG` flag, e.g.
```bash
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

CC -xhip -DNDEBUG -O3 main.cpp -o main
```
then run it with srun.

You should get errors and a suggestion to remove the `-DNDEBUG` flag.

Compile the code again, this time without `-DNDEBUG` and run it.

Try to fix errors as you encounter them. The error message printed to the terminal should point you in the right direction. Remember to recompile the code every time you change something!
