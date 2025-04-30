---
title:  Parallel debugging
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Parallel debugging {.section}

# Debugging

- Bugs are evident in any non-trivial program
- Crashes (Segmentation fault) or incorrect results
- Parallel programs can have also deadlocks and race conditions
    - <https://deadlockempire.github.io>

# Finding bugs

- Building code with Sanitizer
    - Detects out-of-bounds memory access and OpenMP race conditions
    - `-fsanitize=address` (GCC / Clang)
    - `-fsanitize=thread` (GCC / Clang)
- Memory debugging with `valgrind`
   - Detects bad memory accesses, memory leaks, ...
   - `valgrind4hpc` for parallel applications
- Using MPI correctness checkers
    - MUST
    - Intel Trace Analyzer

# Finding bugs

- Print statements in the code
    - Typically cumbersome, especially with compiled languages
    - Might result in lots of clutter in parallel programs
    - Order of printouts from different processes is arbitrary
- "Standard" debuggers
    - `gdb`: common command line debugger
    - Debuggers within IDEs, *e.g.* VS Code
    - No proper support for parallel debugging
- Parallel debuggers
    - Linaro DDT (formerly Arm DDT, Allinea DDT), Totalview, gdb4hpc (commercial products)

# Common features in debuggers

- Setting breakpoints and watchpoints
- Executing code line by line
- Stepping into / out from functions
- Investigating values of variables
- Investigating program stack
- Parallel debuggers allow all of the above on per process/thread basis

# Web resources

- Defensive programming and debugging online course <https://www.futurelearn.com/courses/defensive-programming-and-debugging>
- MUST <https://www.i12.rwth-aachen.de/go/id/nrbe>
- Using `gdb` for parallel debugging <https://www.open-mpi.org/faq/?category=debugging>
- Memory debugging with Valgrind <https://valgrind.org/docs/manual/mc-manual.html#mc-manual.mpiwrap>
