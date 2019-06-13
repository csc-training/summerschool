---
title: "Debugging"
lang: en
---

# What is debugging?

- Searching for programming mistakes (bugs) that cause wrong program
  behavior
- Note that also dependencies (libraries) can have bugs
- Different approaches
    - Compiler checks
    - Source code reviews
    - Source code modifications, adding extra output or logs
    - *Debuggers*

# Compiler checks

- Most compilers have some additional checks that can also be used for
  debugging
    - Usually not enabled for production because of memory consumption
      or performance overhead
- Two analysis types:
    - Static analysis
        - Diagnostics during compilation
    - Runtime analysis:
        - Compiler adds extra code that checks e.g. array indices and
          memory access

# Compiler check examples

- Using GCC address sanitizer to add run-time checks for invalid
  memory references
```console
gcc -g -fsanitize=address myprog.c
```
- Add array bounds run-time checks to Fortran program
```console
gfortran -g -fbounds -check myprog.f90
ftn -g -h bounds myprog.f90
ifort -g -check bounds myprog.f90
```
- Check for accessing uninitialized variables
```console
icc -check uninit myprog.c
```

# Compiling for debugging

- There is no one-to-one mapping from C or Fortran to machine language
    - For example, variables are mapped to registers and memory
      locations
- By default the information that is needed for interpreting the
  program state back to a corresponding source code is not available
    - Can be added to the program code during compilation
    - Combining debugging information with optimization is problematic

# Compiling cont.

- Compiler option that adds the debug information depends on the
  compiler, but usually adding option `-g` is enough
- Optimizations and debugging do not mix
    - Optimizations can change the program logic so that mapping from
      source to program code is not even possible
    - Compilers may generate faulty code with aggressive optimization
      flags
    - Finding bugs that are caused by compiler optimizations are often
      difficult to find

# Optimizations and debugging

- GCC has special optimization flag that can be used together with
  debugging, `-Og`
    - Otherwise same as optimization level O1, but does not enable
      optimizations that make debugging problematic
- Cray toolset has a fast-track debugging, that enables debugging of
  fully optimized code
    - Compile with *-Gfast*
    - Requires support from both compiler and debugger
  
# Debuggers {.section}

# What are debuggers?
- Debugger is a program that can be used to inspect other programs
    - Debuggers can stop program execution
        - Step-by-step execution is also possible
    - They can show (and set) values of variables
    - Machine language code is also accessible with low-level
      information on registers e.g.

# Stopping execution, breakpoint

- Breakpoints are specified source code locations where the debugger
  stops the program execution so that the state can be inspected
    - Usually the breakpoints are set so that program is stopped before
      the wrong behavior occurs
- Breakpoints can be conditional, that is, they can depend on values
  of variables etc.

# Stack frame and backtrace

- When a function is called, some of the arguments are stored in
  registers and other are put into the stack
    - Also local variables of the function are reserved from the stack
- It is possible to move between stack frames and check the values of
  the calling scope
- Backtrace is the full listing of all stack frames
    - Important information on who actually called the current function

# Core files

- OS can be instructed to store the program memory space to disk if
  the program crashes
    - Often not enabled by default
    - Parallel programs can easily generate terabytes of data
- If the program code is available with debugging symbols together
  with memory dump one can inspect the program state at the time of
  crash

# GDB

- GNU project debugger that is very widely available
- Supports multiple programming languages
- Support for many operating systems
- Command line tool, but many GUIs are available

# Parallel debugging

- Parallel debugging is much more difficult than debugging serial
  programs
    - Many bugs are due to synchronization issues and may appear
      randomly
    - Often bugs appear when running with large number of tasks
        - Inspecting the status of thousands of tasks is challenging
    - Debuggers have to be integrated to the batch job system and/or MPI
      process launcher and there are many different alternatives

# What to do when MPI program crashes?

- Try to get as much statistics as possible
    - Backtrace is very useful, use compiler options or other tools to
      generate it
- If not all tasks crash, and the program memory usage is not too
  large, try to generate some core files and inspect them using
  e.g. gdb
- Use a parallel debugger!

# Parallel debuggers

- Unfortunately there are no good open-source parallel debuggers
  available
    - Eclipse Parallel Tools Platform, not a stand-alone debugger, very
      complicated installation
    - PGDB, command line debugger on top of gdb, project is not active
- There are few commercial parallel debuggers
    - ARM (prev. Allinea) DDT
    - RogueWave Totalview

# Available tools at CSC

- Taito has an old version of RogueWave's Totalview
- Puhti and upcomping Mahti have ARM DDT

# Parallel debugging demo

