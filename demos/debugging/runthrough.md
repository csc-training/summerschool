- [Introduction](#orgbdf3d09)
  - [Debugging with gdb](#orgf5df9b4)
    - [gdb](#orgf5e84b8)
    - [Compiling code for debugging](#org35ac802)
    - [Starting the debugger](#orgd48ddbd)
    - [Gdb commands](#org4093bde)
  - [Summary of gdb commands](#org5c3c966)
- [Using compiler tools to detect out-of bounds access](#orgc0d4988)
  - [Fortran](#org69a8397)
  - [C](#orgd5516e5)



<a id="orgbdf3d09"></a>

# Introduction

This is a short walk-through example of debugging an illegal memory access, which is a common mistake in Fortran and especially C programs.


<a id="orgf5df9b4"></a>

## Debugging with gdb


<a id="orgf5e84b8"></a>

### gdb

gdb is a command line debugger, but there are many GUIs that can be used with it. Here we concentrate on the basic command line usage for maximum portability. Command line version of gdb is very widely available on HPC systems.


<a id="org35ac802"></a>

### Compiling code for debugging

Prepare the example program for debugging and add the debug symbols. It can be done by using a compiler flag `-g` before any optimization flags. Note that any higher optimization levels will interfere with debugging. If possible, you should omit all optimizations. For gnu compilers you can use optimization flat `-Og` together with `-g` to enable optimizations that do not interfere with debugging.

Example compilation:

```sh
gcc -Og -g -o debug_demo debug_demo.c
# or
gfortran -Og -g -o debug_demo debug_demo.f90
```

Adding debug symbols to the program code increases the binary size. Some compilers also may disable some of the optimizations when debugging flag is used together with optimization flags. On the other hand, having at least some debugging information (e.g. function names) can help if production code crashes.

Source code for this example can be found in [runthrough](runthrough) directory. You can compile the code using `make` command.


<a id="orgd48ddbd"></a>

### Starting the debugger

Debugger can be used to launch the program or user can attach the debugger to a running program. The latter option is useful for inspecting problems that cause random hangups. For this tutorial we will start the programs using the debugger.

Start your debugging session by launching the gdb with program name as argument:

    [user@system src]$ gdb ./heat_serial 
    GNU gdb (GDB) Fedora 8.0.1-36.fc27
    Copyright (C) 2017 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
    and "show warranty" for details.
    This GDB was configured as "x86_64-redhat-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.
    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from ./heat_serial...done.
    (gdb)

Program itself is not yet started. If we issue `run` command, the program will start and execute all the way to the segmentation fault. Segmentation fault is an error where program is trying to access memory segment that is not part of the address space of the program. Operating system raises the signal and program execution is stopped.

    (gdb) run
    Starting program: /path/to/Courses/debug/src/heat_serial 
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.26-27.fc27.x86_64
    
    Program received signal SIGSEGV, Segmentation fault.
    evolve (curr=curr@entry=0x7fffffffd1d0, prev=prev@entry=0x7fffffffd1b0, a=a@entry=0.5, 
        dt=dt@entry=4.9999999999999996e-05) at core.c:25
    25	                                 2.0 * prev->data[i][j] +
    Missing separate debuginfos, use: dnf debuginfo-install libpng-1.6.31-1.fc27.x86_64 zlib-1.2.11-4.fc27.x86_64
    (gdb) 

When debugging information is available, the debugger can spot the actual source code line that crashed the program.


<a id="org4093bde"></a>

### Gdb commands

Now we have the crashed program, but how can we inspect the state of the program and find the root cause for the problem? Debugger can access the memory space of the program and thus it has all the needed information, including the call stack of the program and values of variables.

We can start by looking at the call stack (backtrace) using `bt` command:

    (gdb) bt
    #0  evolve (curr=curr@entry=0x7fffffffd1d0, prev=prev@entry=0x7fffffffd1b0, a=a@entry=0.5, 
        dt=dt@entry=4.9999999999999996e-05) at core.c:25
    #1  0x000000000040158c in main (argc=<optimized out>, argv=<optimized out>) at main.c:42

Output tells us that we are currently in `evolve` function that is defined in `core.c` file line 25. We can also see that the function was called from the main function, defined in `main.c`, line 42. We can move between stack frames with `up` and `down` commands, if necessary.

Values of arguments of current stack frame can be listed as follows

    (gdb) info args
    curr = 0x7fffffffd1d0
    prev = 0x7fffffffd1b0
    a = 0.5
    dt = 4.9999999999999996e-05

If we look at the source code, that seems to be in line with the function definition. First two arguments are pointers and last two are doubles.

We can also check the values with print command (shortcut p). For example:

    (gdb) p curr->data[0][0]
    $1 = 85

Values of local variables can be listed with `info locals` commmand:

    (gdb) info locals
    i = 0
    j = 0
    dx2 = 0.0001
    dy2 = 0.0001

These four variables match again with the local variable declarations of the `evolve` function.

Let's look at the program code with list (l) command:

    (gdb) l
    20	    dy2 = prev->dy * prev->dy;
    21	    for (i = 0; i < curr->nx + 1; i++) {
    22	        for (j = 0; j < curr->ny + 1; j++) {
    23	            curr->data[i][j] = prev->data[i][j] + a * dt *
    24	                               ((prev->data[i + 1][j] -
    25	                                 2.0 * prev->data[i][j] +
    26	                                 prev->data[i - 1][j]) / dx2 +
    27	                                (prev->data[i][j + 1] -
    28	                                 2.0 * prev->data[i][j] +
    29	                                 prev->data[i][j - 1]) / dy2);

We could also open the source code file in editor by issuing the `edit` command. Please note that the type of the editor depends on system and personal settings.

If you look at the line 25, which was the line that the debugger pointed, there seems to be no problems. But if you look at the line 26 you should be able to spot the mistake in indices and the root cause for the problem, which is on line 21.


<a id="org5c3c966"></a>

## Summary of gdb commands

Here is a short list of common gdb commands:

| Long         | Short    | Action                                    |
|------------ |-------- |----------------------------------------- |
| run          |          | Start the program (with arglist)          |
| backtrace    | bt       | Display program stack                     |
| break `loc`  |          | Set breakpoint for location `loc`         |
| print `expr` | p `expr` | Display the value of `expr`               |
| continue     | c        | Continue program execution (after break)  |
| next         | n        | Execute next program line                 |
| step         | s        | Execute next line, step into any function |
| finish       |          | Continue until end of current function    |

Info command has many subcommands, here are few of them:

| Subcommand  | Action                                              |
|----------- |--------------------------------------------------- |
| args        | Values of argument variables of current stack frame |
| breakpoints | Status of specified breakpoints                     |
| locals      | Values of local variables of current stack frame    |


<a id="orgc0d4988"></a>

# Using compiler tools to detect out-of bounds access


<a id="org69a8397"></a>

## Fortran

Most Fortran compilers support run-time array bounds checking. It is not turned on by default because of the performance overhead. Here is a short list of most common compilers in CSC environment and options to use:

| Compiler | Flag             |
|-------- |---------------- |
| gfortran | `-fbounds-check` |
| ifort    | `-check bounds`  |
| crayftn  | `-h bounds`      |

Please note that there are also other types of invalid memory referencing problems that the bounds checking can not catch. Some compilers have additional run time memory checkers that can catch more complicated bugs.


<a id="orgd5516e5"></a>

## C

Because C does not have any concept of an array built into the language, checking indexing mistakes is more complicated. Many compilers have some options to do run time checking. We present here only the *address sanitizer*, which is an open-source tool integrated to both llvm's c compiler clang and gcc.

If you have a recent gcc installed, you can try to compile and run the test program with address sanitizer. Here is a compilation example:

```sh
gcc -fsanitize=address -g -O -o debug_demo debug_demo.c
# or
gfortran -fsanitize=address -g -O -o debug_demo debug_demo.f90
```
