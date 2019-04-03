# Advanced features in Fortran

# Outline

- Language interoperability
- Object oriented programming
- Fortran coarrays

# Language interoperability issues

- The problem of language interoperability has been present for a long
  time
    - Before Fortran 2003 the standard did not specify the calling
      convention in detail
- Interoperability is becoming even more important than before
    - Steering of computation with scripting languages
    - Utilizing libraries that are written using other programming
      languages

# Before Fortran 2003

- Compilers often, but not always
    - Passed all procedure arguments by-reference (i.e. by-address)
    - Referred to functions and subroutines in lowercase letters and by adding
      an additional underscore after, e.g. **`Func`** becomes **`func_`** to be
      found by the linker
    - Passed function return values via stack
    - Passed **`character`** strings by-reference, with an additional hidden
      length argument, passed by value.
    - Note: Fortran **`character`** strings are not null-character
      terminated

# Portability

- The traditional way to have interoperability with C requires *a
  priori* knowledge of lowercase and underscore policy used by the
  compiler
- Complex cases, such as passing **character** strings or passing
  arguments by value, are generally very error prone and may lead to
  catastrophic errors at runtime
- Often a separate layer of C functions was needed for
  interoperability

# The iso_c_binding module

- Fortran 2003 intrinsic module **`iso_c_binding`** is used with **`use,
  intrinsic :: iso_c_binding`**
- Module contains
    - Access to named constants that represent **kind** type parameters of
      data representations compatible with C types
    - The derived types **`c_ptr`** and **`c_funptr`** corresponding to C
      pointer and C function pointer types, respectively
    - Useful procedures: **`c_loc`**, **`c_funloc`**, **`c_f_pointer`**,
    **`c_associated`**, **`c_f_funpointer`**, **`c_sizeof`**
        (f08)

# Interoperability with Fortran 2003: example

``` fortran
function call_c_func(data, nx, ny) result(stat)
  use, intrinsic :: ISO_C_BINDING
  implicit none
  real(dp), dimension(:,:), intent(in) :: data
  integer, intent(in) :: nx, ny
  integer :: stat
  interface
     ! C-definition is: int func(double *data, const int nx, const int ny)
     function c_func(data, nx, ny) bind(C,name="func") result(stat)
       use, intrinsic :: ISO_C_BINDING
       implicit none
       real(kind=C_DOUBLE) :: data(*)
       integer(kind=C_INT), value, intent(IN) :: nx, ny
       integer(kind=C_INT) :: stat
     end function c_func
  end interface
  stat = c_func(data, nx, ny)
end function call_c_func
```

# Object oriented programming

- Object Oriented Programming (OOP) is programming paradigm where
    - Data and functionality are wrapped inside of an “object”
    - Objects provide methods which operate on (the data of) the object
    - Method is a procedure that is tied to the data of an object
    - Objects can be inherited from other “parent” objects
- Object oriented programming in Fortran facilitated with type extensions
  and type- and object-bound procedures

# Fortran coarrays

- Parallel processing as part of Fortran language standard
    - Only small changes required to convert existing Fortran code to
      support a robust and potentially efficient parallelism
- A Partitioned Global Address Space (PGAS) paradigm
    - Parallelism implemented over “distributed shared memory”
- Integrated into Fortran 2008 standard
    - Compiler support is still incomplete (Cray: excellent, Intel:
      moderate, GNU: experimental)

# Further advance features

- Abstract interfaces and procedure pointers
    - Declaring several procedures with same interface
- Asynchronous I/O
- Submodules
    - Implementation of module can be split into multiple files

# Summary

- Fortran is a modern language that continues to evolve
    - Support for object-oriented programming
    - Complex, generic data structures can be implemented in modern
      Fortran
    - Coarrays provide parallel programming support in the language itself
- Fortran remains as one of the major languages for scientific computation
