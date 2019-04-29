---
title: Procedures and modules
lang:  en
---

# Outline

- Procedures: functions and subroutines
- Modules
- Interfaces

# What are procedures?

- Procedure is block of code that can be called from other code.
- Calling code passes data to procedure via arguments
- Fortran has two types of procedures: *subroutines* and *functions*
    - Subroutines pass data back via arguments
    - Functions return a value

# Procedure declarations

<div class="column">
**Subroutine**

Declaration:

```fortran
subroutine sub(arg1, arg2, ...)
  [declarations]
  [statements]

end subroutine sub
```
Use as:

`call sub(arg1, arg2,...)`
</div>
<div class="column">
**Function**

Declaration:

```fortran
[type] function func(arg1, arg2, ...) & 
       &  [result(val)]
  [declarations]
  [statements]
end function func
```

Use as:

`res = func(arg1, arg2, ...)`
</div>

# Procedure declarations: example

<div class="column">
```fortran
program do_something
  ...
  call dist(x, y, r)
  ...

  contains  
    subroutine dist(x, y, d)
      implicit none
      real :: x, y, d
      d = sqrt(x**2 + y**2)
      end subroutine dist
end program
```
</div>
<div class="column">
```fortran
program do_something
  ...
  r = dist(x, y)
  ...

  contains
    real function dist(x, y)
      implicit none
      real :: x, y
      dist = sqrt(x**2 + y**2)
    end function dist
end program
```
</div>

# Procedure arguments

- Fortran passes call arguments *by reference*
    - Only the memory addresses of the arguments are passed to the called
      procedure
    - Any change to the value of an argument changes the value at the
      calling program
    - The *INTENT* attribute can be used to specify how argument is used

# INTENT attribute

<div class="column">
```fortran
subroutine foo(x, y, z)
  implicit none
  real, intent(in) :: x
  real, intent(inout) :: y
  real, intent(out) :: z
  x = 10 ! compilation error
  y = 10 ! correct
  z = y * x ! correct
end subroutine foo
```
</div>
<div class="column">
- Declares how formal argument is intended to be used for transferring a
value
    - **IN**: the value of the argument is read-only i.e. cannot be
      changed
    - **OUT**: the value of the argument must be provided
    - **INOUT** (the default)
- Compiler uses INTENT for error checking and optimization
- Improves readability of code


# Local variables in procedures

- Local variables can be declared in the procedure
- Local variables are not visible outside the procedure
- By default, local variables do not retain their values through
  successive calls of the procedure

```fortran
subroutine foo(x, y)
implicit none
real, intent(in) :: x
real, intent(out) :: y
integer :: i ! Local variable
...
```

# Local variables in procedures

- If local variable is given **SAVE** attribute, its value is retained
  through successive calls
- Initialization in declaration is done only in the first call to
  procedure, and implicit SAVE is applied

<div class="column">
```fortran
subroutine foo1(x)
  ...
  integer :: i
  i = 0
  i = i + 1
```
</div>
<div class="column">
```fortran
subroutine foo2(x)
  ...
  integer :: i = 0
  i = i + 1
```
</div>

- In `foo1` variable **i** starts always from 0 and gets value 1
- In `foo2` variable **i** gets values 1, 2, 3, … in each successive
  call

# Should I use subroutine or function?

- Main difference is that functions can be used in expressions: 

```fortran
r = dist(x1, y1) + dist(x2, y2)
```

- Recommendation as good programming practice:
    - Use functions for computing value based on input
    - Use subroutines for performing operation that changes some of the
      inputs

# Modular programming

- By now, we have used *internal* procedures and the whole application is
  implemented in a single file.
- Larger applications should be divided into small, minimally dependent *modules*
    - Aim is to build complex behaviour from simple self-contained components
    - Modules can be tested and debugged separately
    - Modules enable easier re-use of code

# Fortran modules

- For most use cases, procedures should be implemented in Fortran modules
- Modules can contains also variables, constants and data structure definitions
- Fortran modules enable
    - Hiding *implementation details*
    - *Grouping* routines and data structures
    - Defining *generic procedures* and custom operators

# Module defition and usage

- Module is defined with the `MODULE` keyword and used from main program 
  or other module with the `USE` keyword
- Depending on the complexity of module, one file can contain a single or 
  multiple module definitions
    - Only related modules should be grouped into the same file

# Module example

<div class="column">
**Declaration**
```fortran
module geometry
  implicit none
  real, parameter :: pi = 3.14

  contains
    real function dist(x, y)
      implicit none
      real :: x, y
      dist = sqrt(x**2 + y**2)
    end function dis

end module geometry
```
</div>
<div class="column">
**Usage**
```fortran
program testprog
  use geometry
  implicit none
  real :: x=1.2 + pi, y=4.5, d
  d = dist(x, y)
end program testprog
```
```fortran
module testmod
  use geometry
  contains
    function test_func()
      implicit none
      real :: x=-2.0, y=3.3
      test_func = dist(x, y) + pi
    end function test_func
    ...
```
</div>

# Building modules

- When working with modules, each source file needs to be compiled separately

```console
$ gfortran -c mymod.f90
$ gfortran -c myprog.f90
```
- When compiling the module, a `.mod` is produced for each module defined in `mymod.f90`.
- When compuling the main program compiler aborts with error if `.mod` is not found for each 
  module **use**d
- In order to produce executable the object files of module and main program need to be 
  linked
```console
$ gfortran -o myexe mymod.o myprog.o
```


# Visibility of module objects

- Variables and procedures in *modules* can be **PRIVATE** or **PUBLIC**
    - **PUBLIC** visible for all program units using the module (the
    default)
    - **PRIVATE** will hide the objects from other program units 
    

``` fortran
module visibility
  real :: x,y 
  real :: x 
  public :: y ! Or 
  real, private :: x 
  real, public :: y
end module
```

# Other procedure types

- In addition to *internal* and *module* procedures Fortran has *intrinsic* and 
  *external* procedures
- Intrinsic procedures are the procedures defined by the programming
  language itself, such as **SIN**
- External procedures should nowadays be avoided
    - can be needed when working with libraries (e.g BLAS and LAPACK) or with old 
      F77 code
    - Compiler cannot check the argument types without explicit **interface** block

# Internal procedures

- Each program unit (program/subroutine/function) may contain internal
  procedures   
  
``` fortran
subroutine mySubroutine 
  ... 
  call myInternalSubroutine
  ... 
contains
  subroutine myInternalSubroutine
    ... 
  end subroutine myInternalSubroutine
end subroutine mySubroutine
```

# Internal procedures

- Declared at the end of a program unit after the **contains** statement
    - Nested **contains** statements are not allowed
- Variable scoping:
    - Parent unit’s variables and objects are accessible
    - Parent unit’s variables are overlapped by local variables with the
      same name
- Can be called only from declaring program unit
- Often used for ”small and local, convenience” procedures

# Interfaces

- For external procedures, compiler cannot check the type and
  properties of arguments and return values 
- An explicit **interface** block can be defined to allow compiler checks
```fortran
interface
  interface-body
end interface
```
- The ***interface-body*** matches the procedure header
    - position, rank and type of arguments
    - return value type and rank (for functions)

# Interfaces

<div class="column">
``` fortran
interface
  subroutine not_dangerous(a, b, c)
    integer :: a, b, c
  end subroutine not_dangerous
end interface
integer :: x, y, z
x=1; y=1; z=1
! Call external subroutine without
! an interface
call dangerous(x,y,z)
! Call external subroutine with
! an interface
call not_dangerous(x,y,z)
```
</div>
<div class="column">
- Wrong calling arguments to **external** procedures may lead to
  errors during the executable linking phase or even when the
  executable is being run
- It is highly recommended to construct **interface** blocks for any
  external procedures used
</div>

# Interfaces: example

``` fortran
! LU decomposition from LAPACK
interface
  subroutine dgetrf(M, N, A, LDA, IPIV, INFO)
    integer :: INFO, LDA, M, N
    integer:: IPIV(*)
    double precision :: A(LDA,*)
  end subroutine dgetrf
end interface
! Euclidean norm from BLAS
interface
  function dnrm2(N, X, INCX)
    integer :: N, INCX
    double precision :: X(*)
    double precision :: dnrm2
  end function dnrm2
end interface
```


# Global data and global variables

- Global variables can be accessed from any program unit

- Module variables with **save** attribute provide controllable way to define
  and use global variables
``` fortran
module commons 
  integer, parameter :: r = 0.42
  integer, save :: n, ntot
  real, save :: abstol, reltol
end module commons
```
- Explicit interface: type checking, limited scope
- Generally, use of global variables is not recommended

# Summary

- Procedural programming makes the code more readable and easier to
  develop
    - Procedures encapsulate some piece of work that makes sense and may
      be worth re-using elsewhere
- Fortran uses *functions* and *subroutines*
    - Values of procedure arguments may be changed upon calling the
      procedure
- Fortran *modules* are used for modular programming and data
  encapsulation
