# Modules and scoping{.section}

# Outline

- Fortran modules
- Module procedures
- Data scoping

# Modular programming

- By now, we have implemented the whole application in a single file
- Larger applications should be divided into small, minimally dependent *modules*
    - Aim is to build complex behaviour from simple self-contained components
    - Modules can be tested and debugged separately
    - Modules enable easier re-use of code

# Fortran modules

- Module can contain procedures, variables, constants and data structure definitions
- Fortran modules enable
    - Hiding *implementation details*
    - *Grouping* routines and data structures
    - Defining *generic procedures* and custom operators

# Module definition and usage

- Module is defined with the `module` keyword and used from main program 
  or other modules with the `use` keyword
- Depending on the complexity of module, one file can contain a single or 
  multiple module definitions
    - Only related modules should be grouped into the same file

# A simple module example

<div class="column">
**Module definition**
```fortran
module geometry
  implicit none

  real, parameter :: pi = 3.14
end module geometry
```
</div>
<div class="column">
**Usage**
```{.fortran emphasize=2-2,6:17-6:18} 
program testprog 
  use geometry
  implicit none

  real :: y  
  y = sin(1.2 * pi)
end program testprog
```
```{.fortran emphasize=2-2,5:32-5:33} 
module testmod
  use geometry
  implicit none

  real, parameter :: twopi = 2*pi
```
</div>

# Building modules

- Each source file needs to be compiled separately

```console
$ gfortran -c mymod.f90
$ gfortran -c myprog.f90
```
- When compiling the module, a `.mod` is produced for each module defined in `mymod.f90`.
- When compiling the main program compiler aborts with error if `.mod` is not found for each 
  module **use**d
- In order to produce executable, the object files of module and main program need to be 
  linked

```console
$ gfortran -o myexe mymod.o myprog.o
```

# Building modules

- Normally **make** (or similar build system) is utilized when working with
  multiple source files

```console
$ make
gfortran -c mymod.f90
gfortran -c myprog.f90
gfortran -o myexe mymod.o myprog.o
```

- By defining dependencies between program units only those units that are 
  affected by changes need to be compiled

```console
$ emacs myprog.f90
$ make
gfortran -c myprog.f90
gfortran -o myexe mymod.o myprog.o
```

# Defining procedures in modules

- In most cases, procedures should be defined in modules
- Procedures are defined after **contains** keyword

<div class="column">
Function definition in module
```{.fortran emphasize=5:3-5:10,6:10-6:22}
module geometry
  implicit none
  real, parameter :: pi = 3.14

  contains
    real function dist(x, y)
      implicit none
      real :: x, y
      dist = sqrt(x**2 + y**2)
    end function dist

end module geometry
```
</div>
<div class="column">
Usage
```{.fortran emphasize=2:3-2:14,6:7-6:10}
program testprog 
  use geometry
  implicit none

  real :: d
  d = dist(2.0, 3.4)

end program testprog
```
</div>

# Defining procedures

- In most cases, procedures should be defined in modules
- Procedures are defined after **contains** keyword

<div class="column">
Subroutine definition in module
```{.fortran emphasize=5:3-5:10,6:5-6:19}
module geometry
  implicit none
  real, parameter :: pi = 3.14

  contains
    subroutine dist(x, y, d)
      implicit none
      real :: x, y, d
      d = sqrt(x**2 + y**2)
    end subroutine dist

end module geometry
```
</div>
<div class="column">
Usage
```{.fortran emphasize=2:3-2:14,6:3-6:11}
program testprog 
  use geometry
  implicit none

  real :: d
  call dist(2.0, 3.4, d)

end program testprog
```
</div>


# Procedure definitions

- Formally, subroutines and functions are defined and used as:

<div class="column">
**Subroutine**

Definition:

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

Definition:

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


# Procedure arguments

- Fortran passes call arguments *by reference*
    - Only the memory addresses of the arguments are passed to the called
      procedure
    - Any change to the value of an argument changes the value at the
      calling program
          - Procedures can have *side-effects*
    - The *intent* attribute can be used to specify how argument is used

# Intent attribute

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
    - **in**: the value of the argument is read-only i.e. cannot be
      changed
    - **out**: the value of the argument must be provided
    - **inout** (the default)
- Compiler uses **intent** for error checking and optimization
- Improves readability of code

# Should I use subroutine or function?

- Main difference is that functions can be used in expressions: 

```fortran
r = dist(x1, y1) + dist(x2, y2)
```

- Recommendation as good programming practice:
    - Use functions for computing value based on input
    - Use subroutines for performing operation that changes some of the
      inputs

# Scoping of variables

- Scope is the region of the program where a particular variable is defined and can be used
- A variable with the same name can be defined in multiple scopes and have different value in them
- In Fortran, variables are normally only available within the program unit that defines them

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

# Variables in modules

- Variables declared in the module definition section are global
    - Can be accessed from any program unit that **use**s the module.
- Modifications in one program unit are seen also elsewhere

<div class="column">
``` fortran
module globals 
  integer :: var
end module commons
...
subroutine foo()
  use globals
  var = 47
end subroutine
```
</div>
<div class="column">
``` fortran
...
subroutine bar()
  use globals
  write(*,*) var 
end subroutine
...
program myprog
...
  call foo()   ! var is modified
  call bar()   ! The new value is written out
```
</div>

- Generally, use of global variables is not recommended

# Limiting visibility of module objects

- Variables and procedures in *modules* can be **private** or **public**
    - **public** visible for all program units using the module (the
    default)
    - **private** will hide the objects from other program units 
    

``` fortran
module visibility
  real :: x,y 
  private :: x
  public :: y  ! public declaration is not in principle needed but can improve readability
  real, private :: z ! Visibility can be declared together with type

  private :: foo ! foo can be called only inside this module

  contains
    subroutine foo()
      ...
  
end module
```

# Other procedure types

- In addition to *intrinsic* and *module* procedures Fortran has *internal* and 
  *external* procedures
  - External procedures should nowadays be avoided
    - can be needed when working with libraries (e.g BLAS and LAPACK) or with old 
      F77 code
   

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



# Summary

- Diving program into modules makes it easer to develop and debug
- Fortran modules can contain procedures, variables and type definitions
- Procedures should in most cases be implemented in modules
- By default, Fortran procedures can modify arguments
    - intent attributes can be used for avoiding unwanted side effects
