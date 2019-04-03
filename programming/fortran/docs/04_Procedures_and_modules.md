# Procedures and modules

# Outline

- Structured programming
- Modules
- Procedures: functions and subroutines
- Interfaces

# Structured programming

- Structured programming based on program sub-units (*functions*,
  *subroutines* and *modules*) enables
    - Testing and debugging separately
    - Re-use of code
    - Improved readability
    - Re-occurring tasks
- The key to success is in well defined data structures and scoping, which
  lead to clean procedure interfaces

# Modular programming

- Modularity means dividing a program into minimally dependent *modules*
    - Enables division of the program into smaller self-contained units
- Fortran modules enable
    - Global definitions of procedures, variables and constants
    - Compilation-time *error checking*
    - Hiding *implementation details*
    - *Grouping* routines and data structures
    - Defining *generic procedures* and custom operators

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

```
subroutine sub(arg1, arg2, ...)
  [declarations]
  [statements]
end subroutine sub**
```
Use as:

`call sub(arg1, arg2,...)`
</div>
<div class="column">
**Function**

Declaration:

```
[type] function func(arg1, arg2, ...) & 
        & [result(val)]
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
subroutine dist(x, y, d)
  implicit none
  real :: x, y, d
  d = sqrt(x**2 + y**2)
end subroutine dist

program do_something
  ...
  call dist(x, y, r)
  ...
```
</div>
<div class="column">
```fortran
real function dist(x, y)
  implicit none
  real :: x, y
  dist = sqrt(x**2 + y**2)
end function dist

program do_something
  ...
  r = dist(x, y)
  ...
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

- In **foo1** variable **i** starts always from 0 and gets value 1
- In **foo2** variable **i** gets values 1, 2, 3, … in each successive
  call

# Should I use subroutine or function?

- Main difference is that functions can be used in expressions: **`r =
  dist(x1, y1) + dist(x2, y2)`**
- Recommendation as good programming practice:
    - Use functions for computing value based on input
    - Use subroutines for performing operation that changes some of the
      inputs

# Procedure types, modules

# Procedure types

- There are four procedure types in Fortran 90: *intrinsic, external*,
  *internal* and *module* procedures
- Procedure types differ in
    - Scoping, i.e. what data and other procedures a procedure can access
    - Interface type, explicit or implicit
- Compiler can check the argument types of the at compile time only if the
  interface is explicit!

# Implicit and explicit procedure types

- The interfaces of the intrinsic, internal and module procedures are
  *explicit*
- The interfaces of the external procedures, such as many library
  subroutines, are *implicit*. You can write an explicit interface to
  those, though.
- Intrinsic procedures are the procedures defined by the programming
  language itself, such as **INTRINSIC SIN**

# Module procedures and variables

<div class="column">
**Declaration**
```fortran
MODULE check
  IMPLICIT NONE
  INTEGER, PARAMETER :: &
    longint = SELECTED_INT_KIND(8)
CONTAINS
  FUNCTION check_this(x) RESULT(z)
   INTEGER(longint):: x, z
   ...
  END FUNCTION
END MODULE check
```
</div>
<div class="column">
**Usage**
```fortran
PROGRAM testprog
  USE check
  IMPLICIT NONE
  INTEGER(KIND=longint) :: x, test
  test = check_this(x)
END PROGRAM testprog
```
</div>

# Visibility of module objects

- Variables and procedures in *modules* can be **PRIVATE** or **PUBLIC**

    - **PUBLIC** visible for all program units using the module (the
    default)
    - **PRIVATE** will hide the objects from other program units 
    

``` fortran
REAL :: x,y 
PRIVATE :: x 
PUBLIC :: y ! Or 
REAL, PRIVATE :: x 
REAL, PUBLIC :: y
```

# Internal procedures

- Each program unit (program/subroutine/function) may contain internal
  procedures 
  
  
``` fortran
SUBROUTINE mySubroutine 
  ... 
  CALL myInternalSubroutine
  ... 
CONTAINS 
  SUBROUTINE myInternalSubroutine
    ... 
  END SUBROUTINE myInternalSubroutine
END SUBROUTINE mySubroutine
```

# Internal procedures

- Declared at the end of a program unit after the **CONTAINS** statement
    - Nested **CONTAINS** statements are not allowed
- Variable scoping:
    - Parent unit’s variables and objects are accessible
    - Parent unit’s variables are overlapped by local variables with the
      same name
- Can be called only from declaring program unit
- Often used for ”small and local, convenience” procedures

# Internal procedures: example

``` fortran
SUBROUTINE parent()
  IMPLICIT NONE
  INTEGER :: i,j
  i = 1; j = 1
  CALL child()
  ! After subroutine call i = 2 and j = 1
CONTAINS
  SUBROUTINE child()
    IMPLICIT NONE
    INTEGER :: j
    i = i + 1 ! Variable i is from the scope of parent
    j = 0 ! Variable j has local scope
  END SUBROUTINE child
END SUBROUTINE parent
```

# External procedures

- Declared in a separate program unit
    - Referred to with the **EXTERNAL** keyword
    - Compiled separately and linked to the final executable
- Avoid using them within a program, module procedures provide much
  better compile time error checking
- External procedures are often needed when using
    - procedures written with different programming language
    - library routines (e.g. BLAS and LAPACK libraries)
    - old F77 subroutines

# Interfaces

- For external procedures, interfaces determine the type and
  properties of arguments and return values
- Defined by an **INTERFACE** block: interface *interface-body* **end
  interface**
- The ***interface-body*** matches the subprogram header
    - position, rank and type of arguments
    - return value type and rank (for functions)

# Interfaces

<div class="column">
``` fortran
INTERFACE
  SUBROUTINE not_dangerous(a, b, c)
    INTEGER :: a, b, c
  END SUBROUTINE not_dangerous
END INTERFACE
INTEGER :: x, y, z
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
- Wrong calling arguments to **EXTERNAL** procedures may lead to
  errors during the executable linking phase or even when the
  executable is being run
- It is highly recommended to construct **INTERFACE** blocks for any
  external procedures used
</div>

# Interfaces: example

``` fortran
! LU decomposition from LAPACK
INTERFACE
  SUBROUTINE DGETRF(M, N, A, LDA, IPIV, INFO)
    INTEGER :: INFO, LDA, M, N
    INTEGER:: IPIV(*)
    DOUBLE PRECISION :: A(LDA,*)
  END SUBROUTINE DGETRF
END INTERFACE
! Euclidean norm from BLAS
INTERFACE
  FUNCTION DNRM2(N, X, INCX)
    INTEGER :: N, INCX
    DOUBLE PRECISION :: X(*)
    DOUBLE PRECISION :: DNRM2
  END FUNCTION DNRM2
END INTERFACE
```


# Global data and global variables

- Global variables can be accessed from any program unit

- Module variables with **SAVE** attribute provide controllable way to define
  and use global variables
``` fortran
MODULE commons 
  INTEGER, PARAMETER :: r = 0.42
  INTEGER, SAVE :: n, ntot
  REAL, SAVE :: abstol, reltol
END MODULE commons
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
