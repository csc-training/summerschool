# Some further useful features in Fortran{.section}

# Outline

- Generic procedures
- Environment variables
- Executing commands

# Generic procedures

- Procedures which perform similar actions but for different data types
  can be defined as *generic procedures*
- Procedures are called using the *generic name* and compiler uses the
  correct procedure based on the argument number, type and dimensions
    - Generic name is defined in an interface section
- Can improve code readability and maintainability
    - Overuse can have the opposite effect

# Generic procedures example

<div class="column">

```fortran
module swapmod
  implicit none
  interface swap
     module procedure swap_real, swap_char
  end interface swap
 contains
  subroutine swap_real(a, b)
    real, intent(inout) :: a, b
    real :: temp
    temp = a; a = b; b = temp
  end subroutine swap_real
  subroutine swap_char(a, b)
    character, intent(inout) :: a, b
    character :: temp
    temp = a; a = b; b = temp
  end subroutine swap_char
end module swapmod
```

</div>
<div class="column">

``` fortran
program switch
  use swapmod
  implicit none
  character :: n,s
  real :: x,y
  n = 'j'
  s = 's'
  x = 10
  y = 20
  print *, x, y
  print *, n, s
  call swap(n, s)
  call swap(x, y)
  print *, x, y
  print *, n, s
end program switch
```

</div>

# Overloading operators

- Procedures can also be assigned to operators
    - For example **`*`**, **`+`**, **`-`**, **`==`**, **`<`**, **`>`**
    - Own operator names can be used too, for example **`.dot.`**
- Especially useful together with user-defined data types

# Overloading operators

- Instead of a procedure name, use operator or assignment keyword
    - Operator: 1 or 2 input arguments with intent(in)
    - Assignment: 2 input arguments with intent(inout) or intent(out) and
      intent(in)

```fortran
interface operator(+)
  module procedure base_sum
end interface operator(+)

function base_sum(x,y) result(z)
  type(base), intent(in) :: x,y
  type(base) :: z
  z % val = x % val + y % val
end function
```

# Operator overloading example

<div class="column">

``` fortran
module operator_demo
  implicit none
  type pair
     real :: a, b
  end type pair
  interface operator (<)
     module procedure is_smaller
  end interface operator (<)
contains
  logical function is_smaller(x, y)
    type(pair), intent(inout) :: x, y
    if (x%a < y%a .and. x%b < y%b) then
       is_smaller = .true.
    else
       is_smaller = .false.
    end if
  end subroutine
end module operator_demo
```

</div>
<div class="column">

``` fortran
program switch
  use operator_demo
  implicit none
  type(pair) :: p1, p2
  p1%a = 1.0; p1%b = 2.0
  p2%a = 3.0; p2%b = 4.0
  if (p1 < p2) then
     print *, ‘p1 is smaller’
  else
     print *, ‘p1 not smaller’
  end if
end program switch
```

</div>


# Environment variables

- Besides command line arguments, environment variables are a common
  way to modify program behaviour
- Fortran has a standardized method for accessing values of
  environment variables

# Environment variables

- Besides command line arguments, environment variables are a common
  way to modify program behaviour

``` fortran
character(len=256) :: enval
integer:: len,stat

! extract hostname
call get_environment_variable('HOSTNAME',enval,len,stat)
if (stat == 0) write (*,'(a,a)') 'host=', enval(1:len)

! extract user
call get_environment_variable('USER',enval,len,stat)
if (stat == 0) write (*,'(a,a)') 'user=', enval(1:len)
```

# Executing commands
    
- Invoking external programs from within a program would be useful
  occasionally
    - No source nor library API available for a useful program
    - Perl/Python/etc parsing scripts
- Fortran (2008) has a standardized method for invoking an external
  command

# Executing commands

- **`call execute_command_line(command[,wait][,exitstat][,cmdstat][,cmdmsg])`**
    - **`command`** is a character string containing the command to be
      invoked
    - **`wait`** is logical value indicating if command termination is to be
      waited (**`.true.`**, the default) or if the command is to be executed
      asynchronously (**`.false.`**) (optional)
    - **`exitstat`** is an integer value containing the return value of the
      command if **`wait=.true.`** (optional)
    - **`cmdstat`** is an integer value. It is assigned a value of zero if
      **`command`** executed successfully. For other return codes, see docs
      (optional)
    - **`cmdmsg`** is a character string containing explanatory message for
      positive values of **`cmdstat`** (optional)

# Executing commands: example

``` fortran
program execcommand
  implicit none
  integer :: estat, cstat
  ! execute a unix command
  call execute_command_line('ls -al', .true., estat, cstat)
  if (estat==0) write (*,'(a)') 'command completed successfully’
end program execcommand
```

# Summary

- Generic procedures
    - Same procedure name for multiple argument types
- Operator overloading
    - Can use operators for user-defined types
- Reading environment variables
- Executing system commands

