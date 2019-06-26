# Fortran arrays {.section}

# Outline

- Array syntax & array sections
- Dynamic memory allocation
- Array intrinsic functions
- Pointers to arrays

# Introduction to Fortran arrays

- Fortran is a very versatile programming language for handling arrays
    - especially multi-dimensional arrays
- Arrays refer to a data type (built-in or derived), but also have one
  or more dimensions specified in the variable declaration
    - Fortran supports up to 15 dimensions

# Recall: declaring arrays

```fortran
integer, parameter :: m = 100, n = 500
integer :: idx(m)
real :: vector(0:n-1)
real :: matrix(m, n)
character (len = 80) :: screen(24)
! or equivalently
integer, dimension(m) :: idx
real, dimension(0:n-1) :: vector
real, dimension(m, n) :: matrix
character(len=80), dimension(24) :: screen
```

By default, indexing starts from 1

# Array syntax

- Arrays can be accessed element-by-element
    - Normally, bounds are not checked so access outside declared bounds 
      results in program crash or garbage results
- Arrays can be treated also with *array syntax*

```fortran
do j = 0, 10
  vector(j) = 0
  idx(j) = j
end do
vector = 0
! or vector(:) = 0
vector(0:10) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
! or
idx(0:10) = [ (j, j = 0, 10) ] ! "implied do"
! the [...] syntax is equivalent to (/ ... /)
```

# Array syntax

Array syntax allows for less explicit do loops

<div class="column">
```fortran
integer :: m = 100, n = 200
real :: a(m,n) , x(n), y(m)
integer :: i , j
y = 0.0
outer_loop : do j = 1, n
   inner_loop : do i = 1, m
      y(i) = y(i) + a(i,j) * x(j)
   end do inner_loop
end do outer_loop
```
</div>
<div class="column">
```fortran
integer :: m = 100, n = 200
real :: a(m,n) , x(n), y(m)
integer :: j
y = 0.0
outer_loop : do j = 1, n
   y(:) = y(:) + a(:,j) * x(j)
end do outer_loop
```
</div>

# Array sections

- With Fortran array syntax we can access a part of an array in an
  intuitive way: *array sections* (”slices” in some other languages)
- When copying array sections, both left and right hand sides has to
  have *conforming dimensions*

```fortran
Sub_Vector(3:N) = 0
Every_Third(::3) = 1  ! (start:end:stride)
Diag_Block(i–1:i+1, j–2:j+2) = k
real :: a(1000, 1000)
a(2:500, 3:300:3) = 4.0
lhs(1:3, 0:9) = rhs(-2:0, 20:29) ! this is ok
! but here is an error :
lhs(1:2, 0:9) = rhs(-2:0, 20:29)
```

# Dynamic memory allocation

- Memory allocation is *static* if the array dimensions have been
  declared at compile time
- If the sizes of an array depends on the input to program, its memory
  should be *allocated* at runtime
    - memory allocation becomes *dynamic*

# Dynamic memory allocation

```fortran
integer :: alloc_stat
integer, allocatable :: idx(:)       ! The shapes (but not sizes) of the arrays need to be
real, allocatable :: mat(:,:)        ! known upon declaration
...
read (*,*) m, n
allocate (idx(0:m–1)) ! You can specify the start index, by default indexing starts from 1
allocate (mat(m,n), stat=alloc_stat) ! optional error code to check if allocation was succesful
if (alloc_stat /= 0) stop
...
deallocate (mat) ! Destroys the allocation (and the contents of the array)
```

# Memory allocation with automatic arrays

- ”Automatic arrays” can be seen in older codes
    - Not recommended

```fortran
subroutine calculate(m, n)
  integer :: m, n ! intended dimensions
  integer :: idx(0:m-1) ! an automatic array
  real :: mat(m,n) ! an automatic array
  ! no explicit allocate – but no checks upon failure either
  ...
  call do_something(m, n, idx, mat)
  ...
  ! no explicit deallocate - memory gets reclaimed automatically
end subroutine calculate
```

# Array intrinsic functions

- Built-in functions can apply various operations on a whole array, not
  just array elements
    - As a result either another array or a scalar value is returned
- A selection of just some part of the array through *masking* is possible

# Array intrinsic functions examples

- **`size(array[,dim])`** returns the number of elements in the array
- **`shape(array)`** returns an integer vector containing the size of
  the array with respect to each of its dimension
- **`count(l_array[,dim])`** returns the count of elements which are
  **`.true.`** in the logical **`l_array`**
- **`sum(array[,dim][,mask])`** returns the sum of the elements
    - optional `mask` argument is logical array with the same shape as `array`

# Array intrinsic functions

- **`any(l_array [, dim])`** returns a scalar value of **`.true.`** if any
  value in logical **`l_array`** is **`.true.`**
- **`all(l_array [, dim])`** returns a scalar value of **`.true.`** if all
  values in logical **`l_array`** are **`.true.`**
- **`minval/maxval (array [,dim] [, mask])`** return the
  minimum/maximum value in a given array
- **`minloc/maxloc (array [, mask])`** return a vector of location(s)
  where the minimum/maximum value(s) are
  found

# Array intrinsic functions: example

```fortran
integer :: j integer, parameter :: m = 10, n = 20
real :: x(m,n), v(n)
call random_number(x)

print *, size(x), size(v) ! prints m * n, n
print *, shape(x)         ! prints m, n
print *, size(shape(x))   ! prints 2
print *, count(x >= 0, dim=2)  ! the result is a vector
print *, sum(x, mask=x < 0.5)  ! performs sum of elements smaller than 0.5

v(1:n) = [ (j, j=1,n) ]
print *, any(v > -1 .and. v < 1)
print *, all(x >= 0, dim=1)
print *, minval(v), maxval(v)
print *, minloc(v), maxloc(v)
```

# Array intrinsic functions

- **`reshape(array,shape)`** returns a reconstructed array with
  different shape than in the input array

```fortran
integer :: m, n
real :: A(m, n), V(m*n)
...
! convert A (m-by-n matrix) into V (vector of length m*n) without loops
v = reshape(a, shape(v))
```

# Array intrinsic functions

Some array functions that manipulate vectors/matrices effectively:

- **`dot_product(v,w)`** returns a dot product of two vectors
- **`matmul(a,b)`** returns matrix multiply of two matrices
- **`transpose(a)`** returns transposed of the input matrix

# Array intrinsic functions

- Array control statements **`forall`** and **`where`** are commonly
  used in the context of manipulating arrays
- They can provide a masked assignment of values using effective
  vector operations

<div class="column">
```fortran
integer :: j, ix(5)
ix(:) = (/ (j, j=1,size(ix)) /)
where (ix == 0) ix = -9999
where (ix< 0)
  ix = -ix
elsewhere
  ix = 0
end where
```
</div>
<div class="column">
```fortran
integer :: j
real :: a(100,100), b(100), c(100)
! fill in diagonal matrix
forall (j=1:100) a(j,j) = b(j)
! fill in lower bi-diagonal matrix
forall (j=2:100) a(j,j-1) = c(j)
```
</div>

# Pointers to arrays

- The **`pointer`** attribute enables to create array (or scalar) *aliasing
  variables*
- Pointer variables are usually employed to *refer* to another array or
  array section
    - The another array needs to be declared with **`target`** attribute
- A pointer variable can also be a sole variable itself, used together
  with **`allocate`** for dynamic memory allocation
    - This is not a recommended practice

**C programmers:** a ”pointer” is a completely different construct in
Fortran

# Pointers to arrays

- Pointers can be used in assignment like normal arrays  

```fortran
integer, pointer :: p(:)  ! Number of dimensions needs to be declared
integer, target :: x(1000)
integer, target, allocatable :: y(:)
...
p => x           ! The pointer array can point to a whole array 
p => x(2:300:5)  ! or a part of it, no need to reallocate
print * p(1)  ! p(1) points to x(2)
p(2) = 0   ! This would change the value of x(3) too

p => y   ! Now p points to another array y (which needs to be allocated)

nullify(p) ! Now p points to nothing
```

# Pointers to arrays

<div class="column">
```fortran
real, pointer :: p_mat(:,:) => null()
real, target :: mat(100,200)

p_mat => mat
if ( associated (p_mat) ) &
print *, 'points to something'

nullify(p_mat)
if (.not. associated (p_mat) ) &
print *, 'points to nothing'
```
</div>
<div class="column">
**associated** function can be used to check whether a pointer is
associated with a target

- Returns **`.true.`** if associated with a target, **`.false.`** if not
- For an uninitialized pointer variables, the return value is
  undefined
</div>

# Summary

- *Arrays* make Fortran language a very versatile vehicle for
  computationally intensive program development
- Using its *array syntax*, vectors and matrices can be initialized
  and used in a very intuitive way
- *Dynamic memory allocation* enables sizing of arrays according to
  particular needs
- *Array intrinsic functions* further simplify coding effort and
  improve code readability
