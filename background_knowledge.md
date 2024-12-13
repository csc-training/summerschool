# Background knowledge

To get most of the summer school, we expect basic working knowledge on Linux and C, C++, or Fortran.
You can use this document to check the expected knowledge and prepare before the summer school starts.

## Linux and command line shell

We will be using supercomputers running Linux through command line interface / shell (e.g. Bash).
Typical session looks like this:
```bash
cd /scratch/my_project      # cd - change directory
mkdir my_dir                # mkdir - make directory
nano my_file.txt            # edit file using nano or any other editor of your choice
ls                          # ls - list directory contents
cp my_file.txt file2.txt    # cp - copy files
mv file2.txt file3.txt      # mv - move files
rm file3.txt                # rm - remove files
```

If you are not familiar with these commands, please check [Linux tutorial](https://docs.csc.fi/support/tutorials/env-guide/).

## C

Here is an example C code `test.c` containing elements that are expected to be familiar.

```cpp
#include <stdio.h>
#include <stdlib.h>

// Function definition
double calculate_sum(int n, const double *array)
{
    // Declare a variable
    double sum = 0.0;

    // Calculate sum by looping over array
    for (int i = 0; i < n; ++i) {
        sum += array[i];
    }

    // Return sum
    return sum;
}

// Main function
int main(int argc, char *argv[])
{
    // Declare variables
    int n = 4;
    double sum;

    // Print
    printf("Hello n=%d\n", n);

    // Declare a fixed-length array
    double a[4] = {1.1, 2.2, 3.3, 4.4};

    // Call a function
    sum = calculate_sum(4, a);
    printf("Sum of a is %f\n", sum);

    // Control statement
    if (sum > 10) {
        printf("Sum is large\n");
    } else {
        printf("Sum is small\n");
    }

    // Allocate a dynamic array
    double *b = (double*)malloc(sizeof(double) * n);

    // Set the array values by looping over array
    for (int i = 0; i < n; ++i) {
        b[i] = 1.1 * i;
    }

    // Call a function
    sum = calculate_sum(n, b);
    printf("Sum of b is %f\n", sum);

    // Free memory allocation
    free(b);

    return 0;
}
```

Compile the code:

    gcc test.c -o test.x

Execute the binary:

    ./test.x

In addition to C, it is beneficial to have knowledge on C++ as we will use some C++ features such as standard library containers (array and vector) in some code examples and exercises.
Please review the C++ section below how this same C code could look like in C++.

## C++

Here is an example C++ code `test.cpp` containing elements that are expected to be familiar (except elements marked "bonus").
Note that the function is "C style" (same as above in C section) to demonstrace interfacing C++ standard library containers (array and vector) with C interface.

```cpp
#include <array>
#include <cstdio>
#include <numeric>
#include <vector>

// C style function definition
double calculate_sum(int n, const double *array)
{
    // Declare a variable
    double sum = 0.0;

    // Calculate sum by looping over array
    for (int i = 0; i < n; ++i) {
        sum += array[i];
    }

    // Return sum
    return sum;
}

// Bonus: C++ style templated function definition
template <typename T>
T calculate_sum_vector(const std::vector<T> &vector)
{
    return std::accumulate(begin(vector), end(vector), static_cast<T>(0));
}

// Main function
int main(int argc, char *argv[])
{
    // Declare variables
    int n = 4;
    double sum;

    // Print
    printf("Hello n=%d\n", n);

    // Declare a fixed-length array
    std::array<double, 4> a = {1.1, 2.2, 3.3, 4.4};

    // Call a function; use a.data() to get a raw pointer
    sum = calculate_sum(size(a), a.data());
    printf("Sum of a is %f\n", sum);

    // Control statement
    if (sum > 10) {
        printf("Sum is large\n");
    } else {
        printf("Sum is small\n");
    }

    // Allocate a dynamic array
    std::vector<double> b(n);

    // Set the array values by looping over array
    for (int i = 0; i < size(b); ++i) {
        b[i] = 1.1 * i;
    }

    // Call a function; use b.data() to get a raw pointer
    sum = calculate_sum(size(b), b.data());
    printf("Sum of b is %f\n", sum);

    // Call a templated function
    sum = calculate_sum_vector(b);
    printf("Sum of b is %f again\n", sum);

    return 0;
}
```

Compile the code:

    g++ test.cpp -o test.x

Execute the binary:

    ./test.x

In addition to C++, it is beneficial to have knowledge on C as we will use libraries with C interface (via raw pointers like above) and the GPU programming frameworks are often based on explicit memory management (like `malloc()` and `free()` in C).
Please review the C section above how this same C++ code could look like in C.

## Fortran

Here is an example Fortran code `test.f90` containing elements that are expected to be familiar.

```f90
! Module definition
module demo
  use iso_fortran_env, only : real64
  implicit none

contains

  ! Module procedure definition
  function calculate_sum(array) result(ret_sum)
    real(real64), intent(in), dimension(:) :: array
    real(real64) :: ret_sum

    ret_sum = sum(array)
  end function calculate_sum

end module demo

program demoprogram
  use demo
  implicit none

  ! Declare variables
  integer, parameter :: n = 4
  integer :: i
  real(real64), dimension(4) :: a
  real(real64), dimension(:), allocatable :: b
  real(real64) :: sum_value

  ! Print
  write(*,'(a,i0)') 'Hello n=', n

  ! Initialize array
  a = [1.1_real64, 2.2_real64, 3.3_real64, 4.4_real64]

  ! Call a function
  sum_value = calculate_sum(a)

  ! Control statement
  if (sum_value > 10.0_real64) then
     write(*,*) 'Sum is large'
  else
     write(*,*) 'Sum is small'
  end if

  ! Allocate array
  allocate(b(n))

  ! Set values
  b = [(1.1_real64 * real(i, real64), i=0, n-1)]

  ! Call function
  sum_value = calculate_sum(b)
  write(*,'(a,f5.3)') 'Sum of b is ', sum_value

  ! Free memory
  deallocate(b)

end program demoprogram
```

Compile the code:

    gfortran test.f90 -o test.x

Execute the binary:

    ./test.x

In addition to Fortran, it is beneficial to have some knowledge on C/C++ languages as the GPU programming languages are mostly based on these languages.
Please review the C and C++ sections above how this same Fortran code could look like in C and/or C++.

