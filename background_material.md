# Background knowledge material

To get most of the summer school, it is expected to have basic working knowledge on Linux and C, C++, or Fortran.
You can use this document to check the expected knowledge and prepare before the summer school starts.

Feel free to contact the course organizers if you have any questions.

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

Here is an example C code `test.c` containing elements that are expected to be familiar. In particular, familiarity with pointers is required.

If you want to refresh your C knowledge, please check, for example, [this C tutorial](https://www.w3schools.com/c/).

```c
#include <stdio.h>
#include <stdlib.h>

// This is a comment

/*
This is a multi-line comment
*/

// Function definition
double calculate_sum(int n, const double *array)
{
    // Declare and initialize a local variable
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

    // Explicit type cast: convert 'n' to double type
    double n_double = (double) n;

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

Compile the code on Linux command line:

    gcc test.c -o test.x

Execute the binary on Linux command line:

    ./test.x

This gives the following output:

    Hello n=4
    Sum of a is 11.000000
    Sum is large
    Sum of b is 6.600000

You should also be familiar with C preprocessor directives:
```c
// Preprocessor "#include" does a copy & paste another file. For standard headers like stdio.h use "#include <some_header.h>" instead
#include "some_header.h"

// Preprocessor constant definition. Use sparingly, prefer const variables instead
#define MY_PREPROCESSOR_CONSTANT 42

/* Preprocessor macro. Somewhat similar to a function, but the compiler literally replaces every occurance of MY_MACRO with the macro body.
So MY_MACRO(2) would become 2 + 1 */
#define MY_MACRO(x) x + 1
```

In addition to C, it is beneficial to have some knowledge on C++ as we will use C++ standard library containers (array and vector) in some code examples and exercises.
Please review the C++ section below how this same C code could look like in C++.

## C++

The C++ language originally started as an object-oriented extension of C but has long since grown into its own programming ecosystem with different conventions and practices.
We make some use of the C++ Standard Template Library (STL) which provides a handy collection of common container types and other helpers. Apart from these our use of C++ features is kept to a minimum for simplicity:
- Very little object-oriented code.
- From STL we mainly use `std::vector` for dynamic arrays, sometimes `std::array` for static arrays. These act as drop-in replacements for raw C-style arrays but provide automatic memory management. The "prefix" `std::` is a namespace specifier; all STL objects and functions reside in the `std` namespace.
- `constexpr` is used for compile-time constants. Type-safe replacement for preprocessor constants created with `#define`.
- Bonus: reference parameters in functions may occasionally be used. Eg. `void some_function(int &a, const std::vector<double>& b);` declares that the `a` parameter is always passed by reference, and `b` is always passed by constant reference.
- Bonus: templated functions may occasionally appear. You can identify these by the syntax involving angle brackets (<...>).
- Bonus: `static_cast` is sometimes used to replace C-style casts (stricter and safer type semantics).

Here is an example C++ code `test.cpp` containing elements that are expected to be familiar (except elements marked "bonus").

If you want to refresh your C++ knowledge, please check, for example, [this C++ tutorial](https://www.w3schools.com/cpp/).

```cpp
#include <vector> // gives std::vector
#include <array> // gives std::array
#include <cstdio> // C-style IO routines (could also just include stdio.h)
#include <numeric> // Bonus: gives std::accumulate

// Function definition like in C
double calculate_sum(int n, const double *array)
{
    // Declare and initialize a local variable
    double sum = 0.0;

    // Calculate sum by looping over array
    for (int i = 0; i < n; ++i) {
        sum += array[i];
    }

    // Return sum
    return sum;
}

/* Bonus: templated C++ function definition. Input std::vector is passed by constant reference,
ie. the function is not allowed to modify the passed object.
*/
template <typename T>
T calculate_sum_vector(const std::vector<T> &vector)
{
    return std::accumulate(begin(vector), end(vector), static_cast<T>(0));
}

// Main function
int main(int argc, char *argv[])
{

    // This is a compile-time constant
    constexpr int exactPi = 3;

    // Declare variables
    int n = 4;
    double sum;

    // Print
    printf("Hello n=%d\n", n);

    // Create a fixed-length array
    std::array<double, 4> a = {1.1, 2.2, 3.3, 4.4};

    // Call a function. Use a.size() to get number of array elements, and a.data() to get a raw pointer to the contained data
    sum = calculate_sum(a.size(), a.data());
    printf("Sum of a is %f\n", sum);

    // Control statement
    if (sum > 10) {
        printf("Sum is large\n");
    } else {
        printf("Sum is small\n");
    }

    // Allocate a dynamic array, initial length 'n' elements
    std::vector<double> b(n);

    // Set the array values by looping over array
    for (int i = 0; i < size(b); ++i) {
        b[i] = 1.1 * i;
    }

    // Call a function; use b.size() to get the current number of elements, and b.data() to get a raw pointer
    sum = calculate_sum(b.size(), b.data());
    printf("Sum of b is %f\n", sum);

    /* Bonus: Call a templated function. This works without explicitly specifying the template parameter
    because the compiler can deduce it from the type of our input argument. */
    sum = calculate_sum_vector(b);
    printf("Sum of b is %f again\n", sum);

    // Resize the array to 10 elements
    b.resize(10);

    // a and b automatically deallocate themselves when going out of scope

    return 0;
}
```

Compile the code on Linux command line:

    g++ test.cpp -o test.x

Execute the binary on Linux command line:

    ./test.x

This gives the following output:

    Hello n=4
    Sum of a is 11.000000
    Sum is large
    Sum of b is 6.600000
    Sum of b is 6.600000 again

In addition to C++, it is beneficial to have knowledge on C as we will use libraries with C interface (via raw pointers like above) and the GPU programming frameworks are often based on explicit memory management (like `malloc()` and `free()` in C).
Please review the C section above how this same C++ code could look like in C.

## Fortran

Here is an example Fortran code `test.f90` containing elements that are expected to be familiar.

If you want to refresh your Fortran knowledge, please check, for example, [this Fortran tutorial](https://fortran-lang.org/en/learn/quickstart/).

! This is a comment

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
  write(*,'(a,f8.3)') 'Sum of a is ', sum_value

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
  write(*,'(a,f8.3)') 'Sum of b is ', sum_value

  ! Free memory
  deallocate(b)

end program demoprogram
```

Compile the code on Linux command line:

    gfortran test.f90 -o test.x

Execute the binary on Linux command line:

    ./test.x

This gives the following output:

    Hello n=4
    Sum of a is   11.000
     Sum is large
    Sum of b is    6.600

In addition to Fortran, it is beneficial to have some knowledge on C/C++ languages as the GPU programming languages are mostly based on these languages.
Please review the C and C++ sections above how this same Fortran code could look like in C and/or C++.

