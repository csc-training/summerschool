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

If you are not familiar with these commands, please check [this Linux tutorial](https://docs.csc.fi/support/tutorials/env-guide/).

## C

Here is an example C code `test.c` containing elements that are expected to be familiar.
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


Many core HPC libraries make extensive use of pointers, so please make sure you are familiar with common pointer syntax. Here is a brief cheatsheet:
```c
double* a; // variable `a` is a pointer to double; uninitialized.
double *b; // same as `double* b;`. Whitespace is irrelevant

double value = 42.0; // not a pointer
a = &value; // take address of `value` and assign it to pointer `a`
*a; // dereference the pointer, gives the value 42.0
double same_value = *a; // 42.0

// Note how the * operator has multiple meanings:
double value_times_value = (*a) * value; // multiplication, 42.0 * 42.0

double array[3] = { 1.0, 2.0, 3.0 }; // array of 3 doubles
a = array; // `a` now points to the first element of `array`
a = &array[0]; // same as above. Can you understand why?
a[2] = 0.0; // Modifies the last element of `array`

// NULL is a special pointer that "points to nothing". It is defined in stddef.h.
// Dereferencing null pointers is not allowed and will usually crash.
double* ptr = NULL;
*ptr = 42.0; // illegal, likely segmentation fault
```

You should also be familiar with C preprocessor directives:
```c
// Preprocessor "#include" does a copy & paste of another file, usually a header.
// For standard headers like stdio.h use "#include <some_standard_header.h>" instead
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

The C++ language originally started as an object-oriented extension of C but has long since grown into its own programming ecosystem with different conventions and practices. C++ is required for the GPU sections of this summer school as the low-level GPU frameworks (CUDA, HIP) are extensions of C++, but with a C-style interface.

We make some use of the C++ Standard Template Library (STL) which provides a handy collection of common container types and other helpers. Apart from these our use of C++ features is kept to a minimum for simplicity:
- Very little object-oriented code.
- From STL we mainly use `std::vector` for dynamic arrays, sometimes `std::array` for static arrays. These act as drop-in replacements for raw C-style arrays but provide automatic memory management. The "prefix" `std::` is a namespace specifier; all STL objects and functions reside in the `std` namespace.
- `constexpr` is used for compile-time constants. This is a type-safe replacement for preprocessor constants created with `#define`.
- Passing variables by reference to functions. Eg. `void some_function(int &a, const std::vector<double>& b);` declares that the integer `a` parameter is always passed by reference, and `b` (dynamic array of doubles) is always passed by constant reference.

Below is an example C++ code `test.cpp` containing elements that are expected to be familiar.

If you want to refresh your C++ knowledge, please check, for example, [this C++ tutorial](https://www.w3schools.com/cpp/).

```cpp
#include <vector> // gives std::vector
#include <array> // gives std::array
#include <cstdio> // C-style IO routines (could also just include stdio.h)

// Function definition like in C. array is passed as a pointer, and n is the number of elements in the array
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

// Same as above, but using C++ style dynamic array which we pass by a const reference
double calculate_sum_cpp(const std::vector<double>& array)
{
    double sum = 0.0;

    // std::vector knows how many elements it holds at any time
    for (int i = 0; i < array.size(); ++i)
    {
        sum += array[i];
    }

    return sum;
}

// Main function
int main(int argc, char *argv[])
{
    // This is a compile-time constant
    constexpr int exactPi = 3;

    // Declare variables
    int n = 4;
    double sum;

    // C-style formatted print can still be used
    printf("Hello n=%d\n", n);

    // Create a fixed-length array holding doubles
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

    // Call a function; use b.size() to get the current number of elements, and b.data() to get a raw pointer to the data
    sum = calculate_sum(b.size(), b.data());
    printf("Sum of b is %f\n", sum);

    // Call another function, this time passing the std::vector object directly.
    sum = calculate_sum_cpp(b);
    printf("Sum of b is still %f\n", sum);

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
    Sum of b is still 6.600000

Some additional C++ will be introduced over the summer school as they appear (template functions, type casting, implementing classes). We do not assume familiarity with these.

In addition to higher-level C++ features like the STL, C-style pointers and manual memory management are still frequently used in HPC. Please ensure you are familiar with these concepts by reviewing the C section above.


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

