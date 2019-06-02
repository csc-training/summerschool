---
title:  Introduction to C
author: CSC Summerschool 
date:   2019-07
lang:   en
---

# Dynamic arrays{.section}

# Dynamic memory management

  - In most cases the exact size of all data structures is not known at 
  compile time because they depend on the input data
  - Dynamic memory management is accomplished by using pointers and 
  manually managing memory (de)allocation
  - Relevant functions are **malloc()**, **realloc()**, **free()**

# malloc

- malloc function is defined in header file stdlib.h: 
- The memory area returned by the malloc is uninitialized!	

void malloc(size\_t `size`{.input});
  : `size`{.input}
    : Size of allocated memory in bytes
  : `return value`{.output} 
    : Pointer of type **void** to a memory location with
allocated space. If allocation fails, **malloc** returns **NULL\!**



# Dynamic arrays

- **malloc** can be used to allocate space for arrays

    - **malloc** returns a pointer to the beginning of the array
    - Elements can be accessed with normal array syntax

- The size of memory allocation can be determined using **sizeof**
operator which returns the size of the argument in bytes


```c
// Example for a dynamic array of five integers 
int *ptr = (int *) malloc(5 * sizeof(int));
```

# free

- **free** deallocates previous allocated object 

void free(void* `ptr`{.input});
  : `ptr`{.input}
    : Pointer to memory area to be deallocated


- After freeing you should not try to access any part of the allocation

- Calling free with a pointer that does not point to a allocated memory
can crash the code

    - Calling free twice is a common mistake\!

# Dynamic arrays

```c
 int n_elems = 32;
 float *prices;

 // allocate memory for the required amount of floats
 prices = malloc(n_elems*sizeof(float));
 for (i=0; i<n_elems; i++) {
	prices[i] = i*1.23;
 }

 // add space for one more float
 prices = realloc(prices, sizeof(float)*(n_elems+1));
 prices[n_elems] = 0.91;

 // de-allocate the memory block
 free(prices);
```

# Pointers to pointers

- It is possible to have pointer references to pointers

    - This is very useful when functions have to manipulate values of arguments of pointer type
    - Multidimensional arrays are also naturally mapped into pointers of pointers

# Pointer to a pointer example
```c
#include<stdio.h>

int main(void) {
	int a = 5;
	int *int_ptr;
	int **int_ptr_ptr;
	int_ptr_ptr = &int_ptr;
	int_ptr = &a;
	printf(“a=%i\n”, **int_ptr_ptr);
	return 0;
}
// Result: a=5
```

# Dynamic multi-dimensional arrays

- Doable, but becomes complicated

- No real multi-dimensional arrays in C, so really just arrays of arrays

    - Two dimensional array maps to a variable that is a pointer to a pointer

- Memory management by hand

    - There are one correct way to do the allocation
    - Easy to make mistakes, _beware here lieth dragons!_
    - More than 2 allocations is never needed

- Often best to just use 1D array and map N dimensional indices

# Dynamic multi-dimensional arrays

- Dynamic 2D array in *contiguous* memory:
    - First, allocate memory for pointers to the first element of each row
    - Second, allocate memory for all elements
    - Third, point each row at the first element of that row

# Dynamic multi-dimensional arrays

```c
 /* allocate memory */
 matrix = malloc(rows * sizeof(float *));
 matrix[0] = malloc(rows * cols * sizeof(float));
 /* point the beginning of each row at the correct address */
 for (i = 0; i < rows; i++){
 	matrix[i] = matrix[0] + i * cols;
 }
 // start using the 2D array
 matrix[0][2] = 3.14;
```
![](images/allocation2-01.svg){.center width=100%}


# **Memory layout**

- Allocating space for the whole array using a single malloc call is the
recommended way

    - Number of expensive malloc calls is minimized
    - Array is represented as one contiguous block in memory
    - It can be copied without looping over rows
    - Most IO and numerical libraries assume contiguous storage

# Dynamic multi-dimensional arrays

  - After using a dynamic multi-dimensional array, remember to free each array inside the main array

```c
 /* When using contiguous memory */

 free(matrix[0]);

 free(matrix);
```

# Using indexing to represent 2D array

- Allocate memory for all elements as normal 1D array
- Compute indices so that the 1D memory area map to the 2D array we want to represent
- Benefits
    - The compiler understands better the code, and hence performance may be better
    - Allocation and deallocation easier
    - Straightforward to generalize to higher dimensions
- Drawbacks
    - Code may look less elegant

# Using indexing to represent 2D array
```c
 /* matrix now normal pointer */
 float *matrix;
 /* allocate memory */
 matrix = malloc(rows * cols * sizeof(float));
 /* start using the array, element [i][j] in 2D array now maps to [i * cols + j]*/
 matrix[0 * cols + 2] = 3.14;
 /* free memory */
 free(matrix);
```
![](images/allocation3-01.svg){.center width=100%}

