---
title:  Introduction to C
author: CSC Summerschool 
date:   2019-07
lang:   en
---


# Functions {.section}

# Functions

- Functions are the only subroutine type in C
    -  But functions do not have to return anything

Function definitions are of form
```c
type func-name(param-list)
{

/* body of function */

}
```

- Here **type** is the return type of the function
    - **void** means that function does not return anything


# main function

- Every C program has a main function where the program execution starts

<div class=column>
int main(int `argc`{.input}, char `*argv[]`{.input})
  : `argc`{.input}
    : Number of arguments on command line
  : `*argv[]`{.input}
    : Array of strings with the arguments on command line. The first argument is always the name of the binary
  : `return value`{.input} 
    : Error code, zero means success, non-zero values are errors 
</div>

<div class=column>
```c
int main(int argc, char *argv[])
{
/* body of function */
}
```
</div>

# Function example

  - This function returns the sum of two integer arguments:

```c
#include <stdio.h>

int add(int a, int b) {
	return a + b;
}

//by using void, argc and argv can be omitted
int main(void) {
	int val;
	val = add(3, 6);
	printf(“Sum is %i\n”, val);
	return 0;
}
```

# **Variable scoping**

<div class=column>
- Variable scoping in C is local unless defined otherwise
    - Variables declared in the function definition are only visible
   inside the function body
    - Variables of calling scope are not visible inside function body
</div>
<div class=colum>
```c
void funcB(float a) {
    int counter;
    //counter not accessible from funcA
    ...
}

int funcA(void) {
    float value1, value2;
    //Not accessible from funcB
    funcB(value1);
    ...
}
```
</div>

# Arguments
<div class=column>
- All arguments are passed as *values*
    - Copy of the value is passed to the function
    - Functions can not change the value of a variable in the *calling scope*

- Pointers can be used to pass a reference to a variable as an argument!
</div>
<div class=column>
```c
void funcB(int a) {

	a += 10;

}

int funcA(void) {
	int var = 0;
	funcB(var);
	// var is still 0\!
	...
}
```
</div>

# Functions with pointers

  - Pointers allow “returning” of multiple values

```c
// passing by reference
void add_pi(float *a, int *b) {
	float pi=3.14;
	*a += pi;
	(*b)++;
}

float a=1.0;
int b=1;
add_pi(&a, &b); // a=4.14 b=2
```

