---
title:  Introduction to C
author: CSC Summerschool 
date:   2019-07
lang:   en
---

# Data structures {.section}

# Arrays

  - Static arrays declared as **\<type\> name\[N\]** where **N** is the size of the array   
    * **int i\[10\]** array of 10 integers   
    * **float f\[3\]** array of 3 floats   
    * **char c\[60\]** string of 60 letters  
  - elements indexed starting from zero   
  	* **i\[3\]** 4th element of array i  
  - multi-dimensional arrays possible   
  	* **int m\[10\]\[20\]** a 10x20 matrix

# Arrays
<div class=column>
```c
 // integer vector
 int triplet[3];
 triplet[0] = 1;
 triplet[1] = 2 * triplet[0];
 triplet[2] =
 triplet[0] + triplet[1];

 // string
char word[] = "hello!";
printf(word); // hello!
```
</div>
<div class=column>
```c
// 2-dim matrix
int i,j;
float matrix[5][5];

for (i=0; i<5; i++){
	for (j=0; j<5; j++){
		matrix[i][j] = i*j;
		}
	 }

// print the 2nd row
for (i=0; i<5; i++){
	printf("%f ", matrix[1][i]);
	}
```
</div>

# Pointer arithmetic and arrays

- Pointers can be added to or subtracted from

  * This shifts the pointer to a new memory address before or after the
    current one
  * Each step is **sizeof(pointer\_datatype)** bytes, i.e. the memory size of the pointer's data type
(Watch out for out-of-bounds memory errors!)

```c
float vector[10], *ptr;

ptr = &vector[0]; // *ptr -> vector[0]
ptr = vector; // name of array works as pointer to the beginning
ptr++; // *ptr -> vector[1]
ptr += 3; // *ptr -> vector[4]
ptr--; // *ptr -> vector[3]
```
# Structures
<div class=column>
  - Multiple variables of arbitrary type can be grouped together in a
    **struct**
  - Member **y** of struct **x** accessed with **x.y** (or **x-\>y** if
    x is a pointer)
</div>
<div class=column>
<small>
```c
struct s {
	int i;
	float f;
} one;

typedef struct {
	int count;
	float price;
} apple;

apple a;
a.count = 7;
a.price = 1.24;
cost = a.count * a.price;

// with pointers
apple *ptr;
ptr = &a;
c = ptr->count;
ptr->price = 0.89;
```
</small>
</div>

