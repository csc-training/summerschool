---
title:  "OpenMP offloading: <br>data movement"
event:  CSC Summer School in High-Performance Computing 2023
lang:   en
---

# OpenMP data environment

- GPU device has a separate memory space from the host CPU
    - unified memory is accessible from both sides
    - OpenMP supports both unified and separate memory
- OpenMP includes constructs and clauses to
  ***allocate variables on the device*** and to define
  ***data transfers to/from the device***
- Data needs to be mapped to the device to be accessible inside the offloaded
  target region
    - host is not allowed to touch the mapped data during the target region
    - variables are implicitly mapped to a target region unless explicitly
      defined in a data clause
        - scalars as *firstprivate*, static arrays copied to/from device


# Example: implicit mapping

```c
int N=1000;
double a=3.14;
double x[N], y[N];
// some code to initialise x and y

#pragma omp target
#pragma omp parallel for
for (int i=0; i < N; i++) {
    y[i] += a * x[i];
}
```

- Implicit copy of **a**, **x** and **y** to the target device when the target
  region is opened and back when it is closed


# Explicit mapping

`#pragma omp target map(type:list)`
  : `-`{.ghost}

- Explicit mapping can be defined with the `map` clause of the `target`
  construct
    - *list* is a comma-separated list of variables
    - *type* is one of:

<small>
<div class=column style="width:45%; margin-left:8%">
`to`
  : copy data to device on entry

`from`
  : copy data from device on exit
</div>
<div class=column style="width:45%">
`tofrom`
  : copy data to device on entry and back on exit

`alloc`
  : allocate on the device (uninitialised)
</div>
</small>


# Example: explicit mapping

```c
int N=1000;
double a=3.14;
double x[N], y[N];
// some code to initialise x and y

#pragma omp target map(to:x) map(tofrom:y)
#pragma omp parallel for
for (int i=0; i < N; i++) {
    y[i] += a * x[i];
}
```

- Both **x** and **y** are copied the device, but only **y** is copied back to
  the host
- Implicit copy of **a** to the device


# Dynamically allocated arrays

- With dynamically allocated arrays, one needs to specify the number of
  elements to be copied

```c++
int N=1000;
double *data = (double *) malloc(N * sizeof(double));

#pragma omp target map(tofrom:data[0:N])
// do something ..
```


# Motivation for optimizing data movement

- When dealing with an accelerator GPU device attached to a PCIe bus,
  **optimizing data movement** is often **essential** to achieve good
  performance
- The four key steps in porting to high performance accelerated code
    1. Identify parallelism
    2. Express parallelism
    3. Express data movement
    4. Optimise loop performance
    5. Go to 1!


# Data region

- Define data mapping for a structured block that may contain multiple target
  regions
    - C/C++: `#pragma omp target data map(type:list)`
    - Fortran: `!omp target data map(type:list)`
    - only maps data, one still needs to define a target region to execute
      code on the device
- Data transfers take place
    - from **the host** to **the device** upon entry to the region
    - from **the device** to **the host** upon exit from the region


# Example: data mapping over multiple target regions

```c++
int N=1000; double a=3.14, b=2.1;
double x[N], y[N], z[N];
// some code to initialise x, y, and z

#pragma omp target data map(to:x)
{
    #pragma omp target map(tofrom:y)
    #pragma omp parallel for
    for (int i=0; i < N; i++)
        y[i] += a * x[i];

    #pragma omp target map(tofrom:z)
    #pragma omp parallel for
    for (int i=0; i < N; i++)
        z[i] += b * x[i];
}
```


# Update

- Update a variable within a data region with the `update` directive
    - C/C++: `#pragma omp target update type(list)`
    - Fortran: `!omp target update type(list)`
    - a single line executable directive
- Direction of data transfer is determined by the *type*, which can be either
  `to` (= copy to device) or `from` (= copy from device)


# Why update?

- Useful for producing snapshots of the device variables on the host or
  for updating variables on the device
    - pass variables to host for visualization
    - communication with other devices on other computing nodes
- Often used in conjunction with
    - asynchronous execution of OpenMP constructs
    - unstructured data regions


# Example: update within a data region

```c++
int N=1000; double a=3.14, b=2.1;
double x[N], y[N], z[N];
// some code to initialise x, y, and z

#pragma omp target data map(to:x)
{
    #pragma omp target map(tofrom:y)
    #pragma omp parallel for
    for (int i=0; i < N; i++)
        y[i] += a * x[i];
    // ... some host code that modifies x ...
    #pragma omp target update to(x)
    #pragma omp target map(tofrom:z)
    #pragma omp parallel for
    for (int i=0; i < N; i++)
        z[i] += b * x[i];
}
```


# Reduction

`reduction(operation:list)`
  : `-`{.ghost}

- Applies the *operation* on the variables in *list* to reduce them to a
  single value
    - local private copies of the variables are created for each thread
    - initialisation depends on the *operation*
- Variables need to be shared in the enclosing parallel region
    - at the end, all local copies are reduced and combined with the original
      shared variable
- Directives that support reduction: `parallel`, `teams`, `scope`,
  `sections`, `for`, `simd`, `loop`, `taskloop`


# Reduction operators in C/C++ and Fortran

| Arithmetic Operator | Initial value |
| ------------------- | ------------- |
| `+`                 | `0`           |
| `-`                 | `0`           |
| `*`                 | `1`           |
| `max`               | least         |
| `min`               | largest       |


# Reduction operators in C/C++ only

<div class="column">
| Logical Operator | Initial value |
| ---------------- | ------------- |
| `&&`             | `1`           |
| `||`             | `0`           |
</div>

<div class="column">
| Bitwise Operator | Initial value |
| ---------------- | ------------- |
| `&`              | `~0`          |
| `|`              | `0`           |
| `^`              | `0`           |
</div>


# Reduction operators in Fortran only

<div class="column">
| Logical Operator | Initial value |
| ---------------- | ------------- |
| `.and.`          | `.true.`      |
| `.or.`           | `.false.`     |
| `.eqv.`          | `.true.`      |
| `.neqv.`         | `.false.`     |
</div>

<div class="column">
| Bitwise Operator | Initial value |
| ---------------- | ------------- |
| `iand`           | all bits on   |
| `ior`            | `0`           |
| `ieor`           | `0`           |
</div>


# Example: reduction

```c++
int N=1000;
double sum=0.0;
double x[N], y[N];
// some code to initialise x and y

#pragma omp target
#pragma omp parallel for reduction(+:sum)
for (int i=0; i < N; i++) {
    sum += y[i] * x[i];
}
```


# Summary

- GPU device has a separate memory space from the host CPU
  - unified memory accessible from both
- Implicit copy of data to/from `target` region
  - explicit data mapping with `map(type:list)`
- Structured data regions
  - `target data map(type:list)`
- Update
- Reduction
