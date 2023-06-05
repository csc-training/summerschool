---
title:  Parallel algorithms
event:  CSC Summer School in High-Performance Computing 2023
lang:   en
---


# Parallel algoritms {.section}

# Parallel algorithms

- General principles
    - Overlapping communication
    - Overlapping CPU and GPU
    - Load balance?
    - DAG?
- Cannon?
- Sudoku? (tasks)
- Domain decomposition with halo exchange
- Some tree algorithm (basic idea)

# What can be calculated in parallel?

- There needs to be independent computations

<div class=column>
Gauss-Seidel iteration:
<small>
```
while True
  for i:
    u[i] = (u[i-1]) + u[i+1]) - h**2 f[i]) / 2

until converged(u)
```
</small>

- cannot be parallelized due to data dependency

</div>
<div class=column>
Jacobi iteration:
<small>
```
while True
  for i:
    u_new[i] = (u_old[i-1]) + u_old[i+1]) - h**2 f[i]) / 2
  swap(u_new, u_old)
until converged(u)
```
</small>

- can be parallelized

</div>

# Data distribution

# Local 

# Communication and synchronization

# Case study: heat equation {.section}

# Heat equation

<div class=column>

- Partial differential equation that describes the variation of temperature in a given region over time

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$

- Temperature variation: $u(x, y, z, t)$
- Thermal diffusivity constant: $\alpha$

</div>

<div class=column>

 ![](img/pot.png){.center width=60%}

</div>

# Numerical solution

- Discretize: Finite difference Laplacian in two dimensions

 <small>
 $$\nabla^2 u \rightarrow \frac{u(i-1,j)-2u(i,j)+u(i+1,j)}{(\Delta x)^2}
  + \frac{u(i,j-1)-2u(i,j)+u(i,j+1)}{(\Delta y)^2} $$
</small>
Temperature field $u(i,j)$

 ![](img/t_field.png){.center width=45%}


# Time evolution

- Explicit time evolution with time step Δt

$$u^{m+1}(i,j) = u^m(i,j) + \Delta t \alpha \nabla^2 u^m(i,j)$$

- Note: algorithm is stable only when

$$\Delta t < \frac{1}{2 \alpha} \frac{(\Delta x \Delta y)^2}{(\Delta x)^2
(\Delta y)^2}$$

- Given the initial condition ($u(t=0) = u^0$) one can follow the time evolution of the temperature field

# Solving heat equation in parallel

- Temperature at each grid point can be updated independently
- Domain decomposition

 ![](img/domain.png){.center width=60%}

- Straightforward in shared memory computer

# Solving heat equation in parallel

- In distributed memory computers, each core can access only its own memory

- Information about neighbouring domains is stored in ”ghost layers”

 ![](img/ghost.png){.center width=50%}

- Before each update cycle, CPU cores communicate boundary data: <br>halo exchange

# Serial code structure

 ![](img/serial_code.png){.center width=70%}

# Parallel code structure

 ![](img/parallel_code.png){.center width=70%}
---
title:  "Parallelizing heat equation"
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---

# Case study: heat equation {.section}

# Heat equation

<div class=column>

- Partial differential equation that describes the variation of temperature in a given region over time

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$

- Temperature variation: $u(x, y, z, t)$
- Thermal diffusivity constant: $\alpha$

</div>

<div class=column>

 ![](img/pot.png){.center width=60%}

</div>


# Numerical solution


- Discretize: Finite difference Laplacian in two dimensions

 <small>
 $$\nabla^2 u \rightarrow \frac{u(i-1,j)-2u(i,j)+u(i+1,j)}{(\Delta x)^2}
  + \frac{u(i,j-1)-2u(i,j)+u(i,j+1)}{(\Delta y)^2} $$
</small>
Temperature field $u(i,j)$

 ![](img/t_field.png){.center width=45%}



# Time evolution


- Explicit time evolution with time step Δt

$$u^{m+1}(i,j) = u^m(i,j) + \Delta t \alpha \nabla^2 u^m(i,j)$$

- Note: algorithm is stable only when

$$\Delta t < \frac{1}{2 \alpha} \frac{(\Delta x \Delta y)^2}{(\Delta x)^2
(\Delta y)^2}$$

- Given the initial condition ($u(t=0) = u^0$) one can follow the time evolution of the temperature field

# Solving heat equation in parallel

- Temperature at each grid point can be updated independently
- Domain decomposition

 ![](img/domain.png){.center width=60%}

- Straightforward in shared memory computer

# Solving heat equation in parallel

- In distributed memory computers, each core can access only its own memory

- Information about neighbouring domains is stored in ”ghost layers”

 ![](img/ghost.png){.center width=50%}

- Before each update cycle, CPU cores communicate boundary data: <br>halo exchange

# Serial code structure

 ![](img/serial_code.png){.center width=70%}

# Parallel code structure

 ![](img/parallel_code.png){.center width=70%}
