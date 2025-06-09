---
title:  "Case study: Heat equation"
event:  CSC Summer School in High-Performance Computing 2025
lang:   en
---

# Case study: heat equation {.section}

# Heat equation

<div class=column>

- Partial differential equation that describes the variation of temperature in a given region over time
  $$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$
- Time-dependent temperature field: $u(x, y, z, t)$
- Thermal diffusivity constant: $\alpha$

</div>

<div class=column>

 ![](img/pot.png){.center width=60%}

</div>


# Numerical solution

- Discretize: Finite difference Laplacian in two dimensions
  $$
  \begin{align*}
  \nabla^2 u \rightarrow& \frac{u(i-1,j)-2u(i,j)+u(i+1,j)}{(\Delta x)^2} \\
                       +& \frac{u(i,j-1)-2u(i,j)+u(i,j+1)}{(\Delta y)^2}
  \end{align*}
  $$
  ![](img/t_field.png){.center width=40%}


# Time evolution

- Explicit time evolution with time step $\Delta t$

  $$u^{m+1}(i,j) = u^m(i,j) + \Delta t \alpha \nabla^2 u^m(i,j)$$

- Note: algorithm is stable only when

  $$\Delta t \leq \frac{1}{2 \alpha} \left(\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}\right)^{-1} $$

- Given the initial condition ($u(t=0) = u^0$) one can follow the time evolution of the temperature field

# Solving heat equation in parallel

- Temperature at each grid point can be updated independently
- Data can be distributed with domain decomposition

  ![](img/domain.png){.center width=60%}

# Solving heat equation in parallel

- Local data dependency: communication is needed for boundary layers
- Information about neighbouring domains is stored in ”ghost layers”

  ![](img/halo-exchange.png){.center width=50%}

- Before each update cycle, CPU cores communicate boundary data: <br>halo exchange

# Serial code structure

<pre style="color:black; padding:1ex">

main():
  initialize_field()
  write_field()


  for time in time_steps:
    evolve_field()
    if write_this_time_step:
      write_field()
    swap_fields()

  write_field()
  finalize_field()

</pre>

# Parallel code structure

<pre style="color:black; padding:1ex">
main():
  <span style="color:var(--csc-blue)">initialize_parallelization()</span>
  initialize_field()
  write_field()

  for time in time_steps:
    <span style="color:var(--csc-blue)">halo_exchange_field()</span>
    evolve_field()
    if write_this_time_step:
      write_field()
    swap_fields()

  write_field()
  finalize_field()
  <span style="color:var(--csc-blue)">finalize_parallelization()</span>
</pre>

