## Laplacian

Write a double do loop for evaluating the Laplacian using the
finite-difference approximation

<!-- Equation
\begin{align*}
\nabla^2 u  &= \frac{u(i-1,j)-2u(i,j)+u(i+1,j)}{(\Delta x)^2} \\
 &+ \frac{u(i,j-1)-2u(i,j)+u(i,j+1)}{(\Delta y)^2}
 \end{align*}
 --> 
![img](https://quicklatex.com/cache3/2d/ql_59f49ed64dbbe76704e0679b8ad7c22d_l3.png)

Start from the skeleton [laplacian.F90](laplacian.F90). Test your
implementation with the function implemented in item a) in [control-structures](../control-structures). The
analytic solution for that function is
<!-- Equation
\nabla^2 u(x,y)  = 4 
 --> 
![img](https://quicklatex.com/cache3/f2/ql_1133b1a8877ffd0acf814919818995f2_l3.png)

for checking the correctness of your implementation.

