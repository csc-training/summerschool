## Data movement clauses and reductions: dot product

Calculate the dot product of two vectors (`r = sum[ A(i) x B(i) ]`) in
parallel using OpenMP.

A working serial code is provided in `dot-product(.c|.F90)`.

1. Try to accelerate the code first using worksharing constructs without any
   clauses. Are you able to get the correct result with the GPU accelerated
   code? Compiler diagnostics with `-Minfo=all` may provide hints about
   possible issues.

2. Add proper clauses to the OpenMP directives so that code works correctly.
   Add also clauses for avoiding unnecessary memory copies (*i.e.* vectors do
   not need to copied back from device). Use compiler diagnostics and runtime
   debug for finding out if the clauses had desired effect.
