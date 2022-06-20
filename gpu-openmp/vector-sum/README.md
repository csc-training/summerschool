## Work Sharing: vector sum

Calculate the sum of two vectors (`C = A + B`) in parallel using OpenMP.

A skeleton code is provided in `sum(.c|.F90)`. Fill in the computational part 
and calculate it in parallel in GPU using OpenMP offloading. Try both `teams`, `parallel`,
`distribute`, `for` / `do` constructs as well as `loop` construct.

Use `-Minfo=all` compiler diagnostics to investigate differences between the two versions

Try to run the code also in CPU only nodes, do you get the same results both with GPU
and CPU?
