## Work Sharing: vector sum

Calculate the sum of two vectors (`C = A + B`) in parallel using OpenACC.

A skeleton code is provided in `sum(.c|.F90)`. The main computation loop
should be parallelised using OpenACC. Try both `acc parallel`and `acc kernels`
and check the compiler diagnostics output. Run the programs and compare the
results.
