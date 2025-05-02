<!-- Adapted from material by EPCC https://github.com/EPCCed/archer2-MPI-2020-05-14 -->

# Parallel calculation of π

An approximation to the value of π can be calculated from the following
expression

$$\frac{\pi}{4} = \int_0^1 \frac{dx}{1+x^2} \approx \frac{1}{N} \sum_{i=1}^N \frac{1}{1+\left( \frac{i-\frac{1}{2}}{N}\right)^2}$$

<!--
\frac{\pi}{4} = \int_0^1 \frac{dx}{1+x^2} \approx \frac{1}{N} \sum_{i=1}^N \frac{1}{1+\left( \frac{i-\frac{1}{2}}{N}\right)^2}
-->

where the answer becomes more accurate with increasing N. As each term is independent,
the summation over i can be parallelized nearly trivially.

Starting from the serial code [pi.cpp](pi.cpp) (or [pi.F90](pi.F90) for Fortran), make a version
that performs the calculation parallel with **two** processes.

1. Divide the range over N in two, so that rank 0 does i=1, 2, ... , N/2 and rank 1 does
   i=N/2 + 1, N/2 + 2, ... , N.

2. Both tasks calculate their own partial sums

3. Once finished with the calculation, rank 1 sends its partial sum to rank 0, which then
   calculates the final result and prints it out.

4. Compare the result to the that of the serial calculation, do you get **exactly** the same
   result? If not, can you explain why?


