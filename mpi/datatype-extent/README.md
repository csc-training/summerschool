## Modifying extent

When the datatype contains gaps in the beginning or in the end, one might need to
modify the *extent* of the datatype.

Starting from [skeleton.c](skeleton.c) or [skeleton.F90](skeleton.F90), create a vector
datatype for sending a column (C) or row (Fortran) of a matrix.

1. Verify that the datatype works by communicating a single column/row.

2. Try to send to columns / rows. What happens? Can you explain why?

3. Create a new datatype with resized extent, so that communicating multiple columns / rows
   succeeds.

4. Try to scatter columns / rows with `MPI_Scatter`
