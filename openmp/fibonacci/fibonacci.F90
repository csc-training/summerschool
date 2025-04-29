! Copyright 2021 CSC - IT Center for science
program fibonacci
  use omp_lib
  use iso_fortran_env, only : REAL64
  implicit none

  integer :: n, res
  real(REAL64) :: t0, t1

  n = 42

  t0 = omp_get_wtime()
  res = fib(n)
  t1 = omp_get_wtime()

  write(*,*) "fib ", n, "=", res
  write(*,*) "calculation took ", t1 - t0, "s"

  contains
    
    recursive function fib(n) result(f)
      implicit none
      integer :: n, f, x, y

      if (n < 2) then
         f = n
      else
         x = fib(n - 1)
         y = fib(n - 2)
         f = x + y
      end if

    end function fib

end program fibonacci
  
