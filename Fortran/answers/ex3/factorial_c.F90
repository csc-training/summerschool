program factorial_c
  use factmod
  implicit none
  integer, parameter :: n = 10
  integer :: numbers(n), facts(n)
  integer :: i

  numbers(1:n) = [ (i, i=1,n) ]

  do i = 1, n
     call factorial(n, numbers, facts)
  end do
  write (*,*) 'Factorials:'
  do i = 1, n
     write (*,'(2I10)') numbers(i), facts(i)
  end do
  
end program factorial_c
