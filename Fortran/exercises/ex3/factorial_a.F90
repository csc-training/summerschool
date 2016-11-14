program factorial_a
  implicit none
  integer, parameter :: n = 10
  integer :: numbers(n), facts(n)
  integer :: i
  
  numbers(1:n) = [ (i, i=1,n) ]
  
  write (*,*) 'Factorials:'
  do i=1,n
     write (*,'(2I10)') numbers(i), factorial(numbers(i))
  end do
  
contains 

! TODO: insert the function "factorial" for calculating n!=1*2*...*n
  
end program factorial_a
