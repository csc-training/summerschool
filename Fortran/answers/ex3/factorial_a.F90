program factorial_a
  implicit none
  integer, parameter :: n = 10
  integer :: numbers(n), facts(n)
  integer :: i
  
  numbers(1:n) = [ (i, i=1,n) ]
  
  write (*,*) 'Factorials:'
  do i = 1, n
     write (*,'(2I10)') numbers(i), factorial(numbers(i))
  end do
  
contains 

  function factorial(n) result(fact)
    implicit none
    integer, intent(in) :: n
    integer :: i
    integer :: fact
    
    fact = 1
    do i = 1, n
       fact = fact * i
    end do
  end function factorial
  
end program factorial_a
