program factorial_b
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
  
contains 

  subroutine factorial(n, numbers, factorials)
    implicit none
    integer, intent(in) :: n, numbers(n)
    integer, intent(out) :: factorials(n)
    integer :: i
    
    factorials(1) = 1
    do i = 2, n
       factorials(i) = factorials(i-1) * i
    end do
  end subroutine factorial
  
end program factorial_b
