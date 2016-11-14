module factmod
  implicit none

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
  
end module factmod
