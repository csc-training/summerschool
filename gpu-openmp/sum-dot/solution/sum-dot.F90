program vectorsum
  implicit none
  integer, parameter :: rk = selected_real_kind(12)
  integer, parameter :: ik = selected_int_kind(9)
  integer, parameter :: nx = 102400

  real(kind=rk), dimension(nx) :: vecA, vecB, vecC
  real(kind=rk)    :: sum, res
  integer(kind=ik) :: i

  ! Initialization of vectors
  do i = 1, nx
     vecA(i) = 1.0_rk/(real(nx - i + 1, kind=rk))
     vecB(i) = vecA(i)**2
  end do

  !$omp target data map(to:vecA, vecB) map(from:vecC)

  !$omp target teams distribute parallel do
  do i = 1, nx
     vecC(i) = vecA(i) + vecB(i)
  end do
  !$omp end target teams distribute parallel do

  res = 0.0

  !$omp target teams distribute parallel do reduction(+:res)
  do i = 1, nx
     res = res + vecC(i) * vecB(i)
  end do
  !$omp end target teams distribute parallel do

  !$omp end target data 

  ! Compute the check value
  write(*,*) 'Reduction sum: ', sum(vecC)
  write(*,*) 'Dot product: ', res
  
end program vectorsum
