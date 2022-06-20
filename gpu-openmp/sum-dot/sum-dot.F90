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

  !TODO start: create a data region and offload the two computations
  !so that data is kept in the device between the computations

  do i = 1, nx
     vecC(i) = vecA(i) + vecB(i)
  end do

  res = 0.0

  do i = 1, nx
     res = res + vecC(i) * vecB(i)
  end do

  !TODO end

  ! Compute the check value
  write(*,*) 'Reduction sum: ', sum(vecC)
  write(*,*) 'Dot product: ', res
  
end program vectorsum
