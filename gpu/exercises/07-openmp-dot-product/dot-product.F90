program dot_product
  implicit none
  integer, parameter :: rk = selected_real_kind(12)
  integer, parameter :: ik = selected_int_kind(9)
  integer, parameter :: nx = 102400

  real(kind=rk), dimension(nx) :: vecA, vecB, vecC
  real(kind=rk)    :: res
  integer(kind=ik) :: i

  ! Initialization of vectors
  do i = 1, nx
     vecA(i) = 1.0_rk/(real(nx - i + 1, kind=rk))
     vecB(i) = vecA(i)**2
  end do

  ! TODO start: offload and parallelize the computation

  res = 0.0
  do i = 1, nx
     res = res + vecA(i) * vecB(i)
  end do

  ! TODO end

  ! Compute the check value
  write(*,*) 'Dot product: ', res

end program dot_product
