program vectors
  use vector_algebra
  implicit none
  type(vector_t) :: v1, v2

  parser: do
     write(*,'(/,A)', advance='no') ' v1 ='
     read(*,*) v1%x, v1%y, v1%z

     write(*,'(A)', advance='no') ' v2 ='
     read(*,*) v2%x, v2%y, v2%z

     write(*,'(2(A,G10.4))') ' |v1|=', abs(v1), ' |v2|=', abs(v2)
     write(*,'(A,3G10.4)') ' v1 + v2 =', v1 + v2
     write(*,'(A,3G10.4)') ' v1 - v2 =', v1 - v2
     write(*,'(A,3G10.4)') ' v1 . v2 =', v1 * v2
     write(*,'(A,3G10.4)') ' v1 x v2 =', v1 .x. v2
  end do parser

end program vectors
