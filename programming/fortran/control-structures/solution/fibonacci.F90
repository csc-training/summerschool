program fibonacci
  implicit none
  integer :: f0, f1, f2

  f0 = 0
  f1 = 1
  write(*,*) f0
  write(*,*) f1
  f2 = f0 + f1
  do while (f2 < 100)
     write(*,*) f2
     f0 = f1
     f1 = f2
     f2 = f0 + f1
  end do

end program fibonacci
