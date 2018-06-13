program check
  implicit none
  integer :: i

  write(*,*) "Give a number"
  read(*,*) i

  if (i < 0) then
     write(*,*) "i is negative", i
  else if (i == 0) then
     write(*,*) "i is zero", i
  else if (i > 100) then
     write(*,*) "i is large", i;
  else
     write(*,*) "i is something else", i
  end if

end program check
