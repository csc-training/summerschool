program hello
  implicit none

  print *, 'Hello world!'
  !$OMP PARALLEL
  print *, 'X'
  !$OMP END PARALLEL

end program hello
