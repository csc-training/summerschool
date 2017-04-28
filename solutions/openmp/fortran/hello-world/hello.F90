program hello
  implicit none

  print *, 'Hello world!'
  !$omp parallel
  print *, 'X'
  !$omp end parallel

end program hello
