program hello
  use mpi_f08
  implicit none
  integer :: ierror
  call MPI_INIT(ierror)
  ! TODO: say hello! in parallel
  write(*,*) 'Hello!'
  call MPI_FINALIZE(ierror)

end program hello
