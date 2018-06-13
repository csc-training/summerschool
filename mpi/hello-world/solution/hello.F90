program ex1
  use mpi
  implicit none
  integer :: rc, myid, ntasks


  call MPI_INIT(rc)

  call MPI_COMM_SIZE(MPI_COMM_WORLD, ntasks, rc)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, rc)

  if(myid == 0) then
     write(*,*) 'In total there are ',ntasks, 'tasks'
  endif

  write(*,*) 'Hello from ',myid

  call MPI_FINALIZE(rc)

end program ex1
