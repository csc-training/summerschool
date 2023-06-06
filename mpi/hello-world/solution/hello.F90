program hello
  use mpi
  implicit none
  integer :: rc, myid, ntasks, namelen
  character(len=MPI_MAX_PROCESSOR_NAME) :: procname

  call mpi_init(rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_get_processor_name(procname, namelen, rc)

  if(myid == 0) then
     write(*,*) 'In total there are ', ntasks, 'tasks'
  endif

  write(*,*) 'Hello from rank ', myid, 'in node ', procname(:namelen)

  call mpi_finalize(rc)

end program hello
