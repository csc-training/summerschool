program hello
  use mpi_f08
  implicit none
  integer :: rc, rank, ntasks, namelen
  character(len=MPI_MAX_PROCESSOR_NAME) :: procname

  call mpi_init(rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)
  call mpi_get_processor_name(procname, namelen, rc)

  if(rank == 0) then
     write(*,*) 'In total there are ', ntasks, 'tasks'
  endif

  write(*,*) 'Hello from rank ', rank, 'in node ', procname(:namelen)

  call mpi_finalize(rc)

end program hello
