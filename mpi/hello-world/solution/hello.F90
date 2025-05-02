program hello
  use mpi_f08
  implicit none
  integer :: rc, rank, ntasks, namelen
  character(len=MPI_MAX_PROCESSOR_NAME) :: procname

  ! Global MPI initialization, must be paired with `mpi_finalize` at end of the program
  call mpi_init(rc)

  ! Query size of MPI "world", ie. all copies of the program that were started by mpirun/srun
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  ! Find the identifier (rank) of this process within the MPI world
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)

  if(rank == 0) then
     write(*,*) 'In total there are ', ntasks, 'tasks'
  endif

  ! Bonus: find name of the processor (node) that this rank is running on.
  ! As stated in the docs for MPI_Get_processor_name,
  ! we must allocate a char array of at least length MPI_MAX_PROCESSOR_NAME.

  call mpi_get_processor_name(procname, namelen, rc)

  write(*,*) 'Hello from rank ', rank, 'in node ', procname(:namelen)

  call mpi_finalize(rc)

end program hello
