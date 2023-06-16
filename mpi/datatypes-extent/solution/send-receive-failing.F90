program datatype1
  use mpi_f08
  implicit none

  integer :: i, j
  integer :: array(8,6)
  integer :: rank, ierr
  type(mpi_datatype) :: vector

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

  ! Initialize arrays
  array = 0
  if (rank == 0) then
     array = reshape([ ((i*10 + j, i=1,8), j=1,6) ], [8, 6] )
  end if

  ! Print data on rank 0
  if (rank == 0) then
     write(*,*) 'Data on rank', rank
     do i=1,8
        write(*,'(*(I3))') array(i, :)
     end do
  end if

  ! Create datatype
  call mpi_type_vector(6, 1, 8, MPI_INTEGER, vector, ierr);
  call mpi_type_commit(vector, ierr)

  ! Send data from rank 0 to rank 1
  if (rank == 0) then
      call mpi_send(array, 2, vector, 1, 0, MPI_COMM_WORLD, ierr)
  else if (rank == 1) then
      call mpi_recv(array, 2, vector, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
  end if

  ! Free datatype
  call mpi_type_free(vector, ierr)

  ! Print received data
  if (rank == 1) then
     write(*,*) 'Received data on rank', rank
     do i=1,8
        write(*,'(*(I3))') array(i, :)
     end do
  end if

  call mpi_finalize(ierr)

end program datatype1
