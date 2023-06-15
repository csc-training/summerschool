program datatype1
  use mpi_f08
  implicit none

  integer :: i, j
  integer :: rank, rc
  integer :: array(8,6)
  type(mpi_datatype) :: vector, tmp
  integer(kind=mpi_address_kind) :: extent, lb

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)

  ! Initialize arrays
  array = 0
  if (rank == 0) then
     array = reshape([ ((i*10 + j, i=1,8), j=1,6) ], [8, 6] )
  end if

  ! Print data on rank 0
  if (rank == 0) then
     write(*,*) 'Data on rank', rank
     do i=1, 8
        write(*,'(*(I3))') array(i, :)
     end do
  end if

  ! Create datatype
  call mpi_type_vector(6, 1, 8, MPI_INTEGER, vector, rc);
  tmp = vector
  extent = storage_size(i) / 8
  lb = 0
  call mpi_type_create_resized(tmp, lb, extent, vector, rc)
  call mpi_type_commit(vector, rc)

  ! Send data from rank 0 to rank 1
  if (rank == 0) then
      call mpi_send(array, 2, vector, 1, 0, MPI_COMM_WORLD, rc)
  else if (rank == 1) then
      call mpi_recv(array, 2, vector, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
  end if

  ! Free datatype
  call mpi_type_free(vector, rc)

  ! Print received data
  if (rank == 1) then
     write(*,*) 'Received data on rank', rank
     do i=1, 8
        write(*,'(*(I3))') array(i, :)
     end do
  end if

  call mpi_finalize(rc)



end program datatype1
