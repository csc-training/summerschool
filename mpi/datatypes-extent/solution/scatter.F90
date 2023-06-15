program datatype1
  use mpi_f08
  implicit none

  integer :: rank, ntasks, rc
  integer :: sendarray(8,6), recvarray(8,6)
  type(mpi_datatype) :: vector, vector2
  integer(kind=mpi_address_kind) :: extent, lb

  integer :: i, j

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  ! initialize arrays
  recvarray = 0

  if (rank == 0) then
     sendarray = reshape([ ((i*10 + j, i=1,8), j=1,6) ], [8, 6] )
     write(*,*) 'Original data'
     do i=1, 8
        write(*,'(*(I3))') sendarray(i, :)
     end do
  end if

  ! create datatype
  call mpi_type_vector(6, 1, 8, MPI_INTEGER, vector, rc);
  extent = storage_size(i) / 8
  lb = 0
  call mpi_type_create_resized(vector, lb, extent, vector2, rc)
  call mpi_type_commit(vector2, rc)

  call mpi_scatter(sendarray, 2, vector2, recvarray, 2, vector2, 0, MPI_COMM_WORLD, rc)

  call mpi_type_free(vector2, rc)

  if (rank == ntasks - 1) then
     write(*,*) 'Received data'
     do i=1, 8
        write(*,'(*(I3))') recvarray(i, :)
     end do
  end if

  call mpi_finalize(rc)



end program datatype1
