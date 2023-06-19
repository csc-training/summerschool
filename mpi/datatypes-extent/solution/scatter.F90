program datatype1
  use mpi_f08
  implicit none

  integer :: i, j
  integer :: array(8,6), recvarray(8,6)
  integer :: rank, ntasks, irank, ierr
  type(mpi_datatype) :: vector, tmp
  integer(kind=mpi_address_kind) :: extent, lb

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)

  ! Initialize arrays
  recvarray = 0
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
  tmp = vector
  extent = storage_size(i) / 8
  lb = 0
  call mpi_type_create_resized(tmp, lb, extent, vector, ierr)
  call mpi_type_commit(vector, ierr)

  ! Scatter rows
  call mpi_scatter(array, 1, vector, recvarray, 1, vector, 0, MPI_COMM_WORLD, ierr)

  ! Free datatype
  call mpi_type_free(vector, ierr)

  ! Print received data
  do irank = 0, ntasks-1
    if (rank == irank) then
       write(*,*) 'Received data on rank', rank
       do i=1,8
          write(*,'(*(I3))') recvarray(i, :)
       end do
    end if
    call mpi_barrier(mpi_comm_world, ierr)
  end do

  call mpi_finalize(ierr)

end program datatype1
