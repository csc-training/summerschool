program datatype1
  use mpi_f08
  implicit none

  integer :: i, j
  integer :: array(8,8)
  integer :: rank, ierr

  ! Declare a variable storing the MPI datatype
  type(mpi_datatype) :: indexedtype
  integer, dimension(4) :: counts, displs

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

  ! Initialize arrays
  array = 0
  if (rank == 0) then
     array = reshape([ ((i*10 + j, i=1,8), j=1,8) ], [8, 8] )
  end if

  ! Print data on rank 0
  if (rank == 0) then
     write(*,*) 'Data on rank', rank
     do i=1,8
        write(*,'(*(I3))') array(i, :)
     end do
  end if

  ! Create datatype
  do i = 1, 4
     counts(i) = i
     displs(i) = i - 1 + 2 * (i - 1) * 8
  end do
  call mpi_type_indexed(4, counts, displs, MPI_INTEGER, indexedtype, ierr)
  call mpi_type_commit(indexedtype, ierr)

  ! Send data from rank 0 to rank 1
  if (rank == 0) then
     call mpi_send(array, 1, indexedtype, 1, 1, MPI_COMM_WORLD, ierr)
  else if (rank == 1) then
     call mpi_recv(array, 1, indexedtype, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, &
          ierr)
  end if

  ! Free datatype
  call mpi_type_free(indexedtype, ierr)

  ! Print received data
  if (rank == 1) then
     write(*,*) 'Received data on rank', rank
     do i=1,8
        write(*,'(*(I3))') array(i, :)
     end do
  end if

  call mpi_finalize(ierr)

end program datatype1
