program datatype1
  use mpi_f08
  implicit none

  integer :: i, j
  integer, dimension(8,8) :: array
  integer :: rank, ierr

  ! Declare a variable storing the MPI datatype
  type(mpi_datatype) :: subarray
  integer, dimension(2) :: sizes, subsizes, offsets

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank ,ierr)

  ! Initialize arrays
  if (rank == 0) then
     do i=1,8
        do j=1,8
           array(i,j) = i*10 + j
        end do
     end do
  else
     array(:,:) = 0
  end if

  ! Print data on rank 0
  if (rank == 0) then
     write(*,*) 'Data in rank 0'
     do i=1,8
        write(*,'(8I3)') array(i, :)
     end do
  end if

  ! Create datatype
  sizes = 8
  subsizes = 4
  offsets = 2
  call mpi_type_create_subarray(2, sizes, subsizes, offsets, MPI_ORDER_FORTRAN, MPI_INTEGER, subarray, ierr)
  call mpi_type_commit(subarray, ierr)

  ! Send data from rank 0 to rank 1
  if (rank == 0) then
     call mpi_send(array(1, 1), 1, subarray, 1, 1, MPI_COMM_WORLD, ierr)
  else if (rank == 1) then
     call mpi_recv(array(1, 1), 1, subarray, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, &
          ierr)
  end if

  ! Free datatype
  call mpi_type_free(subarray, ierr)

  ! Print out the result on rank 1
  if (rank == 1) then
     write(*,*) 'Received data'
     do i=1,8
        write(*,'(8I3)') array(i, :)
     end do
  end if

  call mpi_finalize(ierr)

end program datatype1
