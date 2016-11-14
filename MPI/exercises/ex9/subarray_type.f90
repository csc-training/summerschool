program datatype2

  use mpi

  implicit none

  integer, dimension(8,8) :: array
  integer :: rank, ierr
  integer :: blocktype
  integer, dimension(2) :: sizes, subsizes, offsets
  integer :: i, j

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank ,ierr)

  ! initialize arrays
  if (rank == 0) then
     do i=1,8
        do j=1,8
           array(i,j) = i*10 + j
        end do
     end do
  else
     array(:,:) = 0
  end if

  ! TODO start
  ! create a datatype for a subblock [2:5][3:5] of the 8x8 matrix
  sizes(1) =
  sizes(2) =
  subsizes(1) =
  subsizes(2) =
  offsets(1) =
  offsets(2) =


  ! send a block of a matrix
  if (rank == 0) then
     call mpi_send( ,  ,  , 1, 1, MPI_COMM_WORLD, ierr)
  else if (rank == 1) then
     call mpi_recv( ,  ,  , 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, &
          ierr)
  end if

  ! TODO end

  ! Print out the result
  if (rank == 1) then
     do i=1,8
        write(*,'(8I3)') array(i, :)
     end do
  end if

  call mpi_type_free(blocktype, ierr)
  call mpi_finalize(ierr)

end program datatype2
