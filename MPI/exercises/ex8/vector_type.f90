program datatype1

  use mpi

  implicit none

  integer, dimension(8,8) :: array
  integer :: rank, ierr
  integer :: rowtype
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
  ! create datatype


  ! send first row of matrix
  if (rank == 0) then
     call mpi_send( ,  , , 1, 1, MPI_COMM_WORLD, ierr)
  else if (rank == 1) then
     call mpi_recv( ,  , , 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, &
          ierr)
  end if

  ! TODO end

  ! Print out the result
  if (rank == 1) then
     do i=1,8
        write(*,'(8I3)') array(i, :)
     end do
  end if

  ! TODO free datatype
  call mpi_finalize(ierr)

end program datatype1
