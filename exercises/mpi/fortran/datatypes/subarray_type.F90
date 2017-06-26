program datatype2

  use mpi

  implicit none

  integer, dimension(8,8) :: array
  integer :: rank, ierr
  !TODO: declare variable for block datatype
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

  !TODO: create a datatype for a subblock [2:5][3:5] of the 8x8 matrix
  !sizes(1) =
  !sizes(2) =
  !subsizes(1) =
  !subsizes(2) =
  !offsets(1) = 
  !offsets(2) = 

  !TODO: send a block of a matrix from rank 0 to rank 1

  ! Print out the result
  if (rank == 1) then
     do i=1,8
        write(*,'(8I3)') array(i, :)
     end do
  end if
      
  !TODO: free mpi datatype	

  call mpi_finalize(ierr)

end program datatype2
