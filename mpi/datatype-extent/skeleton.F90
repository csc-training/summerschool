program datatype1
  use mpi_f08
  implicit none

  integer :: rank, rc
  integer :: sendarray(8,6), recvarray(8,6)
  type(mpi_datatype) :: vector, vector2
  integer(kind=mpi_address_kind) :: extent, lb

  integer :: i, j

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc)

  ! initialize arrays
  recvarray = 0

  if (rank == 0) then
     sendarray = reshape([ ((i*10 + j, i=1,8), j=1,6) ], [8, 6] )
     write(*,*) 'Original data'
     do i=1, 8
        write(*,'(*(I3))') sendarray(i, :)
     end do
  end if

  ! TODO create datatype

  ! Communicate with the datatype
  if (rank == 0) then

  else if (rank == 1) then

  end if

  ! free datatype

  ! TODO end

  if (rank == 1) then
     write(*,*) 'Received data'
     do i=1, 8
        write(*,'(*(I3))') recvarray(i, :)
     end do
  end if

  call mpi_finalize(rc)
  
  

end program datatype1
