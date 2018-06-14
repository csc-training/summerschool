program datatype1
  use mpi
  implicit none

  integer, dimension(8,8) :: array
  integer :: rank, ierr
  integer :: indexedtype
  integer, dimension(4) :: counts, displs
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

  if (rank == 0) then
     write(*,*) 'Data in rank 0'
     do i=1,8
        write(*,'(8I3)') array(i, :)
     end do
  end if

  do i=1,4
     counts(i) = i
     displs(i) = i - 1 + 2 * (i - 1) * 8
  end do

  ! create datatype
  call mpi_type_indexed(4, counts, displs, MPI_INTEGER, indexedtype, ierr)
  call mpi_type_commit(indexedtype, ierr)

  ! send first indexed of matrix
  if (rank == 0) then
     call mpi_send(array, 1, indexedtype, 1, 1, MPI_COMM_WORLD, ierr)
  else if (rank == 1) then
     call mpi_recv(array, 1, indexedtype, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, &
          ierr)
  end if

  ! Print out the result
  if (rank == 1) then
     write(*,*) 'Received data'
     do i=1,8
        write(*,'(8I3)') array(i, :)
     end do
  end if

  call mpi_type_free(indexedtype, ierr)
  call mpi_finalize(ierr)

end program datatype1
