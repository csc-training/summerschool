program datatype_struct
  use mpi
  implicit none
  type particle
     real :: coords(3)
     integer :: charge
     character(len=2) :: label
  end type particle
  integer, parameter :: n = 1000
  integer :: i, ierror,  myid,  ntasks, tag
  type(particle) :: particles(n)
  
  integer, parameter :: cnt=3
  integer:: particle_mpi_type, temp_type
  integer:: types(cnt),blocklen(cnt)
  integer(KIND=MPI_ADDRESS_KIND) :: disp(cnt)
  integer(KIND=MPI_ADDRESS_KIND) :: lb1, lb2, extent
  real(8) :: t1,t2
  
  call MPI_INIT(ierror)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierror)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, ntasks, ierror)
  
  ! insert some data for the particle struct
  if(myid == 0) then
    do i = 1, n
      call random_number(particles(i)%coords)
      particles(i)%charge = 54
      particles(i)%label = 'Xe'
    end do  
  end if
  
  ! TODO: Determine the true extent of one particle struct
  call MPI_GET_ADDRESS(particles(1),lb1,ierror)
  call MPI_GET_ADDRESS(particles(2),lb2,ierror)
  extent = lb2 - lb1


  t1=MPI_WTIME()
  ! TODO: send and receive using the MPI_BYTE datatype
  if(myid == 0) then
     do i=1,1000
        call MPI_SEND(particles, n*extent, MPI_BYTE, 1, i, &
             MPI_COMM_WORLD,ierror)
     end do
  else if(myid == 1) then
     do i=1, 1000
        call MPI_RECV(particles, n*extent, MPI_BYTE, 0, i, &
             MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
     end do
  end if
  t2=MPI_WTIME()
  
  write(*,*) "Time: ", myid, (t2-t1) / 1000d0
  write(*,*) "Check:", myid, particles(n)%coords(1)

  call MPI_FINALIZE(ierror)
end program datatype_struct
