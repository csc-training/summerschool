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
  integer(KIND=MPI_ADDRESS_KIND) :: lb, extent
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
  
  ! TODO: define the datatype for type particle
  types=(/ MPI_REAL, MPI_INTEGER, MPI_CHARACTER  /)
  blocklen=(/ 3,1,2 /)
  call MPI_GET_ADDRESS(particles(1)%coords, disp(1), ierror)
  call MPI_GET_ADDRESS(particles(1)%charge, disp(2), ierror)
  call MPI_GET_ADDRESS(particles(1)%label, disp(3), ierror)
  do i = cnt, 1, -1
    disp(i)=disp(i)-disp(1)
  end do
  call MPI_TYPE_CREATE_STRUCT(cnt, blocklen, &
      disp,types, particle_mpi_type,ierror)
  call MPI_TYPE_COMMIT(particle_mpi_type, ierror)

  ! TODO: Check extent. 
  ! (Not really neccessary on most systems.)
  call MPI_TYPE_GET_EXTENT(particle_mpi_type,lb,extent,ierror)
  call MPI_GET_ADDRESS(particles(1),disp(1),ierror)
  call MPI_GET_ADDRESS(particles(2),disp(2),ierror)
  ! TODO: resize the particle_mpi_type
  if(extent /= disp(2)-disp(1)) then
    temp_type=particle_mpi_type
    lb=0
    extent=disp(2)-disp(1)
    call MPI_TYPE_CREATE_RESIZED(temp_type,lb,extent,particle_mpi_type,ierror)
    call MPI_TYPE_COMMIT(particle_mpi_type,ierror)
    call MPI_TYPE_free(temp_type,ierror)
  end if


  t1=MPI_WTIME()
  if(myid == 0) then
     do i=1,1000
        call MPI_SEND(particles, n, particle_mpi_type, 1, i, &
             MPI_COMM_WORLD,ierror)
     end do
  else if(myid == 1) then
     do i=1, 1000
        call MPI_RECV(particles, n, particle_mpi_type, 0, i, &
             MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
     end do
  end if
  t2=MPI_WTIME()
  
  write(*,*) "Time: ", myid, (t2-t1) / 1000d0
  write(*,*) "Check:", myid, particles(n)%coords(1)

  call MPI_TYPE_free(particle_mpi_type, ierror)
  
  call MPI_FINALIZE(ierror)
    

end program datatype_struct
