program parallel_pi
  use mpi_f08
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: dp = REAL64

  integer, parameter :: n = 840
  integer :: myid, ntasks, rc
  
  type(mpi_status) :: status

  integer :: i, istart, istop, chunksize, remainder
  real(dp) :: pi, localpi, x

  call mpi_init(rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)  
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc);

  if (myid == 0) then
     write(*,*) "Computing approximation to pi with n=", n
     write(*,*) "Using", ntasks, "mpi processes"
  end if

  chunksize = n / ntasks
  istart = myid * chunksize + 1
  istop = (myid + 1) * chunksize

  ! Handle possible uneven division
  remainder = mod(n, ntasks)

  if (remainder > 0) then
    if (myid < remainder) then
      ! Assign this task one element more
      istart = istart + myid
      istop = istop + myid + 1
    else
      istart = istart + remainder
      istop = istop + remainder
    end if
  end if

  localpi = 0.0
  do i = istart, istop
     x = (i - 0.5) / n
     localpi = localpi + 1.0 / (1.0 + x**2)
  end do
  
   ! Reduction to rank 0
   if (myid == 0) then 
     pi = localpi
     do i=1, ntasks - 1
        call mpi_recv(localpi, 1, MPI_DOUBLE_PRECISION, i, 0, MPI_COMM_WORLD, &
          status, rc)
        pi = pi + localpi
     end do
     pi = pi * 4.0 / n
     write(*,'(A,F18.16,A,F10.8,A)') 'Approximate pi=', pi, ' (exact pi=', 4.0*atan(1.0_dp), ')'
  else 
     call mpi_send(localpi, 1, MPI_DOUBLE_PRECISION, 0, 0, &
           MPI_COMM_WORLD, rc)
  end if

  call mpi_finalize(rc)

end program

