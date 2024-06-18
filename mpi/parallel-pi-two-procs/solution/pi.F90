program parallel_pi
  use mpi_f08
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: dp = REAL64

  integer, parameter :: n = 840
  integer :: rank, ntasks, rc

  type(mpi_status) :: status

  integer :: i, istart, istop
  real(dp) :: pi, localpi, x

  call mpi_init(rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)
  if (ntasks /= 2) then
     print *, "this example works only with two processes!"
     call mpi_finalize(rc)
     stop
  end if

  call mpi_comm_rank(MPI_COMM_WORLD, rank, rc);

  if (rank == 0) then
     write(*,*) "Computing approximation to pi with n=", n
     write(*,*) "Using", ntasks, "mpi processes"
  end if

  if (rank == 0) then
     istart = 1
     istop = n / 2

     localpi = 0.0
     do i = istart, istop
        x = (i - 0.5) / n
        localpi = localpi + 1.0 / (1.0 + x**2)
     end do

     pi = localpi
     call mpi_recv(localpi, 1, MPI_DOUBLE_PRECISION, 1, 0, MPI_COMM_WORLD, &
          status, rc)
     pi = pi + localpi
     pi = pi * 4.0 / n
     write(*,'(A,F18.16,A,F10.8,A)') 'Approximate pi=', pi, ' (exact pi=', 4.0*atan(1.0_dp), ')'

  else if (rank == 1) then
     istart = n / 2 + 1
     istop = n

     localpi = 0.0
     do i = istart, istop
        x = (i - 0.5) / n
        localpi = localpi + 1.0 / (1.0 + x**2)
     end do

     call mpi_send(localpi, 1, MPI_DOUBLE_PRECISION, 0, 0, &
           MPI_COMM_WORLD, rc)
     end if

     call mpi_finalize(rc)

end program

