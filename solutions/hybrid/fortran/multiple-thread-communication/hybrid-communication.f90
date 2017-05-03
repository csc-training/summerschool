program hybridhello
  use omp_lib
  implicit none
  include 'mpif.h'
  integer :: provided, my_id, ntask, tid, nthreads, msg, i, rc

  call MPI_Init_thread(MPI_THREAD_MULTIPLE, provided, rc)
  if (provided < MPI_THREAD_MULTIPLE) then
     write(*,*) 'MPI version does not support MPI_THREAD_MULTIPLE'
     call MPI_Abort(MPI_COMM_WORLD,rc)
  end if

  call MPI_Comm_rank(MPI_COMM_WORLD, my_id, rc)
  call MPI_Comm_size(MPI_COMM_WORLD, ntask, rc)

  if (ntask < 2) then
     write(*,*) 'Call this program using at least 2 tasks'
     call MPI_Abort(MPI_COMM_WORLD, rc)
  end if

!$omp parallel default(shared) private(msg, tid, nthreads, i)
  nthreads = omp_get_num_threads()
  tid = omp_get_thread_num()

  if (my_id == 0) then
     !$omp single
     write(*,'(I3,A)') nthreads, ' threads in the master rank'
     !$omp end single
     do i = 1, ntask-1
        call MPI_Send(tid, 1, MPI_INTEGER, i, tid, MPI_COMM_WORLD, rc)
     end do
  else
     call MPI_Recv(msg, 1, MPI_INTEGER, 0, tid, MPI_COMM_WORLD, &
          & MPI_STATUS_IGNORE, rc)
     write(*,'(A,I3,A,I3,A,I3)') 'Rank ', my_id, ' thread ', tid, &
          & ' received ', msg
  end if
!$omp end parallel

  call MPI_Finalize(rc)

end program hybridhello
