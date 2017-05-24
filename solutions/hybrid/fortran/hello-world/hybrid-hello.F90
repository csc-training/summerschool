program hybridhello
  use omp_lib
  implicit none
  include 'mpif.h'
  integer :: provided, my_id, ntask, tid, rc

  call MPI_Init_thread(MPI_THREAD_MULTIPLE, provided, rc)
  if (provided == MPI_THREAD_MULTIPLE) then
     write(*,*) 'MPI library supports MPI_THREAD_MULTIPLE'
  else if (provided == MPI_THREAD_SERIALIZED) then
     write(*,*) 'MPI library supports MPI_THREAD_SERIALIZED'
  else if (provided == MPI_THREAD_FUNNELED) then
     write(*,*) 'MPI library supports MPI_THREAD_FUNNELED'
  else
     write(*,*) 'No multithreading support'
  end if

  call MPI_Comm_rank(MPI_COMM_WORLD, my_id, rc)
  call MPI_Comm_size(MPI_COMM_WORLD, ntask, rc)

!$omp parallel default(shared) private(tid)
  tid = omp_get_thread_num()

  write(*,'(A,I3,A,I3)') 'Rank ', my_id, ' thread ', tid
!$omp end parallel

  call MPI_Finalize(rc)

end program hybridhello
