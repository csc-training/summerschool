program hello
  use omp_lib
  implicit none
  integer :: tid, nthreads

  print *, 'Hello world!'
  !$omp parallel private(tid) shared(nthreads)
  tid = omp_get_thread_num()
  !$omp single
  nthreads = omp_get_num_threads()
  !$omp end single
  !$omp critical
  print *, '  ... from thread ID', tid
  !$omp end critical
  !$omp end parallel
  print *, 'There were', nthreads, 'threads in total.'

end program hello
