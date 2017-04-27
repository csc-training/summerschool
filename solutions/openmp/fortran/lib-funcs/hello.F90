program thread_hello
  use omp_lib
  implicit none

  integer :: tid, nthreads

  !$omp parallel private(tid) shared(nthreads)
  !$omp single
  nthreads = omp_get_num_threads()
  print *, 'There are ', nthreads, ' threads in total.'
  !$omp end single

  tid = omp_get_thread_num()

  !$omp critical
  print *, 'Hello from thread id ', tid, '/', nthreads
  !$omp end critical
  !$omp end parallel
end program thread_hello
