program tasks
  use omp_lib
  implicit none

  integer :: array(4)
  integer :: tid, i

  array = 0
  write(*,'(A30, *(I3))') "Array at the beginning", array

!$omp parallel
!$omp single
  tid = omp_get_thread_num()
  write(*,*) "Tasks created by", tid
  do i = 1, 4
    !$omp task
    tid = omp_get_thread_num()
    array(i) = array(i) + tid
    write(*,*) "Task", i, "executed by thread", tid
    !$omp end task
  end do
!$omp end single
!$omp end parallel

  write(*,'(A30, *(I3))') "Array at the end", array

end program 
