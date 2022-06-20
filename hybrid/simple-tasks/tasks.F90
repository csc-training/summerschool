program tasks
  use omp_lib
  implicit none

  integer :: array(4)
  integer :: tid, i

  array = 0
  write(*,'(A30, *(I3))') "Array at the beginning", array

  ! TODO: launch threads and create tasks so that there 
  ! one task per loop iteration
  do i = 1, 4
    tid = omp_get_thread_num()
    array(i) = array(i) + tid
    write(*,*) "Task", i, "executed by thread", tid
  end do

  ! TODO end

  write(*,'(A30, *(I3))') "Array at the end", array

end program 
