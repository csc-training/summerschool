program sections

  use omp_lib
  implicit none

!$OMP PARALLEL
!$OMP SECTIONS

!$OMP SECTION
  write(*,*) 'First section, thread', omp_get_thread_num()

!$OMP SECTION
  write(*,*) 'Second section, thread', omp_get_thread_num()

!$OMP END SECTIONS
!$OMP END PARALLEL

end program sections

