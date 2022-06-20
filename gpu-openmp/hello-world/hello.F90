program hello
#ifdef _OPENMP
  use omp_lib
#endif
  implicit none

  integer :: num_devices
  logical :: initial_device

  num_devices = omp_get_num_devices()
  print *, "Number of available devices", num_devices

!$omp target 
    initial_device = omp_is_initial_device()
!$omp end target 
    if (initial_device) then
      write(*,*) "Running on host"
    else 
      write(*,*) "Running on device"
    end if

end program
