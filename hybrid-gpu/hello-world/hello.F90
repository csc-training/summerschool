program hello
#ifdef _OPENACC
  use openacc
#endif
  implicit none
#ifdef _OPENACC
  integer(acc_device_kind) :: devtype
#endif

  write (*,*) 'Hello world!'
#ifdef _OPENACC
  devtype = acc_get_device_type()
  write (*,'(A,X,I0)') 'Number of available OpenACC devices:', &
       & acc_get_num_devices(devtype)
  write (*,'(A,X,I0)') 'Type of available OpenACC devices:', devtype
#else
  write (*,*) 'Code compiled without OpenACC'
#endif

end program hello
