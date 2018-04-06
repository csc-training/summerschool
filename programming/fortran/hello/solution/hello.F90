program hello
  integer :: a = 1, b = 2
  real :: c
  c = a+b
  write(*,*) 'Hello world!'
  write(*,*) 'a=', a, 'b=', b, 'a+b=', c, 'sqrt(a+b)', sqrt(c)
end program hello
