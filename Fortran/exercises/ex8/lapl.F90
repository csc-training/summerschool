program lap
  use laplacian_mod
  implicit none
  type(field) :: previous, current

  previous%nx = 16
  previous%ny = 16
  current%nx = 16
  current%ny = 16
  call initialize(previous)
  call print_field(previous)
  call initialize(current)
  call laplacian(previous, current)
  call print_field(current)
  
end program lap
