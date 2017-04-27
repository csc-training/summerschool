program exer1
  implicit none
  integer :: var1, var2
  var1 = 1
  var2 = 2

  ! TODO:
  !   Test different data sharing clauses here
  print *, 'Region 1:       var1=', var1, 'var2=', var2
  var1 = var1 + 1
  var2 = var2 + 1
  !end here :)
  print *, 'After region 1: var1=', var1, 'var2=', var2
  print *

end program exer1
