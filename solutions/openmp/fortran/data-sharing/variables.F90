program exer1
  implicit none
  integer :: var1, var2
  var1 = 1
  var2 = 2

  !$OMP PARALLEL PRIVATE(VAR1, VAR2)
  print *, 'Region 1:       var1=', var1, 'var2=', var2
  var1 = var1 + 1
  var2 = var2 + 1
  !$OMP END PARALLEL
  print *, 'After region 1: var1=', var1, 'var2=', var2
  print *

  !$OMP PARALLEL FIRSTPRIVATE(VAR1, VAR2)
  print *, 'Region 2:       var1=', var1, 'var2=', var2
  var1 = var1 + 1
  var2 = var2 + 1
  !$OMP END PARALLEL
  print *, 'After region 2: var1=', var1, 'var2=', var2
  print *

  !$OMP PARALLEL
  print *, 'Region 3:       var1=', var1, 'var2=', var2
  var1 = var1 + 1
  var2 = var2 + 1
  !$OMP END PARALLEL
  print *, 'After region 3: var1=', var1, 'var2=', var2
  print *

end program exer1
