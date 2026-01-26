! SPDX-FileCopyrightText: 2009 CSC - IT Center for Science Ltd. <www.csc.fi>
!
! SPDX-License-Identifier: MIT

program hello
  implicit none

  print *, 'Hello world!'
  !$omp parallel
  print *, 'X'
  !$omp end parallel

end program hello
