module vector_algebra
  use iso_fortran_env, only : REAL64
  implicit none
  type vector_t
     real(REAL64) :: x, y, z
  end type vector_t

  ! TODO: overload operators needed by the parser

  ! ...

contains
  ! TODO: implement the corresponding functions

  function vector_sum(v1, v2) result(v3)
    !    ...
  end function vector_sum

  ! ...

end module vector_algebra
