module vector_algebra
  use iso_fortran_env, only : REAL64
  implicit none
  type vector_t
     real(REAL64) :: x, y, z
  end type vector_t

  ! TODO: overload operators needed by the parser

  interface operator(+)
     module procedure vector_sum
  end interface operator(+)

  interface operator(-)
     module procedure vector_subs
  end interface operator(-)

  interface operator(*)
     module procedure vector_dot_product
  end interface operator(*)

  interface operator(.x.)
     module procedure vector_cross_product
  end interface operator(.x.)

  interface abs
     module procedure vector_norm
  end interface abs

contains
  ! TODO: implement the corresponding functions

  function vector_sum(v1, v2) result(v3)
    type(vector_t), intent(in) :: v1,v2
    type(vector_t) :: v3
    v3%x=v1%x+v2%x
    v3%y=v1%y+v2%y
    v3%z=v1%z+v2%z
  end function vector_sum

  function vector_subs(v1, v2) result(v3)
    type(vector_t), intent(in)::v1,v2
    type(vector_t)::v3

    v3%x=v1%x-v2%x
    v3%y=v1%y-v2%y
    v3%z=v1%z-v2%z
  end function vector_subs

  function vector_dot_product(v1, v2) result(d)
    type(vector_t), intent(in) :: v1,v2
    real(REAL64) :: d

    d=v1%x*v2%x + v1%y*v2%y + v1%z*v2%z
  end function vector_dot_product

  function vector_cross_product(v1, v2) result(v3)
    type(vector_t), intent(in) :: v1,v2
    type(vector_t) :: v3

    v3%x=v1%y*v2%z-v1%z*v2%y
    v3%y=v1%z*v2%x-v1%x*v2%z
    v3%z=v1%x*v2%y-v1%y*v2%x
  end function vector_cross_product

  function vector_norm(v) result(l)
    type(vector_t) :: v
    real(REAL64) :: l

    l=sqrt(v%x**2 + v%y**2 + v%z**2)
  end function vector_norm

end module vector_algebra
