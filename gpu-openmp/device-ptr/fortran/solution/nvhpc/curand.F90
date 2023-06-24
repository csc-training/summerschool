module curand

  use, intrinsic :: iso_c_binding

  integer(c_int), parameter :: CURAND_RNG_PSEUDO_DEFAULT = 100

  interface

    function curandCreateGenerator(gen, gentype) bind(C, name='curandCreateGenerator') result(stat)
       use, intrinsic :: iso_c_binding
       integer(c_size_t) :: gen
       integer(c_int), value :: gentype
       integer(c_size_t) :: stat

    end function curandCreateGenerator

    function curandGenerateUniform(gen, x, n) bind(C, name='curandGenerateUniform') result(stat)
      use, intrinsic :: iso_c_binding
      integer(c_size_t), value :: gen
      real :: x(*)
      integer(c_size_t), value :: n
      integer(c_size_t) :: stat

    end function curandGenerateUniform

  end interface

end module curand
