module constants_mod
  use precisions_mod, only: rkind

  implicit none(type, external)

  private

  real(rkind), parameter, public :: pi2 = 6.28318530717958623_rkind !< \f$2\pi\f$
  real(rkind), parameter, public :: pi = pi2 / 2.0_rkind !< \f$\pi\f$
  real(rkind), parameter, public :: mu0 = 2.0e-07_rkind * pi2 !< \f$4\pi\cdot10^{-7}\f$
  real(rkind), parameter, public :: goldenmean = 1.618033988749895_rkind !< golden mean = \f$( 1 + \sqrt 5 ) / 2\f$

  real(rkind), parameter, public :: version = 3.23_rkind !< version of SPECTRE

end module constants_mod
