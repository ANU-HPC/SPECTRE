module precisions_mod
  use iso_fortran_env, only: real64

  implicit none

  integer, parameter :: rkind = real64

  !> Machine precision: 0.5*epsilon(1.0_rkind) for 64 bit double precision
  real(rkind), parameter, public :: machprec = 1.11e-16_rkind
  !> Very small number
  real(rkind), parameter, public :: vsmall = 100 * machprec
  !> Small number
  real(rkind), parameter, public :: small = 10000 * machprec
  !> Square root of machine precision
  real(rkind), parameter, public :: sqrtmachprec = sqrt(machprec)
  !> This is used to avoid taking log10(0.0_rkind): see e.g. dforce
  real(rkind), parameter, public :: logtolerance = 1.0e-32_rkind
end module precisions_mod
