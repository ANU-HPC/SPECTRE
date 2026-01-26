module io_mod
  use iso_fortran_env, only: error_unit, input_unit, output_unit

  implicit none

  private

  integer, parameter, public :: stderr = error_unit
  integer, parameter, public :: stdout = output_unit
  integer, parameter, public :: stdin = input_unit

  integer, parameter, public :: iunit = 10 !< input
  integer, parameter, public :: gunit = 13 !< wall geometry
  integer, parameter, public :: aunit = 11 !< vector potential
  integer, parameter, public :: dunit = 12 !< derivative matrix
  integer, parameter, public :: hunit = 14 !< eigenvalues of Hessian
  integer, parameter, public :: munit = 14 !< matrix elements of Hessian
  integer, parameter, public :: lunit = 20 !< local unit
  integer, parameter, public :: vunit = 15 !< for examination of adaptive quadrature
end module io_mod
