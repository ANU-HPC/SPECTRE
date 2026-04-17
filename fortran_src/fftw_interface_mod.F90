module fftw_interface_mod
  ! allow(C121)
  use iso_c_binding

  implicit none(type, external)

  include 'fftw3.f03'

  private

  ! Propagating some FFTW variables
  public :: c_double_complex, FFTW_BACKWARD, FFTW_FORWARD, FFTW_MEASURE, FFTW_DESTROY_INPUT, fftw_plan_dft_2d, fftw_execute_dft, fftw_destroy_plan

  integer, public :: nt_plan, nz_plan
  type(c_ptr), public :: planf !< FFTW-related (?)
  type(c_ptr), public :: planb !< FFTW-related (?)
  complex(c_double_complex), allocatable, public :: cplxin(:, :)!< FFTW-related (?)
  complex(c_double_complex), allocatable, public :: cplxout(:, :)!< FFTW-related (?)
end module fftw_interface_mod
