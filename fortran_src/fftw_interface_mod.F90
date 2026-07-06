! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

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
  ! Per-thread FFT scratch. tfft/invfft fill cplxin, run the shared plan via the new-array execute
  ! fftw_execute_dft(plan, in, out), and read cplxout. Concurrent tfft/invfft calls (the parallel
  ! Jacobian DOF loop) must not share these buffers, so they are threadprivate and allocated per
  ! thread in set_fftw_plan. FFTW new-array execute is reentrant on a shared plan provided each call
  ! uses its own equally-sized buffers, so planf/planb and nt_plan/nz_plan stay shared.
  complex(c_double_complex), allocatable, public :: cplxin(:, :)!< FFTW-related (?)
  complex(c_double_complex), allocatable, public :: cplxout(:, :)!< FFTW-related (?)
  !$OMP THREADPRIVATE(cplxin, cplxout)
end module fftw_interface_mod
