! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

module fft_backend_mod
  ! allow(C121)
  !
  ! FFTW3-API implementation of SPECTRE's 2D complex-to-complex FFT, used by
  ! numerical_mod (tfft/invfft). fft_backend_mod is provided by exactly one
  ! implementation file, selected by the SPECTRE_FFT_BACKEND CMake option;
  ! the other implementation is fft_backend_pocketfft_mod.F90. This one is
  ! compiled for two backends that differ only in where CMake resolves
  ! fftw3.f03 and the library from:
  !   fftw : FFTW (GPL)
  !   mkl  : Intel oneMKL's drop-in FFTW3 interface (no GPL code involved)
  !
  ! Forward (fft_forward) uses the FFTW_FORWARD convention (exp(-i...)),
  ! backward (fft_backward) uses FFTW_BACKWARD (exp(+i...)); both unnormalized,
  ! so the normalization done in numerical_mod is unchanged.
  use iso_c_binding

  implicit none(type, external)

  include 'fftw3.f03'

  private

  integer, public :: nt_plan = 0, nz_plan = 0
  ! Per-thread FFT scratch. tfft/invfft fill cplxin, run the shared plan via the new-array execute
  ! fftw_execute_dft(plan, in, out), and read cplxout. Concurrent tfft/invfft calls (the parallel
  ! Jacobian DOF loop) must not share these buffers, so they are threadprivate and allocated per
  ! thread in fft_init. FFTW new-array execute is reentrant on a shared plan provided each call
  ! uses its own equally-sized buffers, so planf/planb and nt_plan/nz_plan stay shared.
  complex(c_double_complex), allocatable, public :: cplxin(:, :)
  complex(c_double_complex), allocatable, public :: cplxout(:, :)
  !$OMP THREADPRIVATE(cplxin, cplxout)

  public :: c_double_complex, fft_init, fft_forward, fft_backward

  type(c_ptr) :: planf = c_null_ptr
  type(c_ptr) :: planb = c_null_ptr

contains

  subroutine fft_init(nt, nz)
    ! (Re)allocate the transform buffers for an (nt, nz) grid and plan on them
    ! (FFTW_MEASURE overwrites the buffers; they are only filled after this).
    ! Plan creation is not thread-safe and FFTW_MEASURE writes into its buffers, so the shared
    ! plans are created serially on the master's buffers; every other thread then gets its own
    ! equally-sized scratch (cplxin/cplxout are threadprivate).
    use memory_allocation_mod, only: nallocate

    integer, intent(in) :: nt, nz

    if (nt /= nt_plan .or. nz /= nz_plan) then
      nt_plan = nt
      nz_plan = nz

      if (allocated(cplxin)) deallocate(cplxin)
      if (allocated(cplxout)) deallocate(cplxout)

      call nallocate(cplxin, 1, nt, 1, nz)
      call nallocate(cplxout, 1, nt, 1, nz)

      ! Fortran column-major (nt, nz) is an (nz x nt) row-major array to FFTW.
      if (c_associated(planf)) call fftw_destroy_plan(planf)
      planf = fftw_plan_dft_2d(nz, nt, cplxin(:, :), cplxout(:, :), FFTW_FORWARD, FFTW_MEASURE + FFTW_DESTROY_INPUT)

      if (c_associated(planb)) call fftw_destroy_plan(planb)
      planb = fftw_plan_dft_2d(nz, nt, cplxin(:, :), cplxout(:, :), FFTW_BACKWARD, FFTW_MEASURE + FFTW_DESTROY_INPUT)
    end if

    ! Give every other thread its own buffer too; size-check (not thread id) so the master's
    ! freshly-planned buffer is never reallocated. New-array execute only requires matching size.
    !$OMP PARALLEL DEFAULT(shared)
    if (allocated(cplxin)) then
      if (size(cplxin, 1) /= nt_plan .or. size(cplxin, 2) /= nz_plan) deallocate(cplxin)
    end if
    if (allocated(cplxout)) then
      if (size(cplxout, 1) /= nt_plan .or. size(cplxout, 2) /= nz_plan) deallocate(cplxout)
    end if
    if (.not. allocated(cplxin)) allocate(cplxin(1:nt_plan, 1:nz_plan))
    if (.not. allocated(cplxout)) allocate(cplxout(1:nt_plan, 1:nz_plan))
    !$OMP END PARALLEL
  end subroutine fft_init

  subroutine fft_forward()
    call fftw_execute_dft(planf, cplxin(:, :), cplxout(:, :))
  end subroutine fft_forward

  subroutine fft_backward()
    call fftw_execute_dft(planb, cplxin(:, :), cplxout(:, :))
  end subroutine fft_backward

end module fft_backend_mod
