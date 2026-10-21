! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

module fft_backend_mod
  ! allow(C121)
  !
  ! PocketFFT implementation of SPECTRE's 2D complex-to-complex FFT, used by
  ! numerical_mod (tfft/invfft). fft_backend_mod is provided by exactly one
  ! implementation file, selected by the SPECTRE_FFT_BACKEND CMake option;
  ! the other implementation is fft_backend_fftw_mod.F90. PocketFFT
  ! (BSD-3-Clause) is a header-only C++ library, so this module calls the
  ! small C++ bridge extern/fft_shim/spectre_fft2d_pocketfft.cpp through the
  ! bind(C) interface below.
  !
  ! Forward (fft_forward) uses the FFTW_FORWARD convention (exp(-i...)),
  ! backward (fft_backward) uses FFTW_BACKWARD (exp(+i...)); both unnormalized,
  ! so the normalization done in numerical_mod is unchanged.
  use iso_c_binding

  implicit none(type, external)

  private

  integer, public :: nt_plan = 0, nz_plan = 0
  ! Per-thread FFT scratch. Concurrent tfft/invfft calls (the parallel Jacobian DOF loop) must
  ! not share these buffers, so they are threadprivate and allocated per thread in fft_init.
  ! The PocketFFT bridge itself is stateless per call (its internal 1D-plan cache is
  ! thread-local), so nt_plan/nz_plan stay shared.
  complex(c_double_complex), allocatable, public :: cplxin(:, :)
  complex(c_double_complex), allocatable, public :: cplxout(:, :)
  !$OMP THREADPRIVATE(cplxin, cplxout)

  public :: c_double_complex, fft_init, fft_forward, fft_backward

  interface
    ! Unnormalized 2D c2c transform of the (nt, nz) column-major
    ! complex(double) grid cin -> cout; dir > 0 forward, dir < 0 backward.
    subroutine spectre_fft2d_c2c(dir, nt, nz, cin, cout) bind(C, name="spectre_fft2d_c2c")
      import :: c_int, c_double_complex
      integer(c_int), value :: dir, nt, nz
      complex(c_double_complex), intent(in) :: cin(*)
      complex(c_double_complex), intent(out) :: cout(*)
    end subroutine spectre_fft2d_c2c
  end interface

contains

  subroutine fft_init(nt, nz)
    ! (Re)allocate the transform buffers for an (nt, nz) grid. PocketFFT is
    ! plan-less from the caller's side (it caches its 1D plans internally),
    ! so there is nothing else to prepare. cplxin/cplxout are threadprivate:
    ! the master's scratch is (re)sized here, every other thread's in the
    ! parallel region below.
    use memory_allocation_mod, only: nallocate

    integer, intent(in) :: nt, nz

    if (nt /= nt_plan .or. nz /= nz_plan) then
      nt_plan = nt
      nz_plan = nz

      if (allocated(cplxin)) deallocate(cplxin)
      if (allocated(cplxout)) deallocate(cplxout)

      call nallocate(cplxin, 1, nt, 1, nz)
      call nallocate(cplxout, 1, nt, 1, nz)
    end if

    ! Give every other thread its own buffer too; size-check (not thread id) so the master's
    ! buffer is never needlessly reallocated.
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

  subroutine fft_forward(cin, cout)
    ! PocketFFT does not touch its input, but cin is intent(inout) to match the FFTW backend,
    ! whose plans carry FFTW_DESTROY_INPUT: callers must work against either implementation.
    complex(c_double_complex), contiguous, intent(inout) :: cin(:, :)
    complex(c_double_complex), contiguous, intent(out) :: cout(:, :)

    call spectre_fft2d_c2c(1_c_int, int(size(cin, 1), c_int), int(size(cin, 2), c_int), cin, cout)
  end subroutine fft_forward

  subroutine fft_backward(cin, cout)
    complex(c_double_complex), contiguous, intent(inout) :: cin(:, :)
    complex(c_double_complex), contiguous, intent(out) :: cout(:, :)

    call spectre_fft2d_c2c(-1_c_int, int(size(cin, 1), c_int), int(size(cin, 2), c_int), cin, cout)
  end subroutine fft_backward

end module fft_backend_mod
