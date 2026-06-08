! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module fft_backend_mod
  ! allow(C121)
  !
  ! Backend-neutral 2D complex-to-complex FFT used by numerical_mod (tfft/invfft).
  ! Exactly one backend is compiled in, selected by the SPECTRE_FFT_BACKEND CMake
  ! option which defines one of FFT_BACKEND_FFTW / FFT_BACKEND_POCKETFFT /
  ! FFT_BACKEND_PFFFT. The public buffers cplxin/cplxout and the routines
  ! fft_init/fft_forward/fft_backward present an identical interface to the rest
  ! of the core regardless of backend.
  !
  ! Forward (fft_forward) uses the FFTW_FORWARD convention (exp(-i...)),
  ! backward (fft_backward) uses FFTW_BACKWARD (exp(+i...)); both unnormalized,
  ! so the normalization done in numerical_mod is unchanged.
  use iso_c_binding

  implicit none(type, external)

#ifdef FFT_BACKEND_FFTW
  include 'fftw3.f03'
#endif

  private

  integer, public :: nt_plan = 0, nz_plan = 0
  complex(c_double_complex), allocatable, public :: cplxin(:, :)
  complex(c_double_complex), allocatable, public :: cplxout(:, :)

  public :: c_double_complex, fft_init, fft_forward, fft_backward

#ifdef FFT_BACKEND_FFTW
  type(c_ptr) :: planf = c_null_ptr
  type(c_ptr) :: planb = c_null_ptr
#else
  interface
    subroutine spectre_fft2d_c2c(dir, nt, nz, cin, cout) bind(C, name="spectre_fft2d_c2c")
      import :: c_int, c_double_complex
      integer(c_int), value :: dir, nt, nz
      complex(c_double_complex), intent(in) :: cin(*)
      complex(c_double_complex), intent(out) :: cout(*)
    end subroutine spectre_fft2d_c2c
  end interface
#endif

contains

  subroutine fft_init(nt, nz)
    use memory_allocation_mod, only: nallocate

    integer, intent(in) :: nt, nz

    ! Correct plan/buffers already in place
    if (nt == nt_plan .and. nz == nz_plan) return

    nt_plan = nt
    nz_plan = nz

    if (allocated(cplxin)) deallocate(cplxin)
    if (allocated(cplxout)) deallocate(cplxout)

    call nallocate(cplxin, 1, nt, 1, nz)
    call nallocate(cplxout, 1, nt, 1, nz)

#ifdef FFT_BACKEND_FFTW
    if (c_associated(planf)) call fftw_destroy_plan(planf)
    planf = fftw_plan_dft_2d(nz, nt, cplxin(:, :), cplxout(:, :), FFTW_FORWARD, FFTW_MEASURE + FFTW_DESTROY_INPUT)

    if (c_associated(planb)) call fftw_destroy_plan(planb)
    planb = fftw_plan_dft_2d(nz, nt, cplxin(:, :), cplxout(:, :), FFTW_BACKWARD, FFTW_MEASURE + FFTW_DESTROY_INPUT)
#endif
  end subroutine fft_init

  subroutine fft_forward()
#ifdef FFT_BACKEND_FFTW
    call fftw_execute_dft(planf, cplxin(:, :), cplxout(:, :))
#else
    call spectre_fft2d_c2c(1_c_int, int(nt_plan, c_int), int(nz_plan, c_int), cplxin, cplxout)
#endif
  end subroutine fft_forward

  subroutine fft_backward()
#ifdef FFT_BACKEND_FFTW
    call fftw_execute_dft(planb, cplxin(:, :), cplxout(:, :))
#else
    call spectre_fft2d_c2c(-1_c_int, int(nt_plan, c_int), int(nz_plan, c_int), cplxin, cplxout)
#endif
  end subroutine fft_backward

end module fft_backend_mod
