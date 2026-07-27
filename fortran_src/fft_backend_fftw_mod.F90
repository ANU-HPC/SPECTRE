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
  ! Per-thread FFT scratch, passed explicitly to fft_forward/fft_backward. Concurrent tfft/invfft
  ! calls (the parallel Jacobian DOF loop) must not share these buffers, so they are threadprivate
  ! and allocated per thread in fft_init; planf/planb and nt_plan/nz_plan stay shared.
  !
  ! They come from fftw_alloc_complex rather than a Fortran allocate because fftw_execute_dft is
  ! the new-array execute: it is reentrant on a shared plan, but only if the arrays it is handed
  ! match the planned ones in size, stride AND SIMD alignment. Plain allocate guarantees no
  ! particular alignment, so a worker thread's buffer could be less aligned than the master's
  ! planned one and FFTW would then run its aligned kernels on unaligned data. FFTW's allocator
  ! returns uniformly SIMD-aligned memory, which upholds that contract without having to fall
  ! back to FFTW_UNALIGNED plans.
  complex(c_double_complex), pointer, public :: cplxin(:, :) => null()
  complex(c_double_complex), pointer, public :: cplxout(:, :) => null()
  type(c_ptr) :: cin_mem = c_null_ptr
  type(c_ptr) :: cout_mem = c_null_ptr
  !$OMP THREADPRIVATE(cplxin, cplxout, cin_mem, cout_mem)

  public :: c_double_complex, fft_init, fft_forward, fft_backward

  type(c_ptr) :: planf = c_null_ptr
  type(c_ptr) :: planb = c_null_ptr

contains

  subroutine alloc_scratch(nt, nz)
    ! Give the calling thread a fresh, SIMD-aligned (nt, nz) pair of scratch buffers.
    integer, intent(in) :: nt, nz

    call free_scratch()

    cin_mem = fftw_alloc_complex(int(nt, c_size_t) * int(nz, c_size_t))
    cout_mem = fftw_alloc_complex(int(nt, c_size_t) * int(nz, c_size_t))

    call c_f_pointer(cin_mem, cplxin, [nt, nz])
    call c_f_pointer(cout_mem, cplxout, [nt, nz])
  end subroutine alloc_scratch

  subroutine free_scratch()
    ! Release the calling thread's scratch buffers, if it has any.
    if (c_associated(cin_mem)) then
      call fftw_free(cin_mem)
      cin_mem = c_null_ptr
    end if
    if (c_associated(cout_mem)) then
      call fftw_free(cout_mem)
      cout_mem = c_null_ptr
    end if

    nullify(cplxin)
    nullify(cplxout)
  end subroutine free_scratch

  subroutine fft_init(nt, nz)
    ! (Re)allocate the transform buffers for an (nt, nz) grid and plan on them
    ! (FFTW_MEASURE overwrites the buffers; they are only filled after this).
    ! Plan creation is not thread-safe and FFTW_MEASURE writes into its buffers, so the shared
    ! plans are created serially on the master's buffers; every other thread then gets its own
    ! equally-sized, equally-aligned scratch.
    integer, intent(in) :: nt, nz

    if (nt /= nt_plan .or. nz /= nz_plan) then
      nt_plan = nt
      nz_plan = nz

      call alloc_scratch(nt, nz)

      ! Fortran column-major (nt, nz) is an (nz x nt) row-major array to FFTW.
      if (c_associated(planf)) call fftw_destroy_plan(planf)
      planf = fftw_plan_dft_2d(nz, nt, cplxin, cplxout, FFTW_FORWARD, FFTW_MEASURE + FFTW_DESTROY_INPUT)

      if (c_associated(planb)) call fftw_destroy_plan(planb)
      planb = fftw_plan_dft_2d(nz, nt, cplxin, cplxout, FFTW_BACKWARD, FFTW_MEASURE + FFTW_DESTROY_INPUT)
    end if

    ! Give every other thread its own buffer too; size-check (not thread id) so the master's
    ! freshly-planned buffer is never reallocated.
    !$OMP PARALLEL DEFAULT(shared)
    if (associated(cplxin)) then
      if (size(cplxin, 1) /= nt_plan .or. size(cplxin, 2) /= nz_plan) call free_scratch()
    end if
    if (.not. associated(cplxin)) call alloc_scratch(nt_plan, nz_plan)
    !$OMP END PARALLEL
  end subroutine fft_init

  subroutine fft_forward(cin, cout)
    ! cin is intent(inout): the plans carry FFTW_DESTROY_INPUT, so the input is clobbered.
    complex(c_double_complex), contiguous, intent(inout) :: cin(:, :)
    complex(c_double_complex), contiguous, intent(out) :: cout(:, :)

    call fftw_execute_dft(planf, cin, cout)
  end subroutine fft_forward

  subroutine fft_backward(cin, cout)
    complex(c_double_complex), contiguous, intent(inout) :: cin(:, :)
    complex(c_double_complex), contiguous, intent(out) :: cout(:, :)

    call fftw_execute_dft(planb, cin, cout)
  end subroutine fft_backward

end module fft_backend_mod
