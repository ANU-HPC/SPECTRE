! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module intghs_mod

  use omp_lib, only: omp_in_parallel
  use precisions_mod, only: rkind
  use numerical_mod, only: tfft, invfft
  use base_functions_mod, only: get_cheby, get_zernike
  use input_list_mod, only: mpol, Wintghs, Wmacros, enforce_stell_sym, Mpol, Lrad
  ! Tss/Dtc/Dzc/Tsc/Dts/Dzs, Ttc/Tzc/Tts/Tzs and guvijsave are NOT imported: intghs reads/writes
  ! them in the per-thread workspace (pws) via local pointers of the same name. The Ttc/Tzc/Tts/Tzs
  ! (helicity/dMD) projections are written only on the .not.dBdX%L branch (the matrix-free matvec);
  ! the perturbed-RHS path takes dBdX%L=.true. and skips them.
  use allglobal_mod, only: im, in, gaussianweight, gaussianabscissae, &
      pi2pi2nfphalf, dBdX, Lsavedguvij, Ate, Ato, Aze, Azo, &
      mn, Iquad
  use memory_allocation_mod, only: sallocate, nallocate
  use metric_mod, only: compute_guvijsave

  implicit none(type, external)

  private
  public :: intghs, intghs_workspace_init, intghs_workspace_destroy, wk, intghs_alloc_wk_perturbed

  type intghs_workspace
    real(rkind), allocatable :: efmn(:, :)
    real(rkind), allocatable :: ofmn(:, :)
    real(rkind), allocatable :: cfmn(:, :)
    real(rkind), allocatable :: sfmn(:, :)
    real(rkind), allocatable :: evmn(:, :)
    real(rkind), allocatable :: odmn(:, :)
    real(rkind), allocatable :: ijreal(:, :)
    real(rkind), allocatable :: jireal(:, :)
    real(rkind), allocatable :: jkreal(:, :)
    real(rkind), allocatable :: kjreal(:, :)
    real(rkind), allocatable :: Bloweremn(:, :, :)
    real(rkind), allocatable :: Bloweromn(:, :, :)
    real(rkind), allocatable :: gBupper(:, :, :)
    real(rkind), allocatable :: Blower(:, :, :)
    real(rkind), allocatable :: basis(:, :, :, :)
  end type intghs_workspace

  type(intghs_workspace) :: wk !< This is an instance of the intghs_workspace type.

contains

  subroutine intghs(lquad, mn, lvol, lrad, idx, nt_loc, nz_loc, Lcoordinatesingularity, pws)
    use perturbed_workspace_mod, only: perturbed_rhs_workspace
    integer, intent(in) :: lquad, mn, lvol, lrad, idx,  nt_loc, nz_loc
    logical, intent(in) :: Lcoordinatesingularity
    ! Perturbed-path only: the per-thread workspace is required (target: pointer aliasing).
    type(perturbed_rhs_workspace), intent(inout), target :: pws

    integer :: jquad, ll, ll1, ii, jj, ideriv, ifail, Lcurvature
    integer :: mi, ni, bid

    real(rkind) :: lss, jthweight, sbar, dfactor, ik, w(1:lquad)

    integer :: ntz_loc

    ! Local pointers aliasing the per-thread workspace, shadowing the module-global names used
    ! in the body: wk (the intghs scratch), the projection arrays, and guvijsave. This lets the
    ! body run unchanged while reading/writing only thread-local storage.
    type(perturbed_rhs_workspace), pointer :: wk
    real(rkind), pointer :: Tss(:, :), Dtc(:, :), Dzc(:, :), Tsc(:, :), Dts(:, :), Dzs(:, :)
    real(rkind), pointer :: Ttc(:, :), Tzc(:, :), Tts(:, :), Tzs(:, :)
    real(rkind), pointer :: guvijsave(:, :, :, :)

    wk => pws
    Tss => pws%Tss; Dtc => pws%Dtc; Dzc => pws%Dzc
    Tsc => pws%Tsc; Dts => pws%Dts; Dzs => pws%Dzs
    Ttc => pws%Ttc; Tzc => pws%Tzc; Tts => pws%Tts; Tzs => pws%Tzs
    guvijsave => pws%guvijsave

    ntz_loc = nt_loc * nz_loc

    ! Tss = 0.0_rkind
    ! Dtc = 0.0_rkind
    ! Dzc = 0.0_rkind

    ! if (.not.dBdX%L) then
    !   Ttc = 0.0_rkind
    !   Tzc = 0.0_rkind
    ! end if

    ! if (.not.enforce_stell_sym) then
    !   Tsc = 0.0_rkind
    !   Dts = 0.0_rkind
    !   Dzs = 0.0_rkind
    !   if (.not.dBdX%L) then
    !     Tts = 0.0_rkind
    !     Tzs = 0.0_rkind
    !   end if

    ! end if !.not.enforce_stell_sym

    if (dBdX%L) then
      Lcurvature = 3
      ideriv = 1
    else
      Lcurvature = 1
      ideriv = 0
    end if

    if (.not.Lsavedguvij) then
      call compute_guvijsave(lquad, lvol, ideriv, Lcurvature, nt_loc, nz_loc, Lcoordinatesingularity, &
          pws%guvijsave, pws%Rij, pws%Zij, pws%sg, pws%inv_jac, pws%guvij)
    end if

    do jquad = 1, lquad
      lss = gaussianabscissae(jquad, lvol)
      jthweight = gaussianweight(jquad, lvol)
      sbar = (lss + 1.0_rkind) * 1.0_rkind / 2.0_rkind
      if (Lcoordinatesingularity) then
        call get_zernike(sbar, lrad, mpol, wk%basis(0:lrad, 0:mpol, 0:1, jquad))! use Zernike polynomials
      else
        call get_cheby(lss, lrad, wk%basis(0:lrad, 0, 0:1, jquad))
      end if
    end do

    wk%ijreal(1:mn, 1:lquad) = 0.0_rkind
    wk%jkreal(1:mn, 1:lquad) = 0.0_rkind
    wk%efmn(1:mn, 1:lquad) = 0.0_rkind
    wk%cfmn(1:mn, 1:lquad) = 0.0_rkind
    wk%odmn(1:mn, 1:lquad) = 0.0_rkind

    if (.not.enforce_stell_sym) then
      wk%kjreal(1:mn, 1:lquad) = 0.0_rkind
      wk%jireal(1:mn, 1:lquad) = 0.0_rkind
      wk%ofmn(1:mn, 1:lquad) = 0.0_rkind
      wk%sfmn(1:mn, 1:lquad) = 0.0_rkind
      wk%evmn(1:mn, 1:lquad) = 0.0_rkind
    end if

    wk%Blower(1:ntz_loc, 1:3, 1:lquad) = 0.0_rkind

    ! Parallelized for the forward-solve matvec (each jquad writes its own wk slices); the
    ! omp_in_parallel() guard keeps it serial when nested inside the Jacobian's per-DOF region.
    !$OMP PARALLEL DO IF(.not. omp_in_parallel()) DEFAULT(shared) PRIVATE(ii, mi, ni, ll) SCHEDULE(static)
    do jquad = 1, lquad

      ! A_t  : wk%ijreal, wk%jireal
      ! A_z  : wk%jkreal, wk%kjreal
      ! gB^s : wk%evmn, wk%odmn
      ! gB^t : wk%cfmn, wk%sfmn
      ! gB^z : wk%efmn, wk%ofmn

      do ii = 1, mn
        mi = im(ii)
        ni = in(ii)! loop over Fourier harmonics

        if (Lcoordinatesingularity) then
          do ll = mi, lrad, 2 ! loop over Zernike polynomials: Lrad is the radial resolution
            wk%efmn(ii, jquad) = wk%efmn(ii, jquad) + Ate(lvol, idx, ii)%s(ll) * (wk%basis(ll, mi, 1, jquad) * 1.0_rkind &
                &/ 2.0_rkind)
            wk%cfmn(ii, jquad) = wk%cfmn(ii, jquad) - Aze(lvol, idx, ii)%s(ll) * (wk%basis(ll, mi, 1, jquad) * 1.0_rkind &
                &/ 2.0_rkind)
            wk%odmn(ii, jquad) = wk%odmn(ii, jquad) - Ate(lvol, idx, ii)%s(ll) * (wk%basis(ll, mi, 0, jquad)) * ni -&
                & Aze(lvol, idx, ii)%s(ll) * (wk%basis(ll, mi, 0, jquad)) * mi
            wk%ijreal(ii, jquad) = wk%ijreal(ii, jquad) + Ate(lvol, idx, ii)%s(ll) * wk%basis(ll, mi, 0, jquad)
            wk%jkreal(ii, jquad) = wk%jkreal(ii, jquad) + Aze(lvol, idx, ii)%s(ll) * wk%basis(ll, mi, 0, jquad)
            if (.not.enforce_stell_sym) then
              wk%ofmn(ii, jquad) = wk%ofmn(ii, jquad) + Ato(lvol, idx, ii)%s(ll) * (wk%basis(ll, mi, 1, jquad) *&
                  & 1.0_rkind / 2.0_rkind)
              wk%sfmn(ii, jquad) = wk%sfmn(ii, jquad) - Azo(lvol, idx, ii)%s(ll) * (wk%basis(ll, mi, 1, jquad) *&
                  & 1.0_rkind / 2.0_rkind)
              wk%evmn(ii, jquad) = wk%evmn(ii, jquad) + Ato(lvol, idx, ii)%s(ll) * (wk%basis(ll, mi, 0, jquad)) * ni +&
                  & Azo(lvol, idx, ii)%s(ll) * (wk%basis(ll, mi, 0, jquad)) * mi
              wk%jireal(ii, jquad) = wk%jireal(ii, jquad) + Ato(lvol, idx, ii)%s(ll) * wk%basis(ll, mi, 0, jquad)
              wk%kjreal(ii, jquad) = wk%kjreal(ii, jquad) + Azo(lvol, idx, ii)%s(ll) * wk%basis(ll, mi, 0, jquad)
            end if
          end do
        else
          do ll = 0, lrad ! loop over Chebyshev polynomials: Lrad is the radial resolution
            wk%efmn(ii, jquad) = wk%efmn(ii, jquad) + Ate(lvol, idx, ii)%s(ll) * (wk%basis(ll, 0, 1, jquad))
            wk%cfmn(ii, jquad) = wk%cfmn(ii, jquad) - Aze(lvol, idx, ii)%s(ll) * (wk%basis(ll, 0, 1, jquad))
            wk%odmn(ii, jquad) = wk%odmn(ii, jquad) - Ate(lvol, idx, ii)%s(ll) * (wk%basis(ll, 0, 0, jquad)) * ni -&
                & Aze(lvol, idx, ii)%s(ll) * (wk%basis(ll, 0, 0, jquad)) * mi
            wk%ijreal(ii, jquad) = wk%ijreal(ii, jquad) + Ate(lvol, idx, ii)%s(ll) * wk%basis(ll, 0, 0, jquad)
            wk%jkreal(ii, jquad) = wk%jkreal(ii, jquad) + Aze(lvol, idx, ii)%s(ll) * wk%basis(ll, 0, 0, jquad)
            if (.not.enforce_stell_sym) then
              wk%ofmn(ii, jquad) = wk%ofmn(ii, jquad) + Ato(lvol, idx, ii)%s(ll) * (wk%basis(ll, 0, 1, jquad))
              wk%sfmn(ii, jquad) = wk%sfmn(ii, jquad) - Azo(lvol, idx, ii)%s(ll) * (wk%basis(ll, 0, 1, jquad))
              wk%evmn(ii, jquad) = wk%evmn(ii, jquad) + Ato(lvol, idx, ii)%s(ll) * (wk%basis(ll, 0, 0, jquad)) * ni +&
                  & Azo(lvol, idx, ii)%s(ll) * (wk%basis(ll, 0, 0, jquad)) * mi
              wk%jireal(ii, jquad) = wk%jireal(ii, jquad) + Ato(lvol, idx, ii)%s(ll) * wk%basis(ll, 0, 0, jquad)
              wk%kjreal(ii, jquad) = wk%kjreal(ii, jquad) + Azo(lvol, idx, ii)%s(ll) * wk%basis(ll, 0, 0, jquad)
            end if
          end do
        end if ! Lcoordinatesingularity
      end do

    end do
    !$OMP END PARALLEL DO

    ! The per-quadrature FFTs dominate a matrix-free matvec. cplxin/cplxout are threadprivate and
    ! each jquad writes its own wk slices, so this loop parallelizes safely. Guard with
    ! omp_in_parallel(): it threads the forward-solve matvec (called serially) but stays serial when
    ! nested inside the Jacobian's per-DOF parallel region.
    !$OMP PARALLEL DO IF(.not. omp_in_parallel()) DEFAULT(shared) PRIVATE(ii, jj, ifail) SCHEDULE(static)
    do jquad = 1, lquad
      if (enforce_stell_sym) then
        call invfft(mn, im, in, wk%efmn(1:mn, jquad), 0.0_rkind * (wk%efmn(1:mn, jquad)) , wk%cfmn(1:mn, jquad), 0.0_rkind * (wk%&
            efmn(1:mn, jquad)) , wk%gBupper(1:ntz_loc, 3, jquad), wk%gBupper(1:ntz_loc, 2, jquad))
        call invfft(mn, im, in, 0.0_rkind * (wk%efmn(1:mn, jquad)) , wk%odmn(1:mn, jquad), wk%cfmn(1:mn, jquad), 0.0_rkind * (wk%&
            efmn(1:mn, jquad)) , wk%gBupper(1:ntz_loc, 1, jquad), wk%gBupper(1:ntz_loc, 2, jquad))
      else
        call invfft(mn, im, in, wk%efmn(1:mn, jquad), wk%ofmn(1:mn, jquad), wk%cfmn(1:mn, jquad), wk%sfmn(1:mn, jquad), wk%&
            gBupper(1:ntz_loc, 3, jquad), wk%gBupper(1:ntz_loc, 2, jquad))
        call invfft(mn, im, in, wk%evmn(1:mn, jquad), wk%odmn(1:mn, jquad), wk%cfmn(1:mn, jquad), wk%sfmn(1:mn, jquad), wk%&
            gBupper(1:ntz_loc, 1, jquad), wk%gBupper(1:ntz_loc, 2, jquad))
      end if

      do ii = 1, 3
        do jj = 1, 3
          wk%Blower(1:ntz_loc, ii, jquad) = wk%Blower(1:ntz_loc, ii, jquad) + wk%gBupper(1:ntz_loc, jj, jquad) * guvijsave(1:ntz_loc, jj, ii, jquad)
        end do
      end do

      ifail = 0

      call tfft(wk%Blower(1:ntz_loc, 2, jquad), wk%Blower(1:ntz_loc, 3, jquad), mn, im(1:mn), in(1:mn), wk%Bloweremn(1:mn&
          & , 2, jquad), wk%Bloweromn(1:mn, 2, jquad), wk%Bloweremn(1:mn, 3, jquad), wk%Bloweromn(1:mn, 3, jquad), ifail)
      call tfft(wk%Blower(1:ntz_loc, 1, jquad), wk%Blower(1:ntz_loc, 3, jquad), mn, im(1:mn), in(1:mn), wk%Bloweromn(1:mn&
          & , 1, jquad), wk%Bloweremn(1:mn, 1, jquad), wk%Bloweremn(1:mn, 3, jquad), wk%Bloweromn(1:mn, 3, jquad), ifail)
      wk%Bloweremn(1, 1, jquad) = 0.0_rkind
      wk%Bloweromn(1, 2:3, jquad) = 0.0_rkind

    end do
    !$OMP END PARALLEL DO

    w(1:lquad) = gaussianweight(1:lquad, lvol)

    ! Parallelized over modes for the forward-solve matvec (each ii writes its own projection
    ! columns); serial when nested inside the Jacobian's per-DOF region (omp_in_parallel() guard).
    !$OMP PARALLEL DO IF(.not. omp_in_parallel()) DEFAULT(shared) PRIVATE(ll, ll1, ik, bid, dfactor) SCHEDULE(static)
    do ii = 1, mn

      do ll = 0, lrad

        if (ii == 1) then
          ik = 2.0_rkind
        else
          ik = 1.0_rkind
        end if

        if (Lcoordinatesingularity) then
          if (ll < im(ii) .or. mod(ll + im(ii), 2)/= 0) cycle
          ll1 = (ll - mod(ll, 2)) / 2 ! shrinked dof for Zernike
          bid = im(ii)
          dfactor = 1.0_rkind / 2.0_rkind
        else
          ll1 = ll
          bid = 0
          dfactor = 1.0_rkind
        end if

        Tss(ll1, ii) = sum(wk%basis(ll, bid, 0, 1:lquad) * wk%Bloweremn(ii, 1, 1:lquad) * w) * ik * pi2pi2nfphalf
        Dtc(ll1, ii) = sum(wk%basis(ll, bid, 1, 1:lquad) * wk%Bloweremn(ii, 2, 1:lquad) * w) * dfactor * ik * pi2pi2nfphalf
        Dzc(ll1, ii) = sum(wk%basis(ll, bid, 1, 1:lquad) * wk%Bloweremn(ii, 3, 1:lquad) * w) * dfactor * ik * pi2pi2nfphalf

        if (.not.enforce_stell_sym) then
          Tsc(ll1, ii) = sum(wk%basis(ll, bid, 0, 1:lquad) * wk%Bloweromn(ii, 1, 1:lquad) * w) * ik * pi2pi2nfphalf
          Dts(ll1, ii) = sum(wk%basis(ll, bid, 1, 1:lquad) * wk%Bloweromn(ii, 2, 1:lquad) * w) * dfactor * ik * pi2pi2nfphalf
          Dzs(ll1, ii) = sum(wk%basis(ll, bid, 1, 1:lquad) * wk%Bloweromn(ii, 3, 1:lquad) * w) * dfactor * ik * pi2pi2nfphalf
        end if

        if (dBdX%L) cycle ! dMD matrix does not depend on geometry
        Ttc(ll1, ii) = (sum(wk%basis(ll, bid, 0, 1:lquad) * wk%cfmn(ii, 1:lquad) * w) &
            + sum(wk%basis(ll, bid, 1, 1:lquad) * wk%jkreal(ii,&
            & 1:lquad) * w) * dfactor) * ik * pi2pi2nfphalf
        Tzc(ll1, ii) = (sum(wk%basis(ll, bid, 0, 1:lquad) * wk%efmn(ii, 1:lquad) * w) &
            - sum(wk%basis(ll, bid, 1, 1:lquad) * wk%ijreal(ii,&
            & 1:lquad) * w) * dfactor) * ik * pi2pi2nfphalf

        if (.not.enforce_stell_sym) then
          Tts(ll1, ii) = (sum(wk%basis(ll, bid, 0, 1:lquad) * wk%sfmn(ii, 1:lquad) * w) &
              + sum(wk%basis(ll, bid, 1, 1:lquad) * wk%kjreal(ii,&
              & 1:lquad) * w) * dfactor) * ik * pi2pi2nfphalf
          Tzs(ll1, ii) = (sum(wk%basis(ll, bid, 0, 1:lquad) * wk%ofmn(ii, 1:lquad) * w) &
              - sum(wk%basis(ll, bid, 1, 1:lquad) * wk%jireal(ii,&
              & 1:lquad) * w) * dfactor) * ik * pi2pi2nfphalf
        end if

      end do
    end do

  end subroutine intghs

  subroutine intghs_workspace_init(lvol, ntz_loc)
    integer, intent(in) :: lvol, ntz_loc
    integer :: lquad

    lquad = Iquad(lvol)

    call sallocate(wk%gBupper, 1, ntz_loc, 1, 3, 1, lquad, 0.0_rkind)
    call sallocate(wk%Blower, 1, ntz_loc, 1, 3, 1, lquad, 0.0_rkind)
    call sallocate(wk%Bloweremn, 1, mn, 1, 3, 1, lquad, 0.0_rkind)
    call sallocate(wk%Bloweromn, 1, mn, 1, 3, 1, lquad, 0.0_rkind)
    call sallocate(wk%efmn, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%ofmn, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%evmn, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%odmn, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%cfmn, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%sfmn, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%ijreal, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%jkreal, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%jireal, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%kjreal, 1, mn, 1, lquad, 0.0_rkind)
    call sallocate(wk%basis, 0, Lrad(lvol), 0, mpol, 0, 1, 1, lquad, 0.0_rkind)

  end subroutine intghs_workspace_init

  subroutine intghs_workspace_destroy()
    deallocate(wk%gBupper)
    deallocate(wk%Blower)
    deallocate(wk%Bloweremn)
    deallocate(wk%Bloweromn)
    deallocate(wk%efmn)
    deallocate(wk%ofmn)
    deallocate(wk%evmn)
    deallocate(wk%odmn)
    deallocate(wk%cfmn)
    deallocate(wk%sfmn)
    deallocate(wk%ijreal)
    deallocate(wk%jkreal)
    deallocate(wk%jireal)
    deallocate(wk%kjreal)
    deallocate(wk%basis)

  end subroutine intghs_workspace_destroy

  subroutine intghs_alloc_wk_perturbed(Ntz, mn, max_iquad, max_lrad, mpol, alloc_nonsym)
    integer, intent(in) :: Ntz, mn, max_iquad, max_lrad, mpol
    logical, intent(in) :: alloc_nonsym

    call nallocate(wk%gBupper, 1, Ntz, 1, 3, 1, max_iquad)
    call nallocate(wk%Blower, 1, Ntz, 1, 3, 1, max_iquad)
    call nallocate(wk%Bloweremn, 1, mn, 1, 3, 1, max_iquad)
    call nallocate(wk%Bloweromn, 1, mn, 1, 3, 1, max_iquad)
    call nallocate(wk%efmn, 1, mn, 1, max_iquad)
    call nallocate(wk%odmn, 1, mn, 1, max_iquad)
    call nallocate(wk%cfmn, 1, mn, 1, max_iquad)
    call nallocate(wk%ijreal, 1, mn, 1, max_iquad)
    call nallocate(wk%jkreal, 1, mn, 1, max_iquad)
    call nallocate(wk%basis, 0, max_lrad, 0, mpol, 0, 1, 1, max_iquad)
    if (alloc_nonsym) then
      call nallocate(wk%ofmn, 1, mn, 1, max_iquad)
      call nallocate(wk%evmn, 1, mn, 1, max_iquad)
      call nallocate(wk%sfmn, 1, mn, 1, max_iquad)
      call nallocate(wk%kjreal, 1, mn, 1, max_iquad)
      call nallocate(wk%jireal, 1, mn, 1, max_iquad)
    end if
  end subroutine intghs_alloc_wk_perturbed

end module intghs_mod
