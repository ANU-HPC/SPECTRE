! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

!> Laplace-virtual-casing matrix for the direct (non-Picard) free-boundary solve.
!>
!> The normal field produced on the computational boundary by the plasma is, via the virtual-casing
!> principle, a linear operator on the vacuum-volume vector-potential coefficients: the tangential
!> field on the plasma-boundary side (s = -1) of the vacuum annulus is linear in the Ate/Aze
!> (Ato/Azo) coefficients with radial weight TT(ll,0,1) (Henneberg, Hudson, Pfefferle & Helander,
!> J. Plasma Phys. 87, 905870226 (2021), eq. 4.4). This module discretizes that operator as the
!> dense matrix dVC, whose non-zero rows are the Lme (and, without stellarator symmetry, Lmf)
!> normal-field constraint slots of the vacuum Beltrami system. Subtracting dVC from dMA and
!> dropping iBns/iBnc from the right-hand side makes a single linear solve self-consistent, so the
!> mfreeits Picard iteration is not needed. The derivative of dVC with respect to one plasma-boundary
!> harmonic (paper eq. 4.5) feeds the analytic force Jacobian through get_perturbed_rhs.
!>
!> The kernel replicates freeb_mod::surfacecurrent / casinggrid exactly (same vcasingeps
!> regularization, same pi2^2 * virtualcasingfactor / Ctz normalization). tfft is not used because
!> the derivative runs inside OpenMP parallel regions and the FFTW buffers are shared; vc_project
!> is a slow discrete Fourier projection with identical normalization.
module vc_matrix_mod

  use precisions_mod, only: rkind
  use constants_mod, only: pi2
  use errors_mod, only: fatal
  use memory_allocation_mod, only: nallocate, sallocate
  use input_list_mod, only: nvol, Lrad, nfp, cteta, czeta, enforce_stell_sym, vcasingeps, Igeometry
  use allglobal_mod, only: mn, im, in, Mvol, Nt, Nz, Ntz, pi2nfp, &
      iRbc, iZbs, iRbs, iZbc, TT, Ate, Aze, Ato, Azo, Lme, Lmf, dVC, &
      Dxyz, Nxyz, virtualcasingfactor, dBdX, first_free_bound

  implicit none(type, external)

  private
  public :: vc_tables_init, vc_tables_destroy, vc_assemble_dvc, vc_prepare_deriv, vc_apply_a0, vc_deriv_rhs_add, vc_project

  integer :: Cnt = 0 !< theta resolution of the source (plasma-boundary) quadrature grid
  integer :: Cnz = 0 !< zeta resolution of the source grid over the full torus (= czeta * nfp)
  integer, public :: Ctz = 0 !< total number of source quadrature points (= Cnt * Cnz)

  !> source-grid trig tables: cos/sin(m*theta - n*zeta) on the full-torus plasma-boundary grid
  real(rkind), allocatable :: kjcos(:, :)!< (1:mn, 1:Ctz)
  real(rkind), allocatable :: kjsin(:, :)!< (1:mn, 1:Ctz)
  real(rkind), allocatable :: kjczeta(:)!< (1:Ctz) cos(zeta)
  real(rkind), allocatable :: kjszeta(:)!< (1:Ctz) sin(zeta)

  !> computational-boundary projection tables replicating the tfft normalization:
  !> weight 2/Ntz for (m,n) /= (0,0) and 1/Ntz for (0,0)
  real(rkind), allocatable :: prjcos(:, :)!< (1:Ntz, 1:mn)
  real(rkind), allocatable :: prjsin(:, :)!< (1:Ntz, 1:mn)

  !> zeroth-order source geometry at s = -1 of the vacuum volume; refreshed by vc_refresh_geometry
  real(rkind), allocatable :: srcR(:, :)!< (1:Ctz, 0:3) R and its s, theta, zeta derivatives
  real(rkind), allocatable :: srcZ(:, :)!< (1:Ctz, 0:3) Z and its s, theta, zeta derivatives
  real(rkind), allocatable :: srcpos(:, :)!< (1:Ctz, 1:3) Cartesian position
  real(rkind), allocatable :: srcxt(:, :)!< (1:Ctz, 1:3) Cartesian theta tangent
  real(rkind), allocatable :: srcxz(:, :)!< (1:Ctz, 1:3) Cartesian zeta tangent
  real(rkind), allocatable :: srcgtt(:), srcgtz(:), srcgzz(:)!< (1:Ctz) metric elements
  real(rkind), allocatable :: srcinvsg(:)!< (1:Ctz) 1/sqrt(g)

  !> solution-dependent source fields from the ideriv = 0 vector potential; refreshed by vc_prepare_deriv
  real(rkind), allocatable :: srcgBut(:), srcgBuz(:)!< (1:Ctz) sqrt(g) B^theta, sqrt(g) B^zeta at s = -1
  real(rkind), allocatable :: srcBlt(:), srcBlz(:)!< (1:Ctz) covariant B_theta, B_zeta at s = -1

contains

  !> Build the geometry-independent trig tables; requires Dxyz/Nxyz (preset) to exist already.
  subroutine vc_tables_init()
    integer :: ii, jj, kk, kj, jk
    real(rkind) :: teta, zeta, arg

    call fatal(Igeometry /= 3, __FILE__, "the Laplace-virtual-casing matrix requires Igeometry = 3")

    call vc_tables_destroy()

    Cnt = cteta
    Cnz = czeta * nfp
    Ctz = Cnt * Cnz

    call nallocate(kjcos, 1, mn, 1, Ctz)
    call nallocate(kjsin, 1, mn, 1, Ctz)
    call nallocate(kjczeta, 1, Ctz)
    call nallocate(kjszeta, 1, Ctz)

    do kk = 0, Cnz - 1
      zeta = kk * pi2 / Cnz
      do jj = 0, Cnt - 1
        teta = jj * pi2 / Cnt
        kj = 1 + jj + kk * Cnt
        kjczeta(kj) = cos(zeta)
        kjszeta(kj) = sin(zeta)
        do ii = 1, mn
          arg = im(ii) * teta - in(ii) * zeta
          kjcos(ii, kj) = cos(arg)
          kjsin(ii, kj) = sin(arg)
        end do
      end do
    end do

    call nallocate(prjcos, 1, Ntz, 1, mn)
    call nallocate(prjsin, 1, Ntz, 1, mn)

    ! same grid angles as the Dxyz/Nxyz construction in preset and the same coefficient
    ! normalization as tfft; the (0,0) sine projection is identically zero
    do kk = 0, Nz - 1
      zeta = kk * pi2nfp / Nz
      do jj = 0, Nt - 1
        teta = jj * pi2 / Nt
        jk = 1 + jj + kk * Nt
        do ii = 1, mn
          arg = im(ii) * teta - in(ii) * zeta
          prjcos(jk, ii) = cos(arg) * 2.0_rkind / Ntz
          prjsin(jk, ii) = sin(arg) * 2.0_rkind / Ntz
        end do
      end do
    end do
    prjcos(1:Ntz, 1) = 0.5_rkind * prjcos(1:Ntz, 1)

    call nallocate(srcR, 1, Ctz, 0, 3)
    call nallocate(srcZ, 1, Ctz, 0, 3)
    call nallocate(srcpos, 1, Ctz, 1, 3)
    call nallocate(srcxt, 1, Ctz, 1, 3)
    call nallocate(srcxz, 1, Ctz, 1, 3)
    call nallocate(srcgtt, 1, Ctz)
    call nallocate(srcgtz, 1, Ctz)
    call nallocate(srcgzz, 1, Ctz)
    call nallocate(srcinvsg, 1, Ctz)
    call nallocate(srcgBut, 1, Ctz)
    call nallocate(srcgBuz, 1, Ctz)
    call nallocate(srcBlt, 1, Ctz)
    call nallocate(srcBlz, 1, Ctz)

  end subroutine vc_tables_init

  subroutine vc_tables_destroy()
    if (allocated(kjcos)) then
      deallocate(kjcos, kjsin, kjczeta, kjszeta, prjcos, prjsin)
      deallocate(srcR, srcZ, srcpos, srcxt, srcxz, srcgtt, srcgtz, srcgzz, srcinvsg)
      deallocate(srcgBut, srcgBuz, srcBlt, srcBlz)
    end if
    Ctz = 0
  end subroutine vc_tables_destroy

  !> Project a real-space field on the computational-boundary grid onto (cos, sin) harmonics,
  !> with exactly the tfft coefficient normalization.
  subroutine vc_project(field, emn, omn)
    real(rkind), intent(in) :: field(1:Ntz)
    real(rkind), intent(out) :: emn(1:mn), omn(1:mn)
    integer :: ii

    do ii = 1, mn
      emn(ii) = sum(prjcos(1:Ntz, ii) * field(1:Ntz))
      omn(ii) = sum(prjsin(1:Ntz, ii) * field(1:Ntz))
    end do
  end subroutine vc_project

  !> Evaluate the plasma-boundary geometry (value and s, theta, zeta derivatives at s = -1 of the
  !> vacuum volume) on the source grid; same algebra as freeb_mod::surfacecurrent.
  subroutine vc_refresh_geometry()
    integer :: kj, ii
    real(rkind) :: R(0:3), Z(0:3), cz, sz, sg

    !$OMP PARALLEL DO PRIVATE(kj, ii, R, Z, cz, sz, sg)
    do kj = 1, Ctz
      R(0:3) = 0.0_rkind
      Z(0:3) = 0.0_rkind

      do ii = 1, mn
        R(0) = R(0) + iRbc(ii, nvol) * kjcos(ii, kj)
        R(1) = R(1) + (iRbc(ii, Mvol) - iRbc(ii, nvol)) * kjcos(ii, kj) * 0.5_rkind
        R(2) = R(2) - im(ii) * iRbc(ii, nvol) * kjsin(ii, kj)
        R(3) = R(3) + in(ii) * iRbc(ii, nvol) * kjsin(ii, kj)

        Z(0) = Z(0) + iZbs(ii, nvol) * kjsin(ii, kj)
        Z(1) = Z(1) + (iZbs(ii, Mvol) - iZbs(ii, nvol)) * kjsin(ii, kj) * 0.5_rkind
        Z(2) = Z(2) + im(ii) * iZbs(ii, nvol) * kjcos(ii, kj)
        Z(3) = Z(3) - in(ii) * iZbs(ii, nvol) * kjcos(ii, kj)

        if (.not.enforce_stell_sym) then
          R(0) = R(0) + iRbs(ii, nvol) * kjsin(ii, kj)
          R(1) = R(1) + (iRbs(ii, Mvol) - iRbs(ii, nvol)) * kjsin(ii, kj) * 0.5_rkind
          R(2) = R(2) + im(ii) * iRbs(ii, nvol) * kjcos(ii, kj)
          R(3) = R(3) - in(ii) * iRbs(ii, nvol) * kjcos(ii, kj)

          Z(0) = Z(0) + iZbc(ii, nvol) * kjcos(ii, kj)
          Z(1) = Z(1) + (iZbc(ii, Mvol) - iZbc(ii, nvol)) * kjcos(ii, kj) * 0.5_rkind
          Z(2) = Z(2) - im(ii) * iZbc(ii, nvol) * kjsin(ii, kj)
          Z(3) = Z(3) + in(ii) * iZbc(ii, nvol) * kjsin(ii, kj)
        end if
      end do

      srcR(kj, 0:3) = R(0:3)
      srcZ(kj, 0:3) = Z(0:3)

      cz = kjczeta(kj)
      sz = kjszeta(kj)
      srcpos(kj, 1:3) = [R(0) * cz, R(0) * sz, Z(0)]
      srcxt(kj, 1:3) = [R(2) * cz, R(2) * sz, Z(2)]
      srcxz(kj, 1:3) = [R(3) * cz - R(0) * sz, R(3) * sz + R(0) * cz, Z(3)]

      srcgtt(kj) = R(2) * R(2) + Z(2) * Z(2)
      srcgtz(kj) = R(2) * R(3) + Z(2) * Z(3)
      srcgzz(kj) = R(3) * R(3) + Z(3) * Z(3) + R(0) * R(0)

      sg = R(0) * (Z(1) * R(2) - R(1) * Z(2))
      srcinvsg(kj) = 1.0_rkind / sg
    end do
    !$OMP END PARALLEL DO

  end subroutine vc_refresh_geometry

  !> Assemble the dense Laplace-virtual-casing matrix dVC for the vacuum volume: row Lme(ii)
  !> (Lmf(ii) without stellarator symmetry) times the packed vector-potential coefficients gives
  !> the sin (cos) harmonic on the computational boundary of the virtual-casing field
  !> virtualcasingfactor * (j x r).n produced by the vacuum tangential field at s = -1.
  subroutine vc_assemble_dvc(lvol, lmn, lrad)
    integer, intent(in) :: lvol, lmn, lrad

    integer :: ii, jj, kj, jk, ll, jd
    real(rkind) :: rr(1:3), d2, fof, xtn, xzn, wt, wz, norm, w
    real(rkind) :: pe(1:mn), po(1:mn)
    real(rkind), allocatable :: fAte(:, :), fAze(:, :), fAto(:, :), fAzo(:, :)

    call fatal(Ctz == 0, __FILE__, "vc_tables_init must be called before vc_assemble_dvc")
    call fatal(first_free_bound, __FILE__, "the Laplace-virtual-casing matrix is undefined during the LautoinitBn bootstrap")
    call fatal(lvol /= Mvol, __FILE__, "the Laplace-virtual-casing matrix only exists in the vacuum volume")

    call vc_refresh_geometry()

    call sallocate(fAte, 1, lmn, 1, Ntz, 0.0_rkind)
    call sallocate(fAze, 1, lmn, 1, Ntz, 0.0_rkind)
    if (.not.enforce_stell_sym) then
      call sallocate(fAto, 1, lmn, 1, Ntz, 0.0_rkind)
      call sallocate(fAzo, 1, lmn, 1, Ntz, 0.0_rkind)
    end if

    norm = pi2 * pi2 * virtualcasingfactor / real(Ctz, rkind)

    ! field on the computational boundary per unit vector-potential basis coefficient: the kernel
    ! is (j x r).N / |r|^3 with j = Blz x_theta - Blt x_zeta, exactly as surfacecurrent/casinggrid
    !$OMP PARALLEL DO PRIVATE(jk, kj, ii, rr, d2, fof, xtn, xzn, wt, wz)
    do jk = 1, Ntz
      do kj = 1, Ctz
        rr(1:3) = Dxyz(1:3, jk) - srcpos(kj, 1:3)
        d2 = sum(rr * rr) + vcasingeps ** 2
        fof = (1.0_rkind + 3.0_rkind * vcasingeps ** 2 / d2) / (sqrt(d2) * d2)

        xtn = ( (srcxt(kj, 2) * rr(3) - srcxt(kj, 3) * rr(2)) * Nxyz(1, jk) &
            + (srcxt(kj, 3) * rr(1) - srcxt(kj, 1) * rr(3)) * Nxyz(2, jk) &
            + (srcxt(kj, 1) * rr(2) - srcxt(kj, 2) * rr(1)) * Nxyz(3, jk) ) * fof
        xzn = ( (srcxz(kj, 2) * rr(3) - srcxz(kj, 3) * rr(2)) * Nxyz(1, jk) &
            + (srcxz(kj, 3) * rr(1) - srcxz(kj, 1) * rr(3)) * Nxyz(2, jk) &
            + (srcxz(kj, 1) * rr(2) - srcxz(kj, 2) * rr(1)) * Nxyz(3, jk) ) * fof

        ! wt multiplies sqrt(g) B^theta (gBut), wz multiplies sqrt(g) B^zeta (gBuz)
        wt = (srcgtt(kj) * xzn * (-1.0_rkind) + srcgtz(kj) * xtn) * srcinvsg(kj)
        wz = (srcgtz(kj) * xzn * (-1.0_rkind) + srcgzz(kj) * xtn) * srcinvsg(kj)

        ! gBuz = +cos (Ate), +sin (Ato); gBut = -cos (Aze), -sin (Azo); radial weight TT applied at fill
        do ii = 1, lmn
          fAte(ii, jk) = fAte(ii, jk) + kjcos(ii, kj) * wz
          fAze(ii, jk) = fAze(ii, jk) - kjcos(ii, kj) * wt
        end do
        if (.not.enforce_stell_sym) then
          do ii = 1, lmn
            fAto(ii, jk) = fAto(ii, jk) + kjsin(ii, kj) * wz
            fAzo(ii, jk) = fAzo(ii, jk) - kjsin(ii, kj) * wt
          end do
        end if
      end do
    end do
    !$OMP END PARALLEL DO

    dVC = 0.0_rkind

    do jj = 1, lmn

      call vc_project(fAte(jj, 1:Ntz) * norm, pe, po)
      do ll = 0, lrad
        jd = Ate(lvol, 0, jj)%i(ll)
        if (jd == 0) cycle
        w = TT(ll, 0, 1)
        do ii = 2, lmn
          dVC(Lme(lvol, ii), jd) = po(ii) * w
          if (.not.enforce_stell_sym) dVC(Lmf(lvol, ii), jd) = pe(ii) * w
        end do
      end do

      call vc_project(fAze(jj, 1:Ntz) * norm, pe, po)
      do ll = 0, lrad
        jd = Aze(lvol, 0, jj)%i(ll)
        if (jd == 0) cycle
        w = TT(ll, 0, 1)
        do ii = 2, lmn
          dVC(Lme(lvol, ii), jd) = po(ii) * w
          if (.not.enforce_stell_sym) dVC(Lmf(lvol, ii), jd) = pe(ii) * w
        end do
      end do

      if (.not.enforce_stell_sym) then
        call vc_project(fAto(jj, 1:Ntz) * norm, pe, po)
        do ll = 0, lrad
          jd = Ato(lvol, 0, jj)%i(ll)
          if (jd == 0) cycle
          w = TT(ll, 0, 1)
          do ii = 2, lmn
            dVC(Lme(lvol, ii), jd) = po(ii) * w
            dVC(Lmf(lvol, ii), jd) = pe(ii) * w
          end do
        end do

        call vc_project(fAzo(jj, 1:Ntz) * norm, pe, po)
        do ll = 0, lrad
          jd = Azo(lvol, 0, jj)%i(ll)
          if (jd == 0) cycle
          w = TT(ll, 0, 1)
          do ii = 2, lmn
            dVC(Lme(lvol, ii), jd) = po(ii) * w
            dVC(Lmf(lvol, ii), jd) = pe(ii) * w
          end do
        end do
      end if

    end do

    deallocate(fAte, fAze)
    if (.not.enforce_stell_sym) deallocate(fAto, fAzo)

  end subroutine vc_assemble_dvc

  !> Refresh the solution-dependent source fields (tangential field at s = -1 from the ideriv = 0
  !> vector potential of the vacuum volume) and the zeroth-order geometry. Must be called serially,
  !> with converged geometry and solution, before vc_apply_a0 or any vc_deriv_rhs_add.
  subroutine vc_prepare_deriv()
    integer :: kj, ii, ll
    real(rkind) :: ate_b(1:mn), aze_b(1:mn), ato_b(1:mn), azo_b(1:mn)
    real(rkind) :: gBut, gBuz

    call fatal(Ctz == 0, __FILE__, "vc_tables_init must be called before vc_prepare_deriv")

    call vc_refresh_geometry()

    ate_b(1:mn) = 0.0_rkind
    aze_b(1:mn) = 0.0_rkind
    ato_b(1:mn) = 0.0_rkind
    azo_b(1:mn) = 0.0_rkind
    do ii = 1, mn
      do ll = 0, Lrad(Mvol)
        ate_b(ii) = ate_b(ii) + Ate(Mvol, 0, ii)%s(ll) * TT(ll, 0, 1)
        aze_b(ii) = aze_b(ii) + Aze(Mvol, 0, ii)%s(ll) * TT(ll, 0, 1)
      end do
      if (.not.enforce_stell_sym) then
        do ll = 0, Lrad(Mvol)
          ato_b(ii) = ato_b(ii) + Ato(Mvol, 0, ii)%s(ll) * TT(ll, 0, 1)
          azo_b(ii) = azo_b(ii) + Azo(Mvol, 0, ii)%s(ll) * TT(ll, 0, 1)
        end do
      end if
    end do

    !$OMP PARALLEL DO PRIVATE(kj, ii, gBut, gBuz)
    do kj = 1, Ctz
      gBut = 0.0_rkind
      gBuz = 0.0_rkind
      do ii = 1, mn
        gBut = gBut - aze_b(ii) * kjcos(ii, kj)
        gBuz = gBuz + ate_b(ii) * kjcos(ii, kj)
      end do
      if (.not.enforce_stell_sym) then
        do ii = 1, mn
          gBut = gBut - azo_b(ii) * kjsin(ii, kj)
          gBuz = gBuz + ato_b(ii) * kjsin(ii, kj)
        end do
      end if
      srcgBut(kj) = gBut
      srcgBuz(kj) = gBuz
      srcBlt(kj) = (gBut * srcgtt(kj) + gBuz * srcgtz(kj)) * srcinvsg(kj)
      srcBlz(kj) = (gBut * srcgtz(kj) + gBuz * srcgzz(kj)) * srcinvsg(kj)
    end do
    !$OMP END PARALLEL DO

  end subroutine vc_prepare_deriv

  !> Matrix-free application of the virtual-casing operator to the current (ideriv = 0) vacuum
  !> solution: returns the (cos, sin) harmonics on the computational boundary of the plasma-induced
  !> sqrt(g) B^s. Must agree with matmul(dVC, packed solution) on the Lme/Lmf rows to machine
  !> precision, and with freeb_mod::compute_bnorm_plasmab up to quadrature error.
  subroutine vc_apply_a0(lvol, gbne, gbno)
    integer, intent(in) :: lvol
    real(rkind), intent(out) :: gbne(1:mn), gbno(1:mn)

    integer :: jk, kj
    real(rkind) :: rr(1:3), d2, fof, xtn, xzn, acc, norm
    real(rkind) :: gbn(1:Ntz)

    call fatal(lvol /= Mvol, __FILE__, "the virtual-casing operator only exists in the vacuum volume")

    call vc_prepare_deriv()

    norm = pi2 * pi2 * virtualcasingfactor / real(Ctz, rkind)

    !$OMP PARALLEL DO PRIVATE(jk, kj, rr, d2, fof, xtn, xzn, acc)
    do jk = 1, Ntz
      acc = 0.0_rkind
      do kj = 1, Ctz
        rr(1:3) = Dxyz(1:3, jk) - srcpos(kj, 1:3)
        d2 = sum(rr * rr) + vcasingeps ** 2
        fof = (1.0_rkind + 3.0_rkind * vcasingeps ** 2 / d2) / (sqrt(d2) * d2)

        xtn = ( (srcxt(kj, 2) * rr(3) - srcxt(kj, 3) * rr(2)) * Nxyz(1, jk) &
            + (srcxt(kj, 3) * rr(1) - srcxt(kj, 1) * rr(3)) * Nxyz(2, jk) &
            + (srcxt(kj, 1) * rr(2) - srcxt(kj, 2) * rr(1)) * Nxyz(3, jk) ) * fof
        xzn = ( (srcxz(kj, 2) * rr(3) - srcxz(kj, 3) * rr(2)) * Nxyz(1, jk) &
            + (srcxz(kj, 3) * rr(1) - srcxz(kj, 1) * rr(3)) * Nxyz(2, jk) &
            + (srcxz(kj, 1) * rr(2) - srcxz(kj, 2) * rr(1)) * Nxyz(3, jk) ) * fof

        acc = acc + srcBlz(kj) * xtn - srcBlt(kj) * xzn
      end do
      gbn(jk) = acc * norm
    end do
    !$OMP END PARALLEL DO

    call vc_project(gbn, gbne, gbno)

  end subroutine vc_apply_a0

  !> Add the eq. 4.5 term (d dVC / d x) . a0 to the perturbed right-hand side of the vacuum
  !> Beltrami system, for the single plasma-boundary harmonic described by dBdX (%ii, %irz, %issym).
  !> The complete perturbed rhs is -(d dMA / d x - d dVC / d x) . a0; the caller (get_perturbed_rhs)
  !> has already accumulated -(d dMA / d x) . a0 matrix-free. Requires a prior vc_prepare_deriv with
  !> the current geometry and solution. Thread-safe: reads only module tables and the THREADPRIVATE
  !> dBdX; writes only rhs and locals.
  subroutine vc_deriv_rhs_add(lvol, NN, rhs)
    integer, intent(in) :: lvol, NN
    real(rkind), intent(inout) :: rhs(0:NN)

    integer :: ii, kj, jk, mp, np
    real(rkind) :: bc, bs, cz, sz
    real(rkind) :: dR0, dRs, dRt, dRz, dZ0, dZs, dZt, dZz
    real(rkind) :: dpos(1:3), dxt(1:3), dxz(1:3)
    real(rkind) :: dgtt, dgtz, dgzz, dsg, dinvsg, dBlt, dBlz
    real(rkind) :: rr(1:3), d2, dd2, t1, invd2, fof, dfof, xtn, xzn, dxtn, dxzn
    real(rkind) :: norm
    real(rkind) :: dgbn(1:Ntz), de(1:mn), do_(1:mn)

    call fatal(lvol /= Mvol, __FILE__, "the virtual-casing operator only exists in the vacuum volume")

    mp = im(dBdX%ii)
    np = in(dBdX%ii)

    norm = pi2 * pi2 * virtualcasingfactor / real(Ctz, rkind)
    dgbn(1:Ntz) = 0.0_rkind

    do kj = 1, Ctz
      bc = kjcos(dBdX%ii, kj)
      bs = kjsin(dBdX%ii, kj)

      ! derivatives of the source geometry w.r.t. the perturbed harmonic of interface nvol (the
      ! plasma boundary): value, s, theta, zeta slots, matching surfacecurrent's interpolation
      dR0 = 0.0_rkind
      dRs = 0.0_rkind
      dRt = 0.0_rkind
      dRz = 0.0_rkind
      dZ0 = 0.0_rkind
      dZs = 0.0_rkind
      dZt = 0.0_rkind
      dZz = 0.0_rkind
      if (dBdX%irz == 0) then
        if (dBdX%issym == 0) then ! Rbc
          dR0 = bc
          dRs = -bc * 0.5_rkind
          dRt = -mp * bs
          dRz = +np * bs
        else ! Rbs
          dR0 = bs
          dRs = -bs * 0.5_rkind
          dRt = +mp * bc
          dRz = -np * bc
        end if
      else
        if (dBdX%issym == 0) then ! Zbs
          dZ0 = bs
          dZs = -bs * 0.5_rkind
          dZt = +mp * bc
          dZz = -np * bc
        else ! Zbc
          dZ0 = bc
          dZs = -bc * 0.5_rkind
          dZt = -mp * bs
          dZz = +np * bs
        end if
      end if

      cz = kjczeta(kj)
      sz = kjszeta(kj)
      dpos(1:3) = [dR0 * cz, dR0 * sz, dZ0]
      dxt(1:3) = [dRt * cz, dRt * sz, dZt]
      dxz(1:3) = [dRz * cz - dR0 * sz, dRz * sz + dR0 * cz, dZz]

      dgtt = 2.0_rkind * (srcR(kj, 2) * dRt + srcZ(kj, 2) * dZt)
      dgtz = dRt * srcR(kj, 3) + srcR(kj, 2) * dRz + dZt * srcZ(kj, 3) + srcZ(kj, 2) * dZz
      dgzz = 2.0_rkind * (srcR(kj, 3) * dRz + srcZ(kj, 3) * dZz + srcR(kj, 0) * dR0)

      dsg = dR0 * (srcZ(kj, 1) * srcR(kj, 2) - srcR(kj, 1) * srcZ(kj, 2)) &
          + srcR(kj, 0) * (dZs * srcR(kj, 2) + srcZ(kj, 1) * dRt - dRs * srcZ(kj, 2) - srcR(kj, 1) * dZt)
      dinvsg = -dsg * srcinvsg(kj) ** 2

      ! the vector-potential coefficients are held fixed: only the metric factors of Blt/Blz vary
      dBlt = (srcgBut(kj) * dgtt + srcgBuz(kj) * dgtz) * srcinvsg(kj) &
          + (srcgBut(kj) * srcgtt(kj) + srcgBuz(kj) * srcgtz(kj)) * dinvsg
      dBlz = (srcgBut(kj) * dgtz + srcgBuz(kj) * dgzz) * srcinvsg(kj) &
          + (srcgBut(kj) * srcgtz(kj) + srcgBuz(kj) * srcgzz(kj)) * dinvsg

      do jk = 1, Ntz
        ! the computational boundary is fixed, so d(rr) = -dpos
        rr(1:3) = Dxyz(1:3, jk) - srcpos(kj, 1:3)
        d2 = sum(rr * rr) + vcasingeps ** 2
        dd2 = -2.0_rkind * sum(rr * dpos)

        invd2 = 1.0_rkind / d2
        t1 = invd2 / sqrt(d2)! = d2^(-3/2)
        fof = t1 + 3.0_rkind * vcasingeps ** 2 * t1 * invd2
        dfof = (-1.5_rkind * t1 * invd2 - 7.5_rkind * vcasingeps ** 2 * t1 * invd2 ** 2) * dd2

        xtn = (srcxt(kj, 2) * rr(3) - srcxt(kj, 3) * rr(2)) * Nxyz(1, jk) &
            + (srcxt(kj, 3) * rr(1) - srcxt(kj, 1) * rr(3)) * Nxyz(2, jk) &
            + (srcxt(kj, 1) * rr(2) - srcxt(kj, 2) * rr(1)) * Nxyz(3, jk)
        xzn = (srcxz(kj, 2) * rr(3) - srcxz(kj, 3) * rr(2)) * Nxyz(1, jk) &
            + (srcxz(kj, 3) * rr(1) - srcxz(kj, 1) * rr(3)) * Nxyz(2, jk) &
            + (srcxz(kj, 1) * rr(2) - srcxz(kj, 2) * rr(1)) * Nxyz(3, jk)

        dxtn = (dxt(2) * rr(3) - dxt(3) * rr(2) + srcxt(kj, 3) * dpos(2) - srcxt(kj, 2) * dpos(3)) * Nxyz(1, jk) &
            + (dxt(3) * rr(1) - dxt(1) * rr(3) + srcxt(kj, 1) * dpos(3) - srcxt(kj, 3) * dpos(1)) * Nxyz(2, jk) &
            + (dxt(1) * rr(2) - dxt(2) * rr(1) + srcxt(kj, 2) * dpos(1) - srcxt(kj, 1) * dpos(2)) * Nxyz(3, jk)
        dxzn = (dxz(2) * rr(3) - dxz(3) * rr(2) + srcxz(kj, 3) * dpos(2) - srcxz(kj, 2) * dpos(3)) * Nxyz(1, jk) &
            + (dxz(3) * rr(1) - dxz(1) * rr(3) + srcxz(kj, 1) * dpos(3) - srcxz(kj, 3) * dpos(1)) * Nxyz(2, jk) &
            + (dxz(1) * rr(2) - dxz(2) * rr(1) + srcxz(kj, 2) * dpos(1) - srcxz(kj, 1) * dpos(2)) * Nxyz(3, jk)

        dgbn(jk) = dgbn(jk) &
            + (dBlz * xtn + srcBlz(kj) * dxtn - dBlt * xzn - srcBlt(kj) * dxzn) * fof &
            + (srcBlz(kj) * xtn - srcBlt(kj) * xzn) * dfof
      end do
    end do

    dgbn(1:Ntz) = dgbn(1:Ntz) * norm

    call vc_project(dgbn, de, do_)

    do ii = 2, mn
      rhs(Lme(lvol, ii)) = rhs(Lme(lvol, ii)) + do_(ii)
      if (.not.enforce_stell_sym) rhs(Lmf(lvol, ii)) = rhs(Lmf(lvol, ii)) + de(ii)
    end do

  end subroutine vc_deriv_rhs_add

end module vc_matrix_mod
