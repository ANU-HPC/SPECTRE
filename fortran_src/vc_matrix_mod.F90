! SPDX-FileCopyrightText: 2026 Proxima Fusion
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
!> regularization, same pi2^2 * virtualcasingfactor / Ctz normalization). Projections onto the
!> boundary harmonics go through the prjcos/prjsin tables (vc_project, or DGEMM in the batched
!> derivative) with exactly the tfft coefficient normalization; tfft itself is avoided because its
!> FFTW buffers are shared global state.
!>
!> Performance structure: (a) the assembled matrix is cached keyed on the boundary geometry, so
!> force evaluation and Jacobian at the same geometry assemble once; (b) with stellarator symmetry
!> the projected fields are odd under (theta, zeta) -> (-theta, -zeta), so only the first Ncomp
!> target points are evaluated and the rest are mirror-filled; (c) the per-mode accumulation of
!> the assembly is a (mn x Ctz) times (Ctz x Ncomp) product, done blockwise with DGEMM. The
!> target grid spans one field period, which already exploits the nfp symmetry of the kernel.
module vc_matrix_mod

  use precisions_mod, only: rkind
  use constants_mod, only: pi2
  use errors_mod, only: fatal
  use memory_allocation_mod, only: nallocate, sallocate
  use input_list_mod, only: nvol, Lrad, nfp, cteta, czeta, enforce_stell_sym, vcasingeps, Igeometry
  use allglobal_mod, only: mn, im, in, Mvol, Nt, Nz, Ntz, pi2nfp, NAdof, &
      iRbc, iZbs, iRbs, iZbc, TT, Ate, Aze, Ato, Azo, Lme, Lmf, &
      Dxyz, Nxyz, virtualcasingfactor, dBdX, first_free_bound

  implicit none(type, external)

  private
  public :: vc_tables_init, vc_tables_destroy, vc_assemble_dvc, vc_subtract_dvc, vc_dense_rows_dot, &
      vc_prepare_deriv, vc_apply_a0, vc_deriv_rhs_add, vc_deriv_rhs_add_dir, vc_matvec_sub, vc_project

  external :: DGEMM

  integer, parameter :: KBLK = 2048 !< source-point block size of the DGEMM assembly
  integer, parameter :: DBLK = 512 !< source-point block size of the batched derivative

  integer :: Cnt = 0 !< theta resolution of the source (plasma-boundary) quadrature grid
  integer :: Cnz = 0 !< zeta resolution of the source grid over the full torus (= czeta * nfp)
  integer, public :: Ctz = 0 !< total number of source quadrature points (= Cnt * Cnz)
  real(rkind) :: vcnorm = 0.0_rkind !< quadrature normalization pi2^2 * virtualcasingfactor / Ctz

  !> factorized source-grid trig tables: the source grid is a (theta, zeta) tensor grid, so
  !> cos/sin(m theta - n zeta) is combined on the fly from 1-D tables (source point
  !> kj = 1 + jt + kz*Cnt has theta index jt and zeta index kz), keeping the memory
  !> O(mn*(Cnt+Cnz)) instead of O(mn*Ctz)
  real(rkind), allocatable :: tcos(:, :), tsin(:, :)!< (1:mn, 0:Cnt-1) cos/sin(m theta_jt)
  real(rkind), allocatable :: zcos(:, :), zsin(:, :)!< (1:mn, 0:Cnz-1) cos/sin(n zeta_kz)
  real(rkind), allocatable :: czeta1(:), szeta1(:)!< (0:Cnz-1) cos/sin(zeta_kz)

  !> computational-boundary projection tables replicating the tfft normalization:
  !> weight 2/Ntz for (m,n) /= (0,0) and 1/Ntz for (0,0)
  real(rkind), allocatable :: prjcos(:, :)!< (1:Ntz, 1:mn)
  real(rkind), allocatable :: prjsin(:, :)!< (1:Ntz, 1:mn)

  !> stellarator-symmetry folding of the target grid: with enforce_stell_sym every projected field
  !> is odd under (theta, zeta) -> (-theta, -zeta), so only points 1:Ncomp (a contiguous run of
  !> zeta planes) are evaluated and point jk > Ncomp is -field(msrc(jk))
  integer :: Ncomp = 0 !< number of directly evaluated target points (= Ntz without symmetry)
  integer, allocatable :: msrc(:)!< (1:Ntz) mirror source index for jk > Ncomp

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

  !> the assembled operator in compact row form: dVC only populates the Lme (and, without
  !> stellarator symmetry, Lmf) constraint rows, so only those rows are stored — row r = ii-1
  !> is the sine row of mode ii, row r = (mn-1)+(ii-1) the cosine row — and subtracted from the
  !> Lme/Lmf rows of dMA in place (vc_subtract_dvc); the dense (0:NN)^2 form never exists.
  !> The rows persist between calls and double as the geometry-keyed cache: dVC depends only on
  !> the geometry of the two interfaces bounding the vacuum annulus.
  integer :: nvcrow = 0
  integer, allocatable :: dVCidx(:)!< (1:nvcrow) constraint-row slots (Lme, then Lmf)
  real(rkind), allocatable :: dVCrow(:, :)!< (1:nvcrow, 0:NN) the stored rows
  logical :: cache_valid = .false.
  real(rkind), allocatable :: cache_geom(:)!< (1:8*mn) boundary harmonics at assembly time

  !> batched derivative: the perturbed integrand is linear in 8 per-source-point geometry weights
  !> (dR0, dRs, dRt, dRz, dZ0, dZs, dZt, dZz), so the projected harmonics of (d dVC/dx).a0 for
  !> every plasma-boundary degree of freedom are precomputed together by contracting 8
  !> DOF-independent kernel fields with per-DOF weights (blocked DGEMM). Built by
  !> vc_prepare_deriv; consumed by lookup in vc_deriv_rhs_add.
  logical :: batch_valid = .false.
  integer :: batch_n = 0
  integer, allocatable :: batch_ii(:), batch_irz(:), batch_issym(:)
  real(rkind), allocatable :: batch_e(:, :)!< (1:mn, 1:batch_n) cos harmonics of (d dVC/dx).a0
  real(rkind), allocatable :: batch_o(:, :)!< (1:mn, 1:batch_n) sin harmonics of (d dVC/dx).a0

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
    vcnorm = pi2 * pi2 * virtualcasingfactor / real(Ctz, rkind)

    call nallocate(tcos, 1, mn, 0, Cnt - 1)
    call nallocate(tsin, 1, mn, 0, Cnt - 1)
    call nallocate(zcos, 1, mn, 0, Cnz - 1)
    call nallocate(zsin, 1, mn, 0, Cnz - 1)
    call nallocate(czeta1, 0, Cnz - 1)
    call nallocate(szeta1, 0, Cnz - 1)

    do jj = 0, Cnt - 1
      teta = jj * pi2 / Cnt
      do ii = 1, mn
        tcos(ii, jj) = cos(im(ii) * teta)
        tsin(ii, jj) = sin(im(ii) * teta)
      end do
    end do
    do kk = 0, Cnz - 1
      zeta = kk * pi2 / Cnz
      czeta1(kk) = cos(zeta)
      szeta1(kk) = sin(zeta)
      do ii = 1, mn
        zcos(ii, kk) = cos(in(ii) * zeta)
        zsin(ii, kk) = sin(in(ii) * zeta)
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

    ! target-grid folding for stellarator symmetry: the mirror of grid point (jj, kk) is
    ! ((Nt-jj) mod Nt, (Nz-kk) mod Nz), where an odd field takes the negated value. The
    ! evaluated set (the first Ncomp linear indices) always contains the mirror of every
    ! remaining point, including all self-mirrored points (where odd fields vanish).
    call nallocate(msrc, 1, Ntz)
    msrc(1:Ntz) = 0
    if (enforce_stell_sym) then
      if (Nz == 1) then
        Ncomp = (Nt + 1) / 2 + 1
      else
        Ncomp = Nt * min(Nz, (Nz + 3) / 2)
      end if
      do jk = Ncomp + 1, Ntz
        jj = mod(jk - 1, Nt)
        kk = (jk - 1) / Nt
        msrc(jk) = 1 + mod(Nt - jj, Nt) + mod(Nz - kk, Nz) * Nt
        call fatal(msrc(jk) > Ncomp, __FILE__, "mirror source outside the evaluated target set")
      end do
    else
      Ncomp = Ntz
    end if

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

    ! compact row layout of the operator: sine rows first, cosine rows (non-stellsym) after
    if (enforce_stell_sym) then
      nvcrow = mn - 1
    else
      nvcrow = 2 * (mn - 1)
    end if
    call nallocate(dVCidx, 1, nvcrow)
    do ii = 2, mn
      dVCidx(ii - 1) = Lme(Mvol, ii)
      if (.not.enforce_stell_sym) dVCidx(mn - 1 + ii - 1) = Lmf(Mvol, ii)
    end do
    call nallocate(dVCrow, 1, nvcrow, 0, NAdof(Mvol))

    call nallocate(cache_geom, 1, 8 * mn)
    cache_valid = .false.

  end subroutine vc_tables_init

  subroutine vc_tables_destroy()
    if (allocated(tcos)) deallocate(tcos)
    if (allocated(tsin)) deallocate(tsin)
    if (allocated(zcos)) deallocate(zcos)
    if (allocated(zsin)) deallocate(zsin)
    if (allocated(czeta1)) deallocate(czeta1)
    if (allocated(szeta1)) deallocate(szeta1)
    if (allocated(prjcos)) deallocate(prjcos)
    if (allocated(prjsin)) deallocate(prjsin)
    if (allocated(msrc)) deallocate(msrc)
    if (allocated(srcR)) deallocate(srcR)
    if (allocated(srcZ)) deallocate(srcZ)
    if (allocated(srcpos)) deallocate(srcpos)
    if (allocated(srcxt)) deallocate(srcxt)
    if (allocated(srcxz)) deallocate(srcxz)
    if (allocated(srcgtt)) deallocate(srcgtt)
    if (allocated(srcgtz)) deallocate(srcgtz)
    if (allocated(srcgzz)) deallocate(srcgzz)
    if (allocated(srcinvsg)) deallocate(srcinvsg)
    if (allocated(srcgBut)) deallocate(srcgBut)
    if (allocated(srcgBuz)) deallocate(srcgBuz)
    if (allocated(srcBlt)) deallocate(srcBlt)
    if (allocated(srcBlz)) deallocate(srcBlz)
    if (allocated(cache_geom)) deallocate(cache_geom)
    if (allocated(dVCidx)) deallocate(dVCidx)
    if (allocated(dVCrow)) deallocate(dVCrow)
    if (allocated(batch_ii)) deallocate(batch_ii)
    if (allocated(batch_irz)) deallocate(batch_irz)
    if (allocated(batch_issym)) deallocate(batch_issym)
    if (allocated(batch_e)) deallocate(batch_e)
    if (allocated(batch_o)) deallocate(batch_o)
    Cnt = 0
    Cnz = 0
    Ctz = 0
    vcnorm = 0.0_rkind
    Ncomp = 0
    nvcrow = 0
    cache_valid = .false.
    batch_valid = .false.
    batch_n = 0
  end subroutine vc_tables_destroy

  !> Project a real-space field on the computational-boundary grid onto (cos, sin) harmonics,
  !> with exactly the tfft coefficient normalization.
  pure subroutine vc_project(field, emn, omn)
    real(rkind), intent(in) :: field(1:Ntz)
    real(rkind), intent(out) :: emn(1:mn), omn(1:mn)
    integer :: ii

    do ii = 1, mn
      emn(ii) = sum(prjcos(1:Ntz, ii) * field(1:Ntz))
      omn(ii) = sum(prjsin(1:Ntz, ii) * field(1:Ntz))
    end do
  end subroutine vc_project

  !> Pack the geometry the matrix depends on (both interfaces bounding the vacuum annulus).
  pure subroutine vc_pack_geometry(geom)
    real(rkind), intent(out) :: geom(1:8 * mn)

    geom(1:mn) = iRbc(1:mn, nvol)
    geom(mn + 1:2 * mn) = iRbc(1:mn, Mvol)
    geom(2 * mn + 1:3 * mn) = iZbs(1:mn, nvol)
    geom(3 * mn + 1:4 * mn) = iZbs(1:mn, Mvol)
    if (enforce_stell_sym) then
      geom(4 * mn + 1:8 * mn) = 0.0_rkind
    else
      geom(4 * mn + 1:5 * mn) = iRbs(1:mn, nvol)
      geom(5 * mn + 1:6 * mn) = iRbs(1:mn, Mvol)
      geom(6 * mn + 1:7 * mn) = iZbc(1:mn, nvol)
      geom(7 * mn + 1:8 * mn) = iZbc(1:mn, Mvol)
    end if
  end subroutine vc_pack_geometry

  !> Evaluate the plasma-boundary geometry (value and s, theta, zeta derivatives at s = -1 of the
  !> vacuum volume) on the source grid; same algebra as freeb_mod::surfacecurrent.
  subroutine vc_refresh_geometry()
    integer :: kj, ii, jt, kz
    real(rkind) :: R(0:3), Z(0:3), cz, sz, sg, carg, sarg

    !$OMP PARALLEL DO PRIVATE(kj, ii, jt, kz, R, Z, cz, sz, sg, carg, sarg)
    do kj = 1, Ctz
      jt = mod(kj - 1, Cnt)
      kz = (kj - 1) / Cnt

      R(0:3) = 0.0_rkind
      Z(0:3) = 0.0_rkind

      do ii = 1, mn
        carg = tcos(ii, jt) * zcos(ii, kz) + tsin(ii, jt) * zsin(ii, kz)
        sarg = tsin(ii, jt) * zcos(ii, kz) - tcos(ii, jt) * zsin(ii, kz)

        R(0) = R(0) + iRbc(ii, nvol) * carg
        R(1) = R(1) + (iRbc(ii, Mvol) - iRbc(ii, nvol)) * carg * 0.5_rkind
        R(2) = R(2) - im(ii) * iRbc(ii, nvol) * sarg
        R(3) = R(3) + in(ii) * iRbc(ii, nvol) * sarg

        Z(0) = Z(0) + iZbs(ii, nvol) * sarg
        Z(1) = Z(1) + (iZbs(ii, Mvol) - iZbs(ii, nvol)) * sarg * 0.5_rkind
        Z(2) = Z(2) + im(ii) * iZbs(ii, nvol) * carg
        Z(3) = Z(3) - in(ii) * iZbs(ii, nvol) * carg

        if (.not.enforce_stell_sym) then
          R(0) = R(0) + iRbs(ii, nvol) * sarg
          R(1) = R(1) + (iRbs(ii, Mvol) - iRbs(ii, nvol)) * sarg * 0.5_rkind
          R(2) = R(2) + im(ii) * iRbs(ii, nvol) * carg
          R(3) = R(3) - in(ii) * iRbs(ii, nvol) * carg

          Z(0) = Z(0) + iZbc(ii, nvol) * carg
          Z(1) = Z(1) + (iZbc(ii, Mvol) - iZbc(ii, nvol)) * carg * 0.5_rkind
          Z(2) = Z(2) - im(ii) * iZbc(ii, nvol) * sarg
          Z(3) = Z(3) + in(ii) * iZbc(ii, nvol) * sarg
        end if
      end do

      srcR(kj, 0:3) = R(0:3)
      srcZ(kj, 0:3) = Z(0:3)

      cz = czeta1(kz)
      sz = szeta1(kz)
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

  !> Kernel weights at one target point against a contiguous block of source points: wt multiplies
  !> sqrt(g) B^theta (gBut) and wz multiplies sqrt(g) B^zeta (gBuz) in the contribution to
  !> (j x r).N / |r|^3 at the target. Block form so the source loop vectorizes.
  pure subroutine vc_kernel_block(jk, k0, nblk, wt, wz)
    integer, intent(in) :: jk, k0, nblk
    real(rkind), intent(out) :: wt(1:nblk), wz(1:nblk)

    integer :: b, kj
    real(rkind) :: dx, dy, dz, n1, n2, n3
    real(rkind) :: rx, ry, rz, d2, fof, xtn, xzn

    dx = Dxyz(1, jk)
    dy = Dxyz(2, jk)
    dz = Dxyz(3, jk)
    n1 = Nxyz(1, jk)
    n2 = Nxyz(2, jk)
    n3 = Nxyz(3, jk)

    do b = 1, nblk
      kj = k0 + b - 1
      rx = dx - srcpos(kj, 1)
      ry = dy - srcpos(kj, 2)
      rz = dz - srcpos(kj, 3)
      d2 = rx * rx + ry * ry + rz * rz + vcasingeps ** 2
      fof = (1.0_rkind + 3.0_rkind * vcasingeps ** 2 / d2) / (sqrt(d2) * d2)

      xtn = ( (srcxt(kj, 2) * rz - srcxt(kj, 3) * ry) * n1 &
          + (srcxt(kj, 3) * rx - srcxt(kj, 1) * rz) * n2 &
          + (srcxt(kj, 1) * ry - srcxt(kj, 2) * rx) * n3 ) * fof
      xzn = ( (srcxz(kj, 2) * rz - srcxz(kj, 3) * ry) * n1 &
          + (srcxz(kj, 3) * rx - srcxz(kj, 1) * rz) * n2 &
          + (srcxz(kj, 1) * ry - srcxz(kj, 2) * rx) * n3 ) * fof

      wt(b) = (srcgtz(kj) * xtn - srcgtt(kj) * xzn) * srcinvsg(kj)
      wz(b) = (srcgzz(kj) * xtn - srcgtz(kj) * xzn) * srcinvsg(kj)
    end do
  end subroutine vc_kernel_block

  !> Assemble the dense Laplace-virtual-casing matrix dVC for the vacuum volume: row Lme(ii)
  !> (Lmf(ii) without stellarator symmetry) times the packed vector-potential coefficients gives
  !> the sin (cos) harmonic on the computational boundary of the virtual-casing field
  !> virtualcasingfactor * (j x r).n produced by the vacuum tangential field at s = -1.
  !> The result is cached keyed on the boundary geometry; a repeated call at unchanged geometry
  !> (e.g. the Jacobian following a force evaluation) only copies the cache.
  subroutine vc_assemble_dvc()
    integer :: ii, jj, jk, k0, bs_len, ic, b, kj, jt, kz
    real(rkind) :: geom(1:8 * mn)
    real(rkind), allocatable :: fAte(:, :), fAze(:, :), fAto(:, :), fAzo(:, :)
    real(rkind), allocatable :: wtb(:, :), wzb(:, :), cosb(:, :), sinb(:, :)

    call fatal(Ctz == 0, __FILE__, "vc_tables_init must be called before vc_assemble_dvc")
    call fatal(first_free_bound, __FILE__, "the Laplace-virtual-casing matrix is undefined during the LautoinitBn bootstrap")

    call vc_pack_geometry(geom)
    if (cache_valid .and. all(geom == cache_geom)) return ! the stored rows are current

    call vc_refresh_geometry()

    call sallocate(fAte, 1, mn, 1, Ntz, 0.0_rkind)
    call sallocate(fAze, 1, mn, 1, Ntz, 0.0_rkind)
    if (.not.enforce_stell_sym) then
      call sallocate(fAto, 1, mn, 1, Ntz, 0.0_rkind)
      call sallocate(fAzo, 1, mn, 1, Ntz, 0.0_rkind)
    end if
    call sallocate(wtb, 1, KBLK, 1, Ncomp, 0.0_rkind)
    call sallocate(wzb, 1, KBLK, 1, Ncomp, 0.0_rkind)
    call sallocate(cosb, 1, mn, 1, KBLK, 0.0_rkind)
    if (.not.enforce_stell_sym) call sallocate(sinb, 1, mn, 1, KBLK, 0.0_rkind)

    ! field on the computational boundary per unit vector-potential basis coefficient: the kernel
    ! is (j x r).N / |r|^3 with j = Blz x_theta - Blt x_zeta, exactly as surfacecurrent/casinggrid.
    ! gBuz = +cos (Ate), +sin (Ato); gBut = -cos (Aze), -sin (Azo); radial weight TT applied at
    ! the fill. The per-mode accumulation is a blocked matrix product over the source points.
    do k0 = 1, Ctz, KBLK
      bs_len = min(KBLK, Ctz - k0 + 1)

      !$OMP PARALLEL DO PRIVATE(ic)
      do ic = 1, Ncomp
        call vc_kernel_block(ic, k0, bs_len, wtb(1:bs_len, ic), wzb(1:bs_len, ic))
      end do
      !$OMP END PARALLEL DO

      ! materialize this block's trig columns from the factorized tables
      !$OMP PARALLEL DO PRIVATE(b, kj, jt, kz, ii)
      do b = 1, bs_len
        kj = k0 + b - 1
        jt = mod(kj - 1, Cnt)
        kz = (kj - 1) / Cnt
        do ii = 1, mn
          cosb(ii, b) = tcos(ii, jt) * zcos(ii, kz) + tsin(ii, jt) * zsin(ii, kz)
        end do
        if (.not.enforce_stell_sym) then
          do ii = 1, mn
            sinb(ii, b) = tsin(ii, jt) * zcos(ii, kz) - tcos(ii, jt) * zsin(ii, kz)
          end do
        end if
      end do
      !$OMP END PARALLEL DO

      call DGEMM('N', 'N', mn, Ncomp, bs_len, +1.0_rkind, cosb(1, 1), mn, wzb(1, 1), KBLK, 1.0_rkind, fAte(1, 1), mn)
      call DGEMM('N', 'N', mn, Ncomp, bs_len, -1.0_rkind, cosb(1, 1), mn, wtb(1, 1), KBLK, 1.0_rkind, fAze(1, 1), mn)
      if (.not.enforce_stell_sym) then
        call DGEMM('N', 'N', mn, Ncomp, bs_len, +1.0_rkind, sinb(1, 1), mn, wzb(1, 1), KBLK, 1.0_rkind, fAto(1, 1), mn)
        call DGEMM('N', 'N', mn, Ncomp, bs_len, -1.0_rkind, sinb(1, 1), mn, wtb(1, 1), KBLK, 1.0_rkind, fAzo(1, 1), mn)
      end if
    end do

    ! mirror-fill the remaining target points (odd fields under stellarator symmetry)
    do jk = Ncomp + 1, Ntz
      fAte(1:mn, jk) = -fAte(1:mn, msrc(jk))
      fAze(1:mn, jk) = -fAze(1:mn, msrc(jk))
    end do

    dVCrow = 0.0_rkind

    do jj = 1, mn
      call project_into_rows(fAte(jj, 1:Ntz), Ate(Mvol, 0, jj)%i)
      call project_into_rows(fAze(jj, 1:Ntz), Aze(Mvol, 0, jj)%i)
      if (.not.enforce_stell_sym) then
        call project_into_rows(fAto(jj, 1:Ntz), Ato(Mvol, 0, jj)%i)
        call project_into_rows(fAzo(jj, 1:Ntz), Azo(Mvol, 0, jj)%i)
      end if
    end do

    deallocate(fAte, fAze)
    if (.not.enforce_stell_sym) deallocate(fAto, fAzo)
    deallocate(wtb, wzb, cosb)
    if (.not.enforce_stell_sym) deallocate(sinb)

    cache_geom = geom
    cache_valid = .true.

  contains

    !> Project one basis coefficient's boundary field onto harmonics and scatter it, with the
    !> radial weight TT(ll,0,1) of the s = -1 evaluation, into that coefficient's column of the
    !> stored constraint rows.
    subroutine project_into_rows(field, idx)
      real(rkind), intent(in) :: field(1:Ntz)
      integer, intent(in) :: idx(0:)

      integer :: ll, jd, ii
      real(rkind) :: w, pe(1:mn), po(1:mn)

      call vc_project(field * vcnorm, pe, po)
      do ll = 0, ubound(idx, 1)
        jd = idx(ll)
        if (jd == 0) cycle
        w = TT(ll, 0, 1)
        do ii = 2, mn
          dVCrow(ii - 1, jd) = po(ii) * w
          if (.not.enforce_stell_sym) dVCrow(mn - 1 + ii - 1, jd) = pe(ii) * w
        end do
      end do
    end subroutine project_into_rows

  end subroutine vc_assemble_dvc

  !> Matrix-free counterpart of vc_subtract_dvc: subtract the Laplace-virtual-casing rows'
  !> action from an operator application in place, ax(dVCidx(r)) -= dVCrow(r, :) . x0, so a
  !> matvec of the vacuum operator reproduces the vc-corrected dense matrix without forming it.
  !> Ensures the rows are current first (vc_assemble_dvc, cached on the boundary geometry), so
  !> call it only from serial context, exactly like vc_subtract_dvc.
  subroutine vc_matvec_sub(NN, x0, ax)
    integer, intent(in) :: NN
    real(rkind), intent(in) :: x0(0:NN)
    real(rkind), intent(inout) :: ax(1:NN)

    integer :: r

    call vc_assemble_dvc()

    call fatal(NN /= ubound(dVCrow, 2), __FILE__, "vc_matvec_sub: vector size does not match the stored rows")

    do r = 1, nvcrow
      ax(dVCidx(r)) = ax(dVCidx(r)) - dot_product(dVCrow(r, 0:NN), x0(0:NN))
    end do

  end subroutine vc_matvec_sub

  !> Ensure the compact rows are current (vc_assemble_dvc, cached on geometry) and subtract them
  !> from the Lme/Lmf rows of the dense matrix A in place; the dense (0:NN)^2 form of the
  !> operator is never materialized.
  subroutine vc_subtract_dvc(A, NN)
    integer, intent(in) :: NN
    real(rkind), intent(inout) :: A(0:NN, 0:NN)

    integer :: r

    call vc_assemble_dvc()

    call fatal(NN /= ubound(dVCrow, 2), __FILE__, "vc_subtract_dvc: matrix size does not match the stored rows")

    do r = 1, nvcrow
      A(dVCidx(r), 0:NN) = A(dVCidx(r), 0:NN) - dVCrow(r, 0:NN)
    end do

  end subroutine vc_subtract_dvc

  !> (dVC . avec) on the constraint rows of the current assembly, returned as (cos, sin)
  !> harmonics of the plasma-induced sqrt(g) B^s on the computational boundary.
  subroutine vc_dense_rows_dot(NN, avec, gbne, gbno)
    integer, intent(in) :: NN
    real(rkind), intent(in) :: avec(0:NN)
    real(rkind), intent(out) :: gbne(1:mn), gbno(1:mn)

    integer :: ii

    call fatal(.not.cache_valid, __FILE__, "vc_dense_rows_dot requires a prior vc_assemble_dvc")
    call fatal(NN /= ubound(dVCrow, 2), __FILE__, "vc_dense_rows_dot: vector size does not match the stored rows")

    gbne(1:mn) = 0.0_rkind
    gbno(1:mn) = 0.0_rkind
    do ii = 2, mn
      gbno(ii) = dot_product(dVCrow(ii - 1, 0:NN), avec(0:NN))
      if (.not.enforce_stell_sym) gbne(ii) = dot_product(dVCrow(mn - 1 + ii - 1, 0:NN), avec(0:NN))
    end do

  end subroutine vc_dense_rows_dot

  !> Refresh the zeroth-order source geometry and the solution-dependent source fields (tangential
  !> field at s = -1 from the ideriv = 0 vector potential of the vacuum volume). Invalidates any
  !> previously built derivative batch.
  subroutine vc_refresh_source_fields()
    integer :: kj, ii, ll, jt, kz
    real(rkind) :: ate_b(1:mn), aze_b(1:mn), ato_b(1:mn), azo_b(1:mn)
    real(rkind) :: gBut, gBuz, carg, sarg

    call fatal(Ctz == 0, __FILE__, "vc_tables_init must be called before vc_refresh_source_fields")

    batch_valid = .false.

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

    !$OMP PARALLEL DO PRIVATE(kj, ii, jt, kz, gBut, gBuz, carg, sarg)
    do kj = 1, Ctz
      jt = mod(kj - 1, Cnt)
      kz = (kj - 1) / Cnt
      gBut = 0.0_rkind
      gBuz = 0.0_rkind
      do ii = 1, mn
        carg = tcos(ii, jt) * zcos(ii, kz) + tsin(ii, jt) * zsin(ii, kz)
        gBut = gBut - aze_b(ii) * carg
        gBuz = gBuz + ate_b(ii) * carg
      end do
      if (.not.enforce_stell_sym) then
        do ii = 1, mn
          sarg = tsin(ii, jt) * zcos(ii, kz) - tcos(ii, jt) * zsin(ii, kz)
          gBut = gBut - azo_b(ii) * sarg
          gBuz = gBuz + ato_b(ii) * sarg
        end do
      end if
      srcgBut(kj) = gBut
      srcgBuz(kj) = gBuz
      srcBlt(kj) = (gBut * srcgtt(kj) + gBuz * srcgtz(kj)) * srcinvsg(kj)
      srcBlz(kj) = (gBut * srcgtz(kj) + gBuz * srcgzz(kj)) * srcinvsg(kj)
    end do
    !$OMP END PARALLEL DO

  end subroutine vc_refresh_source_fields

  !> Prepare the geometry-derivative machinery: refresh the source geometry and fields, then
  !> precompute the (cos, sin) harmonics of (d dVC / d x) . a0 for every plasma-boundary degree of
  !> freedom (vc_build_deriv_batch), which vc_deriv_rhs_add serves by lookup. Must be called
  !> serially, with converged geometry and solution, before any vc_deriv_rhs_add.
  subroutine vc_prepare_deriv()
    call vc_refresh_source_fields()
    call vc_build_deriv_batch()
  end subroutine vc_prepare_deriv

  !> Precompute the (cos, sin) harmonics of (d dVC / d x) . a0 for every plasma-boundary degree of
  !> freedom at once. The perturbed integrand at a (target jk, source kj) pair is linear in the
  !> 8 geometry weights (dR0, dRs, dRt, dRz, dZ0, dZs, dZt, dZz) of the source point, with
  !> DOF-independent kernel fields K_alpha(jk, kj). A DOF activates only its R-quartet (irz = 0)
  !> or Z-quartet (irz = 1), and the quartet values depend only on the DOF's trig base (cosine for
  !> Rbc/Zbc, sine for Rbs/Zbs), so one 4-slot weight array serves both groups and the contraction
  !> over all DOFs is 4 blocked DGEMMs per group.
  subroutine vc_build_deriv_batch()
    integer :: p, ii, irz, issym, kj, jk, b, k0, blk, a, np_r, np_z, jt, kz
    real(rkind) :: mi, ni, bc, bs, czk, szk
    real(rkind) :: bh_t, bh_z, s2, gBut, gBuz, invsg
    real(rkind) :: dx, dy, dz, n1, n2, n3
    real(rkind) :: rx, ry, rz, d2, invd2, t1, fof, dfd2, xtn, xzn, p1, p2, c1, c2, c3, f0
    real(rkind) :: vtun, vtzn, vzun, vzzn
    real(rkind), allocatable :: btc(:, :), bzc(:, :), vtu(:, :), vtz(:, :), vzu(:, :), vzz(:, :)
    real(rkind), allocatable :: Kern(:, :, :), Wgt(:, :, :), dgb(:, :), dgfull(:, :)

    ! enumerate the plasma-boundary DOFs, R-type (irz = 0) first so each group is a contiguous
    ! column range of the contraction; same skip rules as dfp200 / packxi. Count in closed form:
    ! Rbc(1:mn) + Zbs(2:mn), plus Rbs(2:mn) + Zbc(1:mn) without stellarator symmetry.
    if (allocated(batch_ii)) deallocate(batch_ii, batch_irz, batch_issym, batch_e, batch_o)
    batch_n = merge(2 * mn - 1, 4 * mn - 2, enforce_stell_sym)
    call nallocate(batch_ii, 1, batch_n)
    call nallocate(batch_irz, 1, batch_n)
    call nallocate(batch_issym, 1, batch_n)
    call nallocate(batch_e, 1, mn, 1, batch_n)
    call nallocate(batch_o, 1, mn, 1, batch_n)
    p = 0
    np_r = 0
    do irz = 0, 1
      do ii = 1, mn
        do issym = 0, 1
          if (issym == 1 .and. enforce_stell_sym) cycle
          if (ii == 1 .and. irz == 1 .and. issym == 0) cycle
          if (ii == 1 .and. irz == 0 .and. issym == 1) cycle
          p = p + 1
          batch_ii(p) = ii
          batch_irz(p) = irz
          batch_issym(p) = issym
          if (irz == 0) np_r = p
        end do
      end do
    end do
    call fatal(p /= batch_n, __FILE__, "vc_build_deriv_batch: DOF enumeration disagrees with the closed-form count")
    np_z = batch_n - np_r

    ! per-source-point coefficients of d(B_theta), d(B_zeta) in the 8 weights, and the
    ! DOF-independent vectors of the tangent-perturbation terms
    call nallocate(btc, 1, Ctz, 1, 8)
    call nallocate(bzc, 1, Ctz, 1, 8)
    call nallocate(vtu, 1, Ctz, 1, 3)
    call nallocate(vtz, 1, Ctz, 1, 3)
    call nallocate(vzu, 1, Ctz, 1, 3)
    call nallocate(vzz, 1, Ctz, 1, 3)

    !$OMP PARALLEL DO PRIVATE(kj, gBut, gBuz, invsg, bh_t, bh_z, s2, czk, szk, kz)
    do kj = 1, Ctz
      gBut = srcgBut(kj)
      gBuz = srcgBuz(kj)
      invsg = srcinvsg(kj)
      bh_t = gBut * srcgtt(kj) + gBuz * srcgtz(kj)
      bh_z = gBut * srcgtz(kj) + gBuz * srcgzz(kj)
      s2 = invsg * invsg

      ! d(sqrt g) coefficients: dR0, dRs, dRt, dZs, dZt (columns 1, 2, 3, 6, 7)
      btc(kj, 1) = -bh_t * s2 * (srcZ(kj, 1) * srcR(kj, 2) - srcR(kj, 1) * srcZ(kj, 2))
      btc(kj, 2) = +bh_t * s2 * srcR(kj, 0) * srcZ(kj, 2)
      btc(kj, 3) = invsg * (2.0_rkind * srcR(kj, 2) * gBut + srcR(kj, 3) * gBuz) - bh_t * s2 * srcR(kj, 0) * srcZ(kj, 1)
      btc(kj, 4) = invsg * srcR(kj, 2) * gBuz
      btc(kj, 5) = 0.0_rkind
      btc(kj, 6) = -bh_t * s2 * srcR(kj, 0) * srcR(kj, 2)
      btc(kj, 7) = invsg * (2.0_rkind * srcZ(kj, 2) * gBut + srcZ(kj, 3) * gBuz) + bh_t * s2 * srcR(kj, 0) * srcR(kj, 1)
      btc(kj, 8) = invsg * srcZ(kj, 2) * gBuz

      bzc(kj, 1) = invsg * 2.0_rkind * srcR(kj, 0) * gBuz - bh_z * s2 * (srcZ(kj, 1) * srcR(kj, 2) - srcR(kj, 1) * srcZ(kj, 2))
      bzc(kj, 2) = +bh_z * s2 * srcR(kj, 0) * srcZ(kj, 2)
      bzc(kj, 3) = invsg * srcR(kj, 3) * gBut - bh_z * s2 * srcR(kj, 0) * srcZ(kj, 1)
      bzc(kj, 4) = invsg * (srcR(kj, 2) * gBut + 2.0_rkind * srcR(kj, 3) * gBuz)
      bzc(kj, 5) = 0.0_rkind
      bzc(kj, 6) = -bh_z * s2 * srcR(kj, 0) * srcR(kj, 2)
      bzc(kj, 7) = invsg * srcZ(kj, 3) * gBut + bh_z * s2 * srcR(kj, 0) * srcR(kj, 1)
      bzc(kj, 8) = invsg * (srcZ(kj, 2) * gBut + 2.0_rkind * srcZ(kj, 3) * gBuz)

      kz = (kj - 1) / Cnt
      czk = czeta1(kz)
      szk = szeta1(kz)
      vtu(kj, 1:3) = [-srcxt(kj, 3) * szk, srcxt(kj, 3) * czk, srcxt(kj, 1) * szk - srcxt(kj, 2) * czk]
      vtz(kj, 1:3) = [srcxt(kj, 2), -srcxt(kj, 1), 0.0_rkind]
      vzu(kj, 1:3) = [-srcxz(kj, 3) * szk, srcxz(kj, 3) * czk, srcxz(kj, 1) * szk - srcxz(kj, 2) * czk]
      vzz(kj, 1:3) = [srcxz(kj, 2), -srcxz(kj, 1), 0.0_rkind]
    end do
    !$OMP END PARALLEL DO

    call sallocate(Kern, 1, Ncomp, 1, DBLK, 1, 8, 0.0_rkind)
    call sallocate(Wgt, 1, DBLK, 1, batch_n, 1, 4, 0.0_rkind)
    call sallocate(dgb, 1, Ncomp, 1, batch_n, 0.0_rkind)
    call sallocate(dgfull, 1, Ntz, 1, batch_n, 0.0_rkind)

    do k0 = 1, Ctz, DBLK
      blk = min(DBLK, Ctz - k0 + 1)

      ! DOF-independent kernel fields on this source block
      !$OMP PARALLEL DO PRIVATE(jk, b, kj, kz, dx, dy, dz, n1, n2, n3, rx, ry, rz, d2, invd2, t1, fof, dfd2, &
      !$OMP& xtn, xzn, p1, p2, c1, c2, c3, f0, vtun, vtzn, vzun, vzzn, czk, szk)
      do jk = 1, Ncomp
        dx = Dxyz(1, jk)
        dy = Dxyz(2, jk)
        dz = Dxyz(3, jk)
        n1 = Nxyz(1, jk)
        n2 = Nxyz(2, jk)
        n3 = Nxyz(3, jk)
        do b = 1, blk
          kj = k0 + b - 1
          kz = (kj - 1) / Cnt
          czk = czeta1(kz)
          szk = szeta1(kz)
          rx = dx - srcpos(kj, 1)
          ry = dy - srcpos(kj, 2)
          rz = dz - srcpos(kj, 3)
          d2 = rx * rx + ry * ry + rz * rz + vcasingeps ** 2
          invd2 = 1.0_rkind / d2
          t1 = invd2 / sqrt(d2)
          fof = t1 + 3.0_rkind * vcasingeps ** 2 * t1 * invd2
          dfd2 = -1.5_rkind * t1 * invd2 - 7.5_rkind * vcasingeps ** 2 * t1 * invd2 ** 2

          xtn = (srcxt(kj, 2) * rz - srcxt(kj, 3) * ry) * n1 &
              + (srcxt(kj, 3) * rx - srcxt(kj, 1) * rz) * n2 &
              + (srcxt(kj, 1) * ry - srcxt(kj, 2) * rx) * n3
          xzn = (srcxz(kj, 2) * rz - srcxz(kj, 3) * ry) * n1 &
              + (srcxz(kj, 3) * rx - srcxz(kj, 1) * rz) * n2 &
              + (srcxz(kj, 1) * ry - srcxz(kj, 2) * rx) * n3

          p1 = fof * xtn
          p2 = fof * xzn
          c1 = fof * ( (szk * rz) * n1 - (czk * rz) * n2 + (czk * ry - szk * rx) * n3 )
          c2 = fof * (-ry * n1 + rx * n2)
          c3 = fof * ( (czk * rz) * n1 + (szk * rz) * n2 - (szk * ry + czk * rx) * n3 )
          vtun = fof * (vtu(kj, 1) * n1 + vtu(kj, 2) * n2 + vtu(kj, 3) * n3)
          vtzn = fof * (vtz(kj, 1) * n1 + vtz(kj, 2) * n2)
          vzun = fof * (vzu(kj, 1) * n1 + vzu(kj, 2) * n2 + vzu(kj, 3) * n3)
          vzzn = fof * (vzz(kj, 1) * n1 + vzz(kj, 2) * n2)
          f0 = -2.0_rkind * dfd2 * (srcBlz(kj) * xtn - srcBlt(kj) * xzn)

          Kern(jk, b, 1) = p1 * bzc(kj, 1) - p2 * btc(kj, 1) - srcBlz(kj) * vtun - srcBlt(kj) * c3 &
              + srcBlt(kj) * vzun + f0 * (rx * czk + ry * szk)
          Kern(jk, b, 2) = p1 * bzc(kj, 2) - p2 * btc(kj, 2)
          Kern(jk, b, 3) = p1 * bzc(kj, 3) - p2 * btc(kj, 3) + srcBlz(kj) * c1
          Kern(jk, b, 4) = p1 * bzc(kj, 4) - p2 * btc(kj, 4) - srcBlt(kj) * c1
          Kern(jk, b, 5) = -srcBlz(kj) * vtzn + srcBlt(kj) * vzzn + f0 * rz
          Kern(jk, b, 6) = p1 * bzc(kj, 6) - p2 * btc(kj, 6)
          Kern(jk, b, 7) = p1 * bzc(kj, 7) - p2 * btc(kj, 7) + srcBlz(kj) * c2
          Kern(jk, b, 8) = p1 * bzc(kj, 8) - p2 * btc(kj, 8) - srcBlt(kj) * c2
        end do
      end do
      !$OMP END PARALLEL DO

      ! per-DOF weight quartet (d.0, d.s, d.t, d.z) on this source block
      !$OMP PARALLEL DO PRIVATE(p, ii, b, kj, jt, kz, mi, ni, bc, bs)
      do p = 1, batch_n
        ii = batch_ii(p)
        mi = real(im(ii), rkind)
        ni = real(in(ii), rkind)
        do b = 1, blk
          kj = k0 + b - 1
          jt = mod(kj - 1, Cnt)
          kz = (kj - 1) / Cnt
          bc = tcos(ii, jt) * zcos(ii, kz) + tsin(ii, jt) * zsin(ii, kz)
          bs = tsin(ii, jt) * zcos(ii, kz) - tcos(ii, jt) * zsin(ii, kz)
          if (batch_irz(p) == batch_issym(p)) then ! Rbc, Zbc: cosine base
            Wgt(b, p, 1:4) = [bc, -bc * 0.5_rkind, -mi * bs, +ni * bs]
          else ! Rbs, Zbs: sine base
            Wgt(b, p, 1:4) = [bs, -bs * 0.5_rkind, +mi * bc, -ni * bc]
          end if
        end do
      end do
      !$OMP END PARALLEL DO

      ! contract: R-type DOFs (columns 1:np_r) against the R-quartet of kernel fields, Z-type
      ! against the Z-quartet
      do a = 1, 4
        if (np_r > 0) then
          call DGEMM('N', 'N', Ncomp, np_r, blk, 1.0_rkind, Kern(1, 1, a), Ncomp, &
              Wgt(1, 1, a), DBLK, 1.0_rkind, dgb(1, 1), Ncomp)
        end if
        if (np_z > 0) then
          call DGEMM('N', 'N', Ncomp, np_z, blk, 1.0_rkind, Kern(1, 1, a + 4), Ncomp, &
              Wgt(1, np_r + 1, a), DBLK, 1.0_rkind, dgb(1, np_r + 1), Ncomp)
        end if
      end do
    end do

    ! mirror-fill (odd fields under stellarator symmetry) and project all DOFs at once
    dgfull(1:Ncomp, 1:batch_n) = dgb(1:Ncomp, 1:batch_n)
    do jk = Ncomp + 1, Ntz
      dgfull(jk, 1:batch_n) = -dgb(msrc(jk), 1:batch_n)
    end do

    call DGEMM('T', 'N', mn, batch_n, Ntz, vcnorm, prjcos(1, 1), Ntz, dgfull(1, 1), Ntz, 0.0_rkind, batch_e(1, 1), mn)
    call DGEMM('T', 'N', mn, batch_n, Ntz, vcnorm, prjsin(1, 1), Ntz, dgfull(1, 1), Ntz, 0.0_rkind, batch_o(1, 1), mn)

    deallocate(btc, bzc, vtu, vtz, vzu, vzz)
    deallocate(Kern, Wgt, dgb, dgfull)

    batch_valid = .true.

  end subroutine vc_build_deriv_batch

  !> Matrix-free application of the virtual-casing operator to the current (ideriv = 0) vacuum
  !> solution: returns the (cos, sin) harmonics on the computational boundary of the plasma-induced
  !> sqrt(g) B^s. Must agree with matmul(dVC, packed solution) on the Lme/Lmf rows to machine
  !> precision, and with freeb_mod::compute_bnorm_plasmab up to quadrature error.
  subroutine vc_apply_a0(gbne, gbno)
    real(rkind), intent(out) :: gbne(1:mn), gbno(1:mn)

    integer :: jk, k0, bs_len
    real(rkind) :: acc
    real(rkind) :: gbn(1:Ntz), wtl(1:KBLK), wzl(1:KBLK)

    call vc_refresh_source_fields()

    !$OMP PARALLEL DO PRIVATE(jk, k0, bs_len, acc, wtl, wzl)
    do jk = 1, Ncomp
      acc = 0.0_rkind
      do k0 = 1, Ctz, KBLK
        bs_len = min(KBLK, Ctz - k0 + 1)
        call vc_kernel_block(jk, k0, bs_len, wtl(1:bs_len), wzl(1:bs_len))
        acc = acc + sum(srcgBut(k0:k0 + bs_len - 1) * wtl(1:bs_len)) &
            + sum(srcgBuz(k0:k0 + bs_len - 1) * wzl(1:bs_len))
      end do
      gbn(jk) = acc * vcnorm
    end do
    !$OMP END PARALLEL DO

    do jk = Ncomp + 1, Ntz
      gbn(jk) = -gbn(msrc(jk))
    end do

    call vc_project(gbn, gbne, gbno)

  end subroutine vc_apply_a0

  !> Add the eq. 4.5 term (d dVC / d x) . a0 to the perturbed right-hand side of the vacuum
  !> Beltrami system, for the single plasma-boundary harmonic described by dBdX (%ii, %irz,
  !> %issym), served by lookup from the all-DOF batch of vc_prepare_deriv. The complete perturbed
  !> rhs is -(d dMA / d x - d dVC / d x) . a0; the caller (get_perturbed_rhs) has already
  !> accumulated -(d dMA / d x) . a0 matrix-free. Thread-safe: reads only the batch tables and the
  !> THREADPRIVATE dBdX, writes only rhs.
  subroutine vc_deriv_rhs_add(lvol, NN, rhs)
    integer, intent(in) :: lvol, NN
    real(rkind), intent(inout) :: rhs(0:NN)

    integer :: ii, ip

    call fatal(.not.batch_valid, __FILE__, "vc_deriv_rhs_add requires a prior vc_prepare_deriv")

    do ip = 1, batch_n
      if (batch_ii(ip) == dBdX%ii .and. batch_irz(ip) == dBdX%irz .and. batch_issym(ip) == dBdX%issym) then
        do ii = 2, mn
          rhs(Lme(lvol, ii)) = rhs(Lme(lvol, ii)) + batch_o(ii, ip)
          if (.not.enforce_stell_sym) rhs(Lmf(lvol, ii)) = rhs(Lmf(lvol, ii)) + batch_e(ii, ip)
        end do
        return
      end if
    end do

    call fatal(.true., __FILE__, "vc_deriv_rhs_add: perturbed harmonic is not a plasma-boundary degree of freedom")

  end subroutine vc_deriv_rhs_add

  !> General-direction (Ldirgen) counterpart of vc_deriv_rhs_add: add the direction's
  !> (d dVC / d x) . a0 term by contracting the all-DOF batch of vc_prepare_deriv with the
  !> direction's plasma-boundary coefficients (cRc/cRs/cZs/cZc ~ the dirRc/dirRs/dirZs/dirZc
  !> tables of load_direction_coeffs at innout = 0; psifactor is already folded in there, exactly
  !> as in every other general-direction contraction).
  subroutine vc_deriv_rhs_add_dir(lvol, NN, rhs, cRc, cRs, cZs, cZc)
    integer, intent(in) :: lvol, NN
    real(rkind), intent(inout) :: rhs(0:NN)
    real(rkind), intent(in) :: cRc(1:mn), cRs(1:mn), cZs(1:mn), cZc(1:mn)

    integer :: ii, ip
    real(rkind) :: w, acc_o(1:mn), acc_e(1:mn)

    call fatal(.not.batch_valid, __FILE__, "vc_deriv_rhs_add_dir requires a prior vc_prepare_deriv")

    acc_o(1:mn) = 0.0_rkind
    acc_e(1:mn) = 0.0_rkind
    do ip = 1, batch_n
      if (batch_irz(ip) == 0) then
        w = merge(cRc(batch_ii(ip)), cRs(batch_ii(ip)), batch_issym(ip) == 0)
      else
        w = merge(cZs(batch_ii(ip)), cZc(batch_ii(ip)), batch_issym(ip) == 0)
      end if
      if (w == 0.0_rkind) cycle
      acc_o(1:mn) = acc_o(1:mn) + w * batch_o(1:mn, ip)
      if (.not.enforce_stell_sym) acc_e(1:mn) = acc_e(1:mn) + w * batch_e(1:mn, ip)
    end do

    do ii = 2, mn
      rhs(Lme(lvol, ii)) = rhs(Lme(lvol, ii)) + acc_o(ii)
      if (.not.enforce_stell_sym) rhs(Lmf(lvol, ii)) = rhs(Lmf(lvol, ii)) + acc_e(ii)
    end do

  end subroutine vc_deriv_rhs_add_dir

end module vc_matrix_mod
