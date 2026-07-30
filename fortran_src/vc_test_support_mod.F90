! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> Numerical diagnostics and reference oracles used only by the direct
!> (Laplace-virtual-casing) free-boundary tests. wrapper_funcs_mod forwards the
!> public entry points across the f90wrap boundary.
module vc_test_support_mod

  use allglobal_mod, only: mn, im, in, Mvol, Nt, Nz, Ntz, NAdof, iRbc, iZbs, iRbs, iZbc, &
      Ate, Aze, Ato, Azo, ImagneticOK, solution, dBdX, &
      Lcoordinatesingularity, Lplasmaregion, Lvacuumregion
  use beltrami_operator_mod, only: beltrami_operator, op_alloc, op_release
  use errors_mod, only: fatal
  use forces_mod, only: get_LU_beltrami_matrices, get_perturbed_solution
  use input_list_mod, only: Igeometry, Lfreebound, Lrad, Nvol, enforce_stell_sym, lvcvacuum
  use intghs_mod, only: intghs_workspace_destroy, intghs_workspace_init
  use magnetic_field_mod, only: compute_gB_contravariant
  use memory_allocation_mod, only: sallocate
  use memory_mod, only: allocate_Beltrami_matrices, allocate_geometry_matrices, &
      deallocate_Beltrami_matrices, deallocate_geometry_matrices
  use packing_mod, only: packab
  use precisions_mod, only: rkind
  use vc_matrix_mod, only: vc_apply_a0, vc_assemble_dvc, vc_dense_rows_dot, vc_prepare_deriv
  use volume_mod, only: lregion

  implicit none(type, external)

  private
  public :: get_vc_consistency, evaluate_vacuum_field, solve_evaluate_vacuum_field_deriv

contains

  !> Cross-check of the Laplace-virtual-casing matrix: the plasma-induced normal-field harmonics on
  !> the computational boundary computed (a) matrix-free from the kernel quadrature (vc_apply_a0)
  !> and (b) as dVC times the packed vacuum solution. (a) and (b) must agree to machine precision;
  !> (a) must agree with get_bnorm_plasmab up to quadrature error.
  subroutine get_vc_consistency(nmn, apply_e, apply_o, dense_e, dense_o)
    integer, intent(in) :: nmn
    real(rkind), intent(out) :: apply_e(1:nmn), apply_o(1:nmn), dense_e(1:nmn), dense_o(1:nmn)

    integer :: NN, ii, ll, jd
    real(rkind), allocatable :: avec(:)

    call fatal(nmn /= mn, __FILE__, "get_vc_consistency: nmn must equal mn")
    call fatal(lvcvacuum /= 1, __FILE__, "get_vc_consistency requires lvcvacuum = 1")

    NN = NAdof(Mvol)
    call vc_assemble_dvc()

    call sallocate(avec, 0, NN, 0.0_rkind)
    do ii = 1, mn
      do ll = 0, Lrad(Mvol)
        jd = Ate(Mvol, 0, ii)%i(ll)
        if (jd > 0) avec(jd) = Ate(Mvol, 0, ii)%s(ll)
        jd = Aze(Mvol, 0, ii)%i(ll)
        if (jd > 0) avec(jd) = Aze(Mvol, 0, ii)%s(ll)
        if (.not.enforce_stell_sym) then
          jd = Ato(Mvol, 0, ii)%i(ll)
          if (jd > 0) avec(jd) = Ato(Mvol, 0, ii)%s(ll)
          jd = Azo(Mvol, 0, ii)%i(ll)
          if (jd > 0) avec(jd) = Azo(Mvol, 0, ii)%s(ll)
        end if
      end do
    end do

    call vc_dense_rows_dot(NN, avec(0:NN), dense_e(1:mn), dense_o(1:mn))

    call vc_apply_a0(apply_e(1:mn), apply_o(1:mn))

    deallocate(avec)

  end subroutine get_vc_consistency

  !> Solve for the derivative of the vacuum-volume vector potential with respect to one raw
  !> plasma-boundary harmonic (ii, irz, issym), at fixed enclosed fluxes, and leave it in slot -1
  !> of Ate/Aze/Ato/Azo(Mvol,...). Uses the same matrix perturbation theory as the force Jacobian
  !> (get_perturbed_solution), including the Laplace-virtual-casing term when lvcvacuum = 1.
  !> Requires a converged field (solve_field / dforce) first.
  subroutine solve_vacuum_field_deriv(ii, irz, issym)
    integer, intent(in) :: ii, irz, issym

    integer :: NN
    type(beltrami_operator) :: oBI

    call fatal(Lfreebound /= 1, __FILE__, "solve_vacuum_field_deriv requires Lfreebound = 1")
    call fatal(ii < 1 .or. ii > mn, __FILE__, "solve_vacuum_field_deriv: invalid mode index ii")
    call fatal(irz < 0 .or. irz > 1 .or. issym < 0 .or. issym > 1, __FILE__, "solve_vacuum_field_deriv: invalid irz/issym")
    call fatal(ii == 1 .and. irz == 1 .and. issym == 0, __FILE__, &
        "solve_vacuum_field_deriv: the constant Zbs mode is not a degree of freedom")
    call fatal(ii == 1 .and. irz == 0 .and. issym == 1, __FILE__, &
        "solve_vacuum_field_deriv: the constant Rbs mode is not a degree of freedom")
    call fatal(issym == 1 .and. enforce_stell_sym, __FILE__, &
        "solve_vacuum_field_deriv: no non-stellarator-symmetric degrees of freedom")
    call fatal(.not.allocated(ImagneticOK), __FILE__, "solve_vacuum_field_deriv requires solver initialization")
    if (allocated(ImagneticOK)) then
      call fatal(.not.ImagneticOK(Mvol), __FILE__, "solve_vacuum_field_deriv requires a converged field")
    end if

    call lregion(Mvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
    NN = NAdof(Mvol)

    call op_alloc(oBI, Mvol, NN, .false.)
    call allocate_geometry_matrices(Mvol, .true., Ntz)
    call allocate_Beltrami_matrices(Mvol, .true.)
    call intghs_workspace_init(Mvol, Ntz)

    call packab('P', Mvol, NN, solution(1:NN, 0), 0)
    call get_LU_beltrami_matrices(Mvol, oBI, NN, Nt, Nz)

    if (lvcvacuum == 1) call vc_prepare_deriv()

    dBdX%vol = Mvol
    dBdX%ii = ii
    dBdX%irz = irz
    dBdX%issym = issym
    dBdX%innout = 0
    call get_perturbed_solution(Mvol, oBI, NN, Nt, Nz)
    dBdX%L = .false.

    call intghs_workspace_destroy()
    call deallocate_Beltrami_matrices(.true.)
    call deallocate_geometry_matrices(.true.)
    call op_release(oBI)

  end subroutine solve_vacuum_field_deriv

  !> Vacuum-volume coordinate map at one point: R, Z and their (s, theta, zeta) derivatives from
  !> the linear-in-s interpolation between the plasma boundary (interface Nvol) and the
  !> computational boundary (interface Mvol). Igeometry = 3 only.
  pure subroutine vacuum_geometry_at_point(lss, teta, zeta, RR, ZZ)
    real(rkind), intent(in) :: lss, teta, zeta
    real(rkind), intent(out) :: RR(0:3), ZZ(0:3)

    integer :: ii, mi, ni
    real(rkind) :: fin, fout, arg, carg, sarg, rc, zs, rs, zc

    fin = (1.0_rkind - lss) * 0.5_rkind
    fout = (1.0_rkind + lss) * 0.5_rkind

    RR(0:3) = 0.0_rkind
    ZZ(0:3) = 0.0_rkind

    do ii = 1, mn
      mi = im(ii)
      ni = in(ii)
      arg = mi * teta - ni * zeta
      carg = cos(arg)
      sarg = sin(arg)

      rc = iRbc(ii, Nvol) * fin + iRbc(ii, Mvol) * fout
      zs = iZbs(ii, Nvol) * fin + iZbs(ii, Mvol) * fout
      RR(0) = RR(0) + rc * carg
      RR(1) = RR(1) + (iRbc(ii, Mvol) - iRbc(ii, Nvol)) * carg * 0.5_rkind
      RR(2) = RR(2) - mi * rc * sarg
      RR(3) = RR(3) + ni * rc * sarg
      ZZ(0) = ZZ(0) + zs * sarg
      ZZ(1) = ZZ(1) + (iZbs(ii, Mvol) - iZbs(ii, Nvol)) * sarg * 0.5_rkind
      ZZ(2) = ZZ(2) + mi * zs * carg
      ZZ(3) = ZZ(3) - ni * zs * carg

      if (.not.enforce_stell_sym) then
        rs = iRbs(ii, Nvol) * fin + iRbs(ii, Mvol) * fout
        zc = iZbc(ii, Nvol) * fin + iZbc(ii, Mvol) * fout
        RR(0) = RR(0) + rs * sarg
        RR(1) = RR(1) + (iRbs(ii, Mvol) - iRbs(ii, Nvol)) * sarg * 0.5_rkind
        RR(2) = RR(2) + mi * rs * carg
        RR(3) = RR(3) - ni * rs * carg
        ZZ(0) = ZZ(0) + zc * carg
        ZZ(1) = ZZ(1) + (iZbc(ii, Mvol) - iZbc(ii, Nvol)) * carg * 0.5_rkind
        ZZ(2) = ZZ(2) - mi * zc * sarg
        ZZ(3) = ZZ(3) + ni * zc * sarg
      end if
    end do

  end subroutine vacuum_geometry_at_point

  !> Evaluate the converged field at arbitrary (s, theta, zeta) points of the vacuum volume.
  !>
  !> Returns gb_out = (sqrt(g) B^s, sqrt(g) B^theta, sqrt(g) B^zeta), bcyl_out =
  !> (B_R, B_phi, B_Z), and rz_out = (R, Z).
  subroutine evaluate_vacuum_field(npts, sc, tc, zc, gb_out, bcyl_out, rz_out)
    integer, intent(in) :: npts
    real(rkind), intent(in) :: sc(1:npts), tc(1:npts), zc(1:npts)
    real(rkind), intent(out) :: gb_out(1:3, 1:npts), bcyl_out(1:3, 1:npts), rz_out(1:2, 1:npts)

    call evaluate_vacuum_field_slot(npts, sc, tc, zc, 0, 0, 0, 0, gb_out, bcyl_out, rz_out)
  end subroutine evaluate_vacuum_field

  !> Solve and evaluate the derivative with respect to the supplied raw plasma-boundary harmonic.
  !> The derivative is at fixed (s, theta, zeta) and fixed enclosed fluxes. drz_out is the movement
  !> of the coordinate point and bcyl_out includes this map contribution.
  subroutine solve_evaluate_vacuum_field_deriv(npts, sc, tc, zc, ii, irz, issym, gb_out, bcyl_out, drz_out)
    integer, intent(in) :: npts, ii, irz, issym
    real(rkind), intent(in) :: sc(1:npts), tc(1:npts), zc(1:npts)
    real(rkind), intent(out) :: gb_out(1:3, 1:npts), bcyl_out(1:3, 1:npts), drz_out(1:2, 1:npts)

    call solve_vacuum_field_deriv(ii, irz, issym)
    call evaluate_vacuum_field_slot(npts, sc, tc, zc, -1, ii, irz, issym, gb_out, bcyl_out, drz_out)
  end subroutine solve_evaluate_vacuum_field_deriv

  subroutine evaluate_vacuum_field_slot(npts, sc, tc, zc, aslot, ii, irz, issym, gb_out, bcyl_out, dmap_out)
    integer, intent(in) :: npts, aslot, ii, irz, issym
    real(rkind), intent(in) :: sc(1:npts), tc(1:npts), zc(1:npts)
    real(rkind), intent(out) :: gb_out(1:3, 1:npts), bcyl_out(1:3, 1:npts), dmap_out(1:2, 1:npts)

    integer :: ip, mi, ni
    real(rkind) :: gB0(1:3), gBd(1:3), RR(0:3), ZZ(0:3), sg, dsg, arg, bc, bs, fin
    real(rkind) :: dR(0:3), dZ(0:3), bcyl0(1:3)

    call fatal(Lfreebound /= 1, __FILE__, "vacuum-field evaluation requires Lfreebound = 1")
    call fatal(Igeometry /= 3, __FILE__, "vacuum-field evaluation requires Igeometry = 3")
    call fatal(.not.allocated(ImagneticOK), __FILE__, "vacuum-field evaluation requires solver initialization")
    if (allocated(ImagneticOK)) then
      call fatal(.not.ImagneticOK(Mvol), __FILE__, "vacuum-field evaluation requires a converged field")
    end if

    call lregion(Mvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

    do ip = 1, npts

      call compute_gB_contravariant(Mvol, zc(ip), [sc(ip), tc(ip)], 0, gB0)
      call vacuum_geometry_at_point(sc(ip), tc(ip), zc(ip), RR, ZZ)
      sg = RR(0) * (ZZ(1) * RR(2) - RR(1) * ZZ(2))

      bcyl0(1) = (gB0(1) * RR(1) + gB0(2) * RR(2) + gB0(3) * RR(3)) / sg
      bcyl0(2) = gB0(3) * RR(0) / sg
      bcyl0(3) = (gB0(1) * ZZ(1) + gB0(2) * ZZ(2) + gB0(3) * ZZ(3)) / sg

      if (aslot == 0) then
        gb_out(1:3, ip) = gB0(1:3)
        bcyl_out(1:3, ip) = bcyl0(1:3)
        dmap_out(1:2, ip) = [RR(0), ZZ(0)]
        cycle
      end if

      call compute_gB_contravariant(Mvol, zc(ip), [sc(ip), tc(ip)], -1, gBd)

      ! displacement of the coordinate map: the perturbed harmonic sits on the inner interface
      ! (the plasma boundary), interpolated with weight (1-s)/2 across the vacuum annulus
      mi = im(ii)
      ni = in(ii)
      arg = mi * tc(ip) - ni * zc(ip)
      bc = cos(arg)
      bs = sin(arg)
      fin = (1.0_rkind - sc(ip)) * 0.5_rkind

      dR(0:3) = 0.0_rkind
      dZ(0:3) = 0.0_rkind
      if (irz == 0) then
        if (issym == 0) then ! Rbc
          dR(0:3) = [bc * fin, -bc * 0.5_rkind, -mi * bs * fin, +ni * bs * fin]
        else ! Rbs
          dR(0:3) = [bs * fin, -bs * 0.5_rkind, +mi * bc * fin, -ni * bc * fin]
        end if
      else
        if (issym == 0) then ! Zbs
          dZ(0:3) = [bs * fin, -bs * 0.5_rkind, +mi * bc * fin, -ni * bc * fin]
        else ! Zbc
          dZ(0:3) = [bc * fin, -bc * 0.5_rkind, -mi * bs * fin, +ni * bs * fin]
        end if
      end if

      dsg = dR(0) * (ZZ(1) * RR(2) - RR(1) * ZZ(2)) &
          + RR(0) * (dZ(1) * RR(2) + ZZ(1) * dR(2) - dR(1) * ZZ(2) - RR(1) * dZ(2))

      gb_out(1:3, ip) = gBd(1:3)
      bcyl_out(1, ip) = (gBd(1) * RR(1) + gBd(2) * RR(2) + gBd(3) * RR(3) &
          + gB0(1) * dR(1) + gB0(2) * dR(2) + gB0(3) * dR(3)) / sg - bcyl0(1) * dsg / sg
      bcyl_out(2, ip) = (gBd(3) * RR(0) + gB0(3) * dR(0)) / sg - bcyl0(2) * dsg / sg
      bcyl_out(3, ip) = (gBd(1) * ZZ(1) + gBd(2) * ZZ(2) + gBd(3) * ZZ(3) &
          + gB0(1) * dZ(1) + gB0(2) * dZ(2) + gB0(3) * dZ(3)) / sg - bcyl0(3) * dsg / sg
      dmap_out(1:2, ip) = [dR(0), dZ(0)]

    end do

  end subroutine evaluate_vacuum_field_slot

end module vc_test_support_mod
