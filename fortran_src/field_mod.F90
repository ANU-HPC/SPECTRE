! SPDX-FileCopyrightText: 2026 Proxima Fusion
! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module field_mod
#include "mpi_precomp.h"
  USE_MPI
  use omp_lib

  use, intrinsic :: ieee_arithmetic

  use allglobal_mod, only : myid, cpus, MPI_COMM_SPEC, ext, ivol, im, in, Mvol, Lcoordinatesingularity, Lplasmaregion, &
      Lvacuumregion, cheby, zernike, Ate, Aze, Ato, Azo, sg, inv_jac, guvij, Rij, Zij, Nt, Nz, efmn, ofmn, cfmn, sfmn, &
      pi2nfp, dtflux, dpflux, ncpu, mn, Ntz, LGdof, NGdof, iRbc, iZbs, iRbs, iZbc, &
      ImagneticOK, solution, &
      IPDt, LocalConstraint, dBBdmp, dmupfdx, dRodR, dRodZ, dZodR, dZodZ, dRadR, dRadZ, dZadR, dZadZ, &
      ijreal, ijimag, jireal, Iquad, &
      LcoordinateSingularity, iRij, iZij, dRij, dZij, tRij, tZij, NAdof, mns, dBdX, sweight, &
      psifactor, IsMyVolume, IsMyVolumeValue, WhichCpuID, cosi, sini, Bemn, &
      Bomn, lsavedguvij, guvijsave, Tss, Dtc, Dzc, Ttc, Tzc, Tsc, Dts, Dzs, Tts, Tzs, &
      build_vector_potential, vvolume, xoffset, IPdtdPf, dvolume, jiimag, jkimag, jkreal, kjimag, kjreal, dMA, dMD, &
      Ldirgen, force_jvp_status
  use constants_mod, only : pi2, mu0
  use errors_mod, only: fatal
  use vc_matrix_mod, only: vc_prepare_deriv
  use forces_mod, only: dfp100, get_perturbed_solution, evaluate_dmupfdx, get_LU_beltrami_matrices, &
      invalidate_dmupfdx_lhs, ensure_dmupfdx_lhs, get_perturbed_rhs, unpack_perturbed_solution, &
      load_direction_coeffs, evaluate_dmupfdx_dir
  use geometry_mod, only: compute_coordinates
  use intghs_mod, only: intghs_workspace_init, intghs_workspace_destroy, intghs_alloc_wk_perturbed
  use perturbed_workspace_mod, only: perturbed_rhs_workspace, alloc_perturbed_workspace, dealloc_perturbed_workspace, &
      matvec_multipliers, alloc_matvec_multipliers, dealloc_matvec_multipliers
  use input_list_mod, only : Nfp, Igeometry, Nvol, Lrad, tflux, pflux, phiedge, pscale, helicity, &
      adiabatic, gamma, Rbc, Zbs, Rbs, Zbc, Lconstraint, Lfreebound, lvcvacuum, &
      mu, Isurf, Ivolume, Mpol, enforce_stell_sym, ngmresrestart, nitergmres, epsgmres_jac
  use io_mod, only: stderr
  use memory_allocation_mod, only: sallocate, nallocate
  use memory_mod, only: allocate_Beltrami_matrices, &
      allocate_geometry_matrices, &
      deallocate_Beltrami_matrices, &
      deallocate_geometry_matrices
  use mpi_collectives_mod, only: broadcast
  use numerical_mod, only: invfft, set_fftw_plan
  use packing_mod, only: packxi, packab
  use precisions_mod, only: rkind, small
  use beltrami_operator_mod, only: beltrami_operator, op_alloc, op_release, op_solve_multi, &
      op_select_backend, op_bind_iter_sparse, op_solve_multi_iter, BK_ITER, op_cache_take, op_cache_put
  use metric_mod, only: compute_guvijsave
  use matrices_mod, only: matrix
  use chebyshev_mod, only: volume_integrate_chebyshev
  use volume_mod, only: lregion, volume
  use coordinate_mod, only: calc_axis_deriv_grid

  implicit none

  private
  public :: solve_field, solve_field_deriv, calc_b2mag_jump, calc_b2mag_jump_deriv, force_real_lforce
  public :: solve_field_deriv_dir
  public :: alloc_ntz_arrays, dealloc_ntz_arrays

  real(rkind), allocatable :: freal_jac(:, :, :, :, :)
  real(rkind), allocatable :: freal_dBBdmp(:, :, :, :)

  real(rkind), allocatable :: efmn_vec(:, :, :, :), cfmn_vec(:, :, :, :), ofmn_vec(:, :, :, :), sfmn_vec(:, :, :, :)

  integer, public :: mpol_fjac, ntor_fjac, nt_alloc, nz_alloc

contains

  subroutine solve_field(position, recompute_axis)
    logical, intent(in) :: recompute_axis
    real(rkind), intent(out) :: position(0:NGdof)

    integer :: vvol, ii, Ndofgl
    integer :: NN, cpu_id, ideriv
    real(rkind) :: Xdof(1:Mvol - 1)
    integer :: ipiv(1:Mvol)
    real(rkind), allocatable :: Fvec(:), dpfluxout(:)
    integer :: idgesv

    call set_fftw_plan(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), &
        iRbs(1:mn, 0:Mvol), iZbc(1:mn, 0:Mvol), 'U', .false., recompute_axis)

    if (LocalConstraint) then ! Local constraint case - simply call dfp100 and then dfp200

      call sallocate(Fvec, 1, Mvol - 1, 0.0_rkind)

      Ndofgl = 0
      Fvec(1:Mvol - 1) = 0
      Xdof(1:Mvol - 1) = dpflux(2:Mvol) + xoffset

      ! Solve for field
      dBdX%L = .false.
      call dfp100(Ndofgl, Xdof, Fvec, .false.)

      deallocate(Fvec)

    else ! Global constraint - call the master thread calls hybrd1 on dfp100, others call dfp100_loop.

      IPDtdPf = 0.0_rkind
      Xdof(1:Mvol - 1) = dpflux(2:Mvol) + xoffset

      if (Lfreebound == 1) then
        ! Mvol-1 surface current plus 1 poloidal linking current constraints
        Ndofgl = Mvol
      else
        ! add an additional constraint to make the total pflux = 0
        if (Igeometry == 1) then
          Ndofgl = Mvol
        else
          ! Mvol-1 surface current constraints
          Ndofgl = Mvol - 1
        end if
      end if

      call sallocate(Fvec, 1, Ndofgl, 0.0_rkind)

      call dfp100(Ndofgl, Xdof(1:Mvol - 1), Fvec(1:Ndofgl), .false.)

      call sallocate(dpfluxout, 1, Ndofgl, 0.0_rkind)
      if (myid == 0) then

        dpfluxout = Fvec
        call DGESV(Ndofgl, 1, IPdtdPf, Ndofgl, ipiv, dpfluxout, Ndofgl, idgesv)

        ! one step Newton's method
        dpflux(2:Mvol) = dpflux(2:Mvol) - dpfluxout(1:Mvol - 1)

        if (Igeometry == 1) then
          dpflux(1) = dpflux(1) - dpfluxout(Mvol)
        end if

        if (Lfreebound == 1) then
          dtflux(Mvol) = dtflux(Mvol) - dpfluxout(Mvol)
        end if
      end if

      ! Broadcast the field and pflux
      call broadcast(dpfluxout(1:Ndofgl), Ndofgl, 0)
      call broadcast(dpflux(1:Mvol), Mvol, 0)
      if (Lfreebound == 1) then
        call broadcast(dtflux(Mvol), 0)
      end if

      do vvol = 2, Mvol
        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        NN = NAdof(vvol)

        call sallocate(solution, 1, NN, 0, 2, 0.0_rkind)

        ! Pack field and its derivatives
        call packab('P', vvol, NN, solution(1:NN, 0), 0)
        call packab('P', vvol, NN, solution(1:NN, 2), 2)

        if (Lfreebound == 1 .and. (vvol == Mvol) ) then
          call packab('P', vvol, NN, solution(1:NN, 1), 1)
          solution(1:NN, 0) = solution(1:NN, 0) - dpfluxout(vvol - 1) * solution(1:NN, 2) & ! derivative w.r.t pflux
              - dpfluxout(vvol) * solution(1:NN, 1)! derivative w.r.t tflux
        else
          solution(1:NN, 0) = solution(1:NN, 0) - dpfluxout(vvol - 1) * solution(1:NN, 2)
        end if

        ! Unpack field in vector potential Fourier harmonics
        call packab('U', vvol, NN, solution(1:NN, 0), 0)

        deallocate(solution)
      end do

      !add an additional constraint to make the total pflux 0
      if (Igeometry == 1) then

        vvol = 1

        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then

        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        else
          NN = NAdof(vvol)

          call sallocate(solution, 1, NN, 0, 2, 0.0_rkind)

          ! Pack field and its derivatives
          call packab('P', vvol, NN, solution(1:NN, 0), 0)
          call packab('P', vvol, NN, solution(1:NN, 2), 2)

          ! compute the field with renewed dpflux via single Newton method step
          solution(1:NN, 0) = solution(1:NN, 0) - dpfluxout(Mvol) * solution(1:NN, 2)

          ! Unpack field in vector potential Fourier harmonics
          call packab('U', vvol, NN, solution(1:NN, 0), 0)

          deallocate(solution)
        end if
      end if

      deallocate(Fvec)
      deallocate(dpfluxout)
    end if

    ! Finally broadcast the field information to all threads from the thread which did the computation
    do vvol = 1, Mvol
      cpu_id = WhichCpuID(vvol, ncpu)

      call broadcast(ImagneticOK(vvol), cpu_id)

      do ideriv = 0, 2
        if (ideriv /= 0) cycle
        do ii = 1, mn
          call broadcast(Ate(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
          call broadcast(Aze(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
        end do
      end do

      if (.not.enforce_stell_sym) then
        do ideriv = 0, 2
          if (ideriv /= 0) cycle
          do ii = 1, mn
            call broadcast(Ato(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
            call broadcast(Azo(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
          end do
        end do
      end if 
    end do
  end subroutine solve_field

  subroutine calc_b2mag_jump(dBBtotal, nt_dbb, nz_dbb)
    real(rkind), intent(out) :: dBBtotal(1:Mvol - 1, 1:(nt_dbb*nz_dbb))
    integer, intent(in) :: nt_dbb, nz_dbb

    integer :: Lcurvature, ll, NN, vvol, iocons, ntz_dbb

    real(rkind) :: lss
    real(rkind) :: dBBcurr(1:Mvol, 1:(nt_dbb*nz_dbb), 0:1)
    real(rkind) :: dAt(1:(nt_dbb*nz_dbb)), dAz(1:(nt_dbb*nz_dbb)) 

    ntz_dbb = nt_dbb * nz_dbb

    dBBcurr(1:Mvol, 1:(nt_dbb*nz_dbb), 0:1) = 0.0_rkind
    dAt(1:ntz_dbb) = 0.0_rkind
    dAz(1:ntz_dbb) = 0.0_rkind

    Lcurvature = 1

    Lsavedguvij = .false.

    call set_fftw_plan(nt_dbb, nz_dbb)
    call alloc_ntz_arrays(nt_dbb, nz_dbb)

    if (LocalConstraint) then

      do vvol = 1, Mvol
        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

        dBdX%vol = vvol
        ll = Lrad(vvol)
        NN = NAdof(vvol)

        do iocons = 0, 1 ! construct field magnitude on inner and outer interfaces; inside do vvol
          if (vvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis)
          if (vvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     ; there are no constraints at outer boundary
          
          lss = 2.0_rkind * iocons - 1.0_rkind

          call compute_coordinates(vvol, lss, Lcurvature, nt_dbb, nz_dbb, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)! get coordinates and derivatives wrt Rj, Zj, at specific radial location

          ! Compute covariant vector potential. Stored in efmn, ofmn, cfmn, sfmn.
          call build_vector_potential(vvol, iocons, 0, Lcoordinatesingularity, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn))

          call invfft(mn, im, in, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), dAt(1:ntz_dbb), dAz(1:ntz_dbb))

          dBBcurr(vvol, 1:ntz_dbb, iocons) = 0.5_rkind * (dAz(1:ntz_dbb) * dAz(1:ntz_dbb) * guvij(1:ntz_dbb, 2, 2, 0) &
              - 2.0_rkind * dAz(1:ntz_dbb) * dAt(1:ntz_dbb) * guvij(1:ntz_dbb, 2, 3, 0) &
              + dAt(1:ntz_dbb) * dAt(1:ntz_dbb) * guvij(1:ntz_dbb, 3, 3, 0)) * inv_jac(1:ntz_dbb) ** 2
        end do
      end do

    else ! CASE SEMI GLOBAL CONSTRAINT
      do vvol = 1, Mvol
        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
        ll = Lrad(vvol)
        NN = NAdof(vvol)

        do iocons = 0, 1 ! construct magnitude of magnetic field on inner and outer interfaces
          if (vvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (i.e. coordinate axis)
          if (vvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary -> there are no constraints at outer boundary
          
          lss = 2.0_rkind * iocons - 1.0_rkind
          
          call compute_coordinates(vvol, lss, Lcurvature, nt_dbb, nz_dbb, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)! get coordinates and derivatives wrt Rj, Zj, at specific radial location
          
          call build_vector_potential(vvol, iocons, 0, Lcoordinatesingularity, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn))
          
          call invfft(mn, im, in, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), dAt(1:ntz_dbb), dAz(1:ntz_dbb))

          dBBcurr(vvol, 1:ntz_dbb, iocons) = 0.5_rkind * (dAz(1:ntz_dbb) * dAz(1:ntz_dbb) * guvij(1:ntz_dbb, 2, 2, 0) &
              - 2.0_rkind * dAz(1:ntz_dbb) * dAt(1:ntz_dbb) * guvij(1:ntz_dbb, 2, 3, 0) &
              + dAt(1:ntz_dbb) * dAt(1:ntz_dbb) * guvij(1:ntz_dbb, 3, 3, 0)) * inv_jac(1:ntz_dbb) ** 2
        end do
      end do
    end if

    do vvol = 1, Mvol
      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      call broadcast(dBBcurr(vvol, 1:ntz_dbb, 0:1), 2 * ntz_dbb, modulo(vvol - 1, ncpu))
    end do

    do vvol = 1, Mvol - 1
      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

      if (ImagneticOK(vvol) .and. ImagneticOK(vvol + 1)) then ! the magnetic fields in the volumes adjacent to this interface are valid
        dBBtotal(vvol, 1:ntz_dbb) = dBBcurr(vvol + 1, 1:ntz_dbb, 0) - dBBcurr(vvol, 1:ntz_dbb, 1)
      else
        write(stderr, *)"Error: weird!"
      end if
    end do

  end subroutine calc_b2mag_jump

  subroutine solve_field_deriv(position, mpol_rz, ntor_rz, recompute_axis)
    real(rkind), intent(in) :: position(0:NGdof)
    integer, intent(in) :: mpol_rz, ntor_rz
    logical, intent(in) :: recompute_axis

    logical :: LInnerVolume
    integer :: id_lim, ndof_total, vflag, lvol, ll, NN, ideriv
    integer :: vvol, innout, ii, jj, irz, issym, iocons, idof, idof_total
    integer :: Lcurvature, id, ierr
    integer :: cpu_id, cpu_id1, even_or_odd, vol_parity
    integer :: tag, tag2
    MPI_STATUS_TYPE :: stat
    integer :: lldof, lquad, max_iquad, max_lrad, max_nadof, max_lldof

    real(rkind) :: dBB(1:Ntz, -1:2)

    integer :: ndof_batch_max, nb, ic, idgetrs, NN_v, NN_v1
    integer, allocatable :: dof_idof(:), dof_ii(:), dof_irz(:), dof_issym(:)
    real(rkind), allocatable :: rhs_v(:, :), rhs_v1(:, :), rhs_col(:)

    type(perturbed_rhs_workspace), allocatable :: pws(:)
    real(rkind), allocatable :: rhs_col_t(:, :)
    integer :: nthreads, tid
    integer :: aslot, idof_total_base ! per-thread Ate storage slot; base of the idof_total counter
    logical :: Lcoord_v, Lcoord_v1 ! per-iteration region flags for the parallel RHS build
    integer :: bknd, its_j, ierr_j ! iterative-backend selection + per-interface GMRES bookkeeping
    type(matvec_multipliers) :: lm_jac ! matvec Lagrange-multiplier scratch for the iterative column solves

    type(beltrami_operator) :: oBI(1:Mvol)

    dBB(1:Ntz, -1:2) = 0.0_rkind

    mpol_fjac = mpol_rz
    ntor_fjac = ntor_rz

    if (Lconstraint == 1 .or. Lconstraint == 3) then
      id_lim = 2 ! evaluate B^2 and derivatives wrt interface geometry, mu, and pflux
    else
      id_lim = 0 ! evaluate B^2 and derivative wrt interface geometry
    end if

    if (enforce_stell_sym) then
      ndof_total = (2*mn-1)*(mvol-1)*2
    else
      ndof_total = (2*mn-1)*(mvol-1)*2*2 ! To be checked if sufficient
    end if

    call sallocate(efmn_vec, 1, mn, -1, id_lim, 0, 1, 1, ndof_total, 0.0_rkind)
    call sallocate(cfmn_vec, 1, mn, -1, id_lim, 0, 1, 1, ndof_total, 0.0_rkind)
    call sallocate(ofmn_vec, 1, mn, -1, id_lim, 0, 1, 1, ndof_total, 0.0_rkind)
    call sallocate(sfmn_vec, 1, mn, -1, id_lim, 0, 1, 1, ndof_total, 0.0_rkind)
    
    if (LocalConstraint) then
      call sallocate(dmupfdx, 1, Mvol, 1, 1, 1, 2, 1, LGdof, 0, 1, 0.0_rkind)
    else
      call sallocate(dmupfdx, 1, Mvol, 1, Mvol - 1, 1, 2, 1, LGdof, 1, 1, 0.0_rkind)! TODO change the format to put vvol in last index position...
    end if

    ! Rebuild the cached dmupfdx LHS when geometry changes.
    call invalidate_dmupfdx_lhs()

    do vvol = 1, Mvol
      cpu_id = WhichCpuID(vvol, ncpu)

      call broadcast(ImagneticOK(vvol), cpu_id)

      do ideriv = 0, 2
        if ( (.not..true.) .and. (ideriv /= 0) ) cycle
        do ii = 1, mn
          call broadcast(Ate(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
          call broadcast(Aze(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
        end do
      end do

      if (.not.enforce_stell_sym) then
        do ideriv = 0, 2
          if ( (.not..true.) .and. (ideriv /= 0) ) cycle
          do ii = 1, mn
            call broadcast(Ato(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
            call broadcast(Azo(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
          end do
        end do
      end if
    end do

    call set_fftw_plan(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), &
      iRbs(1:mn, 0:Mvol), iZbc(1:mn, 0:Mvol), 'U', .true., recompute_axis)

    idof_total = 0

    ! Virtual-casing source fields and the all-DOF (d dVC/dx).a0 batch consumed by vc_deriv_rhs_add:
    ! built once, serially, from the converged solution and geometry. Both constraint branches below
    ! reach vc_deriv_rhs_add through get_perturbed_rhs, which is then a thread-safe lookup.
    if (lvcvacuum == 1 .and. Lfreebound == 1) call vc_prepare_deriv()

    if (LocalConstraint) then

      do vvol = 1, Mvol

        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.

        dBdX%vol = vvol ! Label
        ll = Lrad(vvol)! Shorthand
        NN = NAdof(vvol)! shorthand

        vflag = 1
        call volume(vvol, vflag, Nt, Nz, Lcoordinatesingularity)! compute volume

        do iocons = 0, 1 ! construct field magnitude on inner and outer interfaces; inside do vvol

          if (vvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis)
          if (vvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     ; there are no constraints at outer boundary

          ideriv = 0
          id = ideriv
          call force_real_lforce(vvol, iocons, ideriv, Nt, Nz, dBB(1:Ntz, id), Lcoordinatesingularity)

        end do ! end of do iocons = 0, 1

        ! Allocate the Beltrami operator handle (factor storage has an extra dimension for the
        ! helicity constraint, so dimension is NN+1 -> 0:NN)
        call op_alloc(oBI(vvol), vvol, NN, Lconstraint == 2)

        ! initialize matrices for Beltrami linear system
        call allocate_geometry_matrices(vvol, .true., Ntz)
        call allocate_Beltrami_matrices(vvol, .true.)
        call intghs_workspace_init(vvol, Ntz)

        call packab('P', vvol, NN, solution(1:NN, 0), 0)! packing, put the solutions back to the solution matrix

        ! Invert beltrami matrix. Required for matrix perturbation theory
        call get_LU_Beltrami_matrices(vvol, oBI(vvol), NN, Nt, Nz)

        call intghs_workspace_destroy()
        call deallocate_Beltrami_matrices(.true.)
        call deallocate_geometry_matrices(.true.)
      end do

      dBdX%L = .true. ! will need derivatives

      do vvol = 1, Mvol

        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.

        dBdX%vol = vvol ! Label
        ll = Lrad(vvol)! Shorthand
        NN = NAdof(vvol)! shorthand

        ! initialize matrices for Beltrami linear system
        call allocate_geometry_matrices(vvol, .true., Ntz)
        call allocate_Beltrami_matrices(vvol, .true.)
        call intghs_workspace_init(vvol, Ntz)    

        idof = 0 ! labels degree of freedom; local to interface

        do ii = 1, mn ! loop over deformations in Fourier harmonics; inside do vvol

          dBdX%ii = ii ! controls construction of derivatives in subroutines called below
          do irz = 0, 1 ! loop over deformations in R and Z; inside do vvol; inside do ii

            if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z
            dBdX%irz = irz ! controls construction of derivatives

            do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms

              if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
              if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
              if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

              dBdX%issym = issym ! controls construction of derivatives
              idof = idof + 1 ! this labels the degree-of-freedom that the derivative is taken with respect to; this is outside do innout

              do innout = 0, 1 ! loop over deformations to inner and outer interface; inside do vvol; inside do ii; inside do irz

                if (vvol == 1 .and. innout == 0) cycle ! no degrees of freedom at coordinate axis / fixed inner boundary
                if (vvol == Mvol .and. innout == 1) cycle ! no degress of freedom                      fixed outer boundary; for linearized displacement

                dBdX%innout = innout

                ! Evaluate derivatives of field w.r.t geometry
                call get_perturbed_solution(vvol, oBI(vvol), NN, Nt, Nz)

                ! Helicity multiplier and poloidal flux derivatives w.r.t geometry
                call evaluate_dmupfdx(innout, idof, ii, Nt, Nz, -1, Lcoordinatesingularity)

                idof_total = idof_total + 1

                call calc_vecpot_deriv(vvol, idof, idof_total, Lcoordinatesingularity, -1)

                Lsavedguvij = .false.
              end do ! matches do innout
            end do ! matches do issym
          end do ! matches do irz
        end do ! matches do ii

        call intghs_workspace_destroy()
        call deallocate_Beltrami_matrices(.true.)
        call deallocate_geometry_matrices(.true.)

        call op_release(oBI(vvol))
      end do ! matches do vvol = 1, Mvol

    else ! Global constraint
      
      do vvol = 1, Mvol
        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.
        ll = Lrad(vvol)! Shorthand
        NN = NAdof(vvol)! shorthand

        vflag = 1
        call volume(vvol, vflag, Nt, Nz, Lcoordinatesingularity)! compute volume

        do iocons = 0, 1 ! construct field magnitude on inner and outer interfaces; inside do vvol

          if (vvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis)
          if (vvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     ; there are no constraints at outer boundary

          ideriv = 0
          id = ideriv
          call force_real_lforce(vvol, iocons, ideriv, Nt, Nz, dBB(1:Ntz, id), Lcoordinatesingularity)

        end do ! end of do iocons = 0, 1
      end do

      dBdX%L = .true. ! will need derivatives

      max_iquad = 1
      max_lrad = 1
      max_nadof = 1
      max_lldof = 1

      ! Each volume will need solution and derivative information in vacuum region. Thus, broadcast!
      ! Derivative of w.r.t geometry only required if plasma interface is perturbed - this is
      ! calculated below

      ! Evaluation of froce gradient in case of semi-global constraint (for now, this is specific to Lconstraint=3)
      ! Loop over all geometrical degrees of freedom. Thi means we loop over
      ! - each interface (vvol)
      ! - each Fourier mode (ii)
      ! - on R or Z (irz)
      ! - on symmetric and non stellarator symmetric terms (issym)

      ! Then the perturbed solution is evaluated on volumes sharing the interface vvol (volume vvol and vvol+1).
      ! i.e. relevant routines are run with lvol=vvol, innout=1 and lvol=vvol+1, innout=0

      ! First invert Beltrami matrices and store them in OBI
      do vvol = 1, Mvol
        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc. ; TODO: maybe not necessary, remove
        NN = NAdof(vvol)! shorthand
        ll = Lrad(vvol)

        bknd = op_select_backend(vvol)
        if (bknd == BK_ITER) then
          oBI(vvol)%vvol = vvol ! no dense NN x NN factor; op_bind_iter_sparse allocates the CSR factor
          oBI(vvol)%NN = NN
        end if

        ! Parallelization: each rank factorizes only its own volumes. Every perturbed solve below
        ! runs on the volume's owning rank (the LinnerVolume selection), and the perturbed
        ! solutions are exchanged directly (MPI_SEND/RECV of the slot -1 vector potential), so an
        ! operator handle is never consumed off-rank: nothing is broadcast, no off-rank dense
        ! factor is allocated, and the matrix-free backend works per-rank exactly as single-rank.
        call IsMyVolume(vvol)
        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        if (Lcoordinatesingularity) then
          ! lldof = (Lrad(lvol) - mod(Lrad(lvol),2)) / 2
          max_lldof = max(max_lldof, (ll - mod(ll, 2)) / 2)
        else
          ! lldof = Lrad(lvol)
          max_lldof = max(max_lldof, ll)
        end if

        max_iquad = max(max_iquad, Iquad(vvol))
        max_lrad = max(max_lrad, ll)
        max_nadof = max(max_nadof, NN)

        ! Invert LHS of Beltrami system and store it in oBI. This will be used to
        ! evaluate derivatives of solution.

        if (bknd == BK_ITER) then
          ! matrix-free: integrate only the diagonal mode-pairs and build the exact same-(m,n)
          ! preconditioner from them; never form the dense dMA/dMD (this is the Jacobian memory win).
          ! Passing .false. to allocate_geometry_matrices selects the mode-collapsed tensors and
          ! skips dMA/dMD.
          call allocate_geometry_matrices(vvol, .false., Ntz)
          dBdX%L = .false.
          ! Lsavedguvij is THREADPRIVATE and its value on entry here is not established, so
          ! volume_integrate_chebyshev could otherwise skip the metric build and integrate a stale
          ! guvijsave. Force the recompute.
          Lsavedguvij = .false.
          call volume_integrate_chebyshev(Iquad(vvol), mn, vvol, ll, Nt, Nz, diag_only=.true.)
          call op_bind_iter_sparse(oBI(vvol), NN, Lcoordinatesingularity, mu(vvol))
          call deallocate_geometry_matrices(.false.)
          dBdX%L = .true.
        else
          call op_alloc(oBI(vvol), vvol, NN, Lconstraint == 2)
          call allocate_geometry_matrices(vvol, .true., Ntz)
          call allocate_Beltrami_matrices(vvol, .true.)

          call get_LU_beltrami_matrices(vvol, oBI(vvol), NN, Nt, Nz)

          call deallocate_Beltrami_matrices(.true.)
          call deallocate_geometry_matrices(.true.)
        end if
      end do ! end of vvol = 1, Mvol

      call nallocate(solution, 1, max_nadof, -1, 2)
      call nallocate(guvijsave, 1, Ntz, 1, 3, 1, 3, 1, max_iquad)
      call intghs_alloc_wk_perturbed(Ntz, mn, max_iquad, max_lrad, mpol, .not.enforce_stell_sym)
      call nallocate(Tss, 0, max_lldof, 1, mn)
      call nallocate(Dtc, 0, max_lldof, 1, mn)
      call nallocate(Dzc, 0, max_lldof, 1, mn)

      if (.not.enforce_stell_sym) then
        call nallocate(Tsc, 0, max_lldof, 1, mn)
        call nallocate(Dts, 0, max_lldof, 1, mn)
        call nallocate(Dzs, 0, max_lldof, 1, mn)
      end if

      ! Possible number of DOFs of a single perturbed interface (2 R/Z * 2 sym * mn modes).
      ndof_batch_max = 4 * mn
      allocate(dof_idof(1:ndof_batch_max), dof_ii(1:ndof_batch_max))
      allocate(dof_irz(1:ndof_batch_max), dof_issym(1:ndof_batch_max))
      allocate(rhs_v(1:max_nadof, 1:ndof_batch_max), rhs_v1(1:max_nadof, 1:ndof_batch_max))
      allocate(rhs_col(0:max_nadof))

      nthreads = omp_get_max_threads()
      allocate(pws(0:nthreads - 1))
      do tid = 0, nthreads - 1
        call alloc_perturbed_workspace(pws(tid), Ntz, mn, max_iquad, max_lrad, mpol, max_lldof)
      end do
      allocate(rhs_col_t(0:max_nadof, 0:nthreads - 1))
      call alloc_matvec_multipliers(lm_jac, mn) ! scratch for the iterative column solves (serial phase)

      ! Warm the DOF-independent dmupfdx LHS cache serially, BEFORE any parallel region calls
      ! evaluate_dmupfdx: its own lazy build-if-invalid check is a data race if two threads can
      ! reach it concurrently with the cache still cold. See ensure_dmupfdx_lhs for details.
      if (Lconstraint == 3) call ensure_dmupfdx_lhs(Nt, Nz)

      ! Loop on perturbed interfaces
      do even_or_odd = 0, 1 ! First loop on even interfaces perturbation, then on odd interfaces. This allow efficient parallelization
        do vvol = 1, Mvol - 1 !labels which interface is perturbed

          ! Parallelization - parallelization on perturbed interface and not on volume (outermost loop)
          vol_parity = MODULO(vvol, 2)
          if ( (vol_parity == 0) .and. (even_or_odd == 1) ) cycle ! even_or_odd=1 thus perturb only odd  interfaces
          if ( (vol_parity == 1) .and. (even_or_odd == 0) ) cycle ! even_or_odd=0 thus perturb only even interfaces

          call IsMyVolume(vvol)

          if (IsMyVolumeValue == 0) then ! This CPU does not deal with interface's inner volume

            call IsMyVolume(vvol + 1)

            if (IsMyVolumeValue == 0) then ! This CPU does not deal with interface's outer volume either - cycle
              cycle
            else if (IsMyVolumeValue == -1) then
              call fatal(.true., __FILE__, "Unassociated volume")
            else
              LInnerVolume = .false.
            end if

          else if (IsMyVolumeValue == -1) then
            call fatal(.true., __FILE__, "Unassociated volume")
          else
            LinnerVolume = .true.
          end if

          dBdX%vol = vvol ! Perturbed interface
          idof = 0 ! labels degree of freedom of interface vvol

          ! On one MPI rank all DOFs of this perturbed interface share the LU factors
          ! oBI(vvol) and oBI(vvol+1). Gather DOF RHS into matrices and run ONE multi-RHS DGETRS
          ! instead of one single-RHS solve per (DOF, volume). 
          if (ncpu == 1 .and. Lconstraint /= 2) then

            NN_v = NAdof(vvol)
            NN_v1 = NAdof(vvol + 1)
            nb = 0

            ! Pass 1a: enumerate the kept DOFs of this interface and assign each a column index.
            do ii = 1, mn
              do irz = 0, 1
                if (irz == 1 .and. Igeometry < 3) cycle
                do issym = 0, 1
                  if (issym == 1 .and. enforce_stell_sym) cycle
                  if (ii == 1 .and. irz == 1 .and. issym == 0) cycle
                  if (ii == 1 .and. irz == 0 .and. issym == 1) cycle

                  idof = idof + 1

                  if (im(ii) > mpol_fjac .or. abs(in(ii)) > ntor_fjac) cycle

                  nb = nb + 1
                  dof_idof(nb) = idof
                  dof_ii(nb) = ii
                  dof_irz(nb) = irz
                  dof_issym(nb) = issym
                end do
              end do
            end do

            ! Pass 1b: build each kept DOF's RHS column for both neighbouring volumes. Parallel over
            ! DOFs: scratch lives in pws(tid); dBdX, the region flags and Lsavedguvij are THREADPRIVATE
            ! (dBdX%vol is therefore re-set per iteration); get_perturbed_rhs holds no shared state.
            ! Region flags go to the loop-private Lcoord_v/Lcoord_v1.
            !$OMP PARALLEL DO IF(nthreads > 1) SCHEDULE(dynamic) DEFAULT(shared) &
            !$OMP& PRIVATE(ic, tid, Lcoord_v, Lcoord_v1)
            do ic = 1, nb
              tid = omp_get_thread_num()

              dBdX%vol = vvol
              dBdX%ii = dof_ii(ic)
              dBdX%irz = dof_irz(ic)
              dBdX%issym = dof_issym(ic)
              dBdX%L = .true.

              ! lvol = vvol (innout = 1)
              dBdX%innout = 1
              call lregion(vvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
              Lsavedguvij = .false.
              call get_perturbed_rhs(vvol, NN_v, Nt, Nz, Lcoord_v, rhs_col_t(0:NN_v, tid), pws(tid))
              rhs_v(1:NN_v, ic) = rhs_col_t(1:NN_v, tid)

              ! lvol = vvol + 1 (innout = 0)
              dBdX%innout = 0
              call lregion(vvol + 1, Igeometry, Nvol, Lcoord_v1, Lplasmaregion, Lvacuumregion)
              Lsavedguvij = .false.
              call get_perturbed_rhs(vvol + 1, NN_v1, Nt, Nz, Lcoord_v1, rhs_col_t(0:NN_v1, tid), pws(tid))
              rhs_v1(1:NN_v1, ic) = rhs_col_t(1:NN_v1, tid)
            end do
            !$OMP END PARALLEL DO

            ! Solve all DOF columns against each volume's operator. Dense: one batched DGETRS reusing
            ! the shared factor. Iterative: a warm-started GMRES sweep (op_solve_multi_iter) so the
            ! spectrally similar columns converge cheaply after the first — this is the serial phase
            ! between the parallel RHS build and unpack, so pws(0) is free to use.
            if (nb > 0) then
              if (oBI(vvol)%backend == BK_ITER) then
                call lregion(vvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
                call op_solve_multi_iter(oBI(vvol), mu(vvol), Lcoord_v, -1, max_nadof, rhs_v, nb, &
                    pws(0), lm_jac, ngmresrestart, nitergmres, epsgmres_jac, its_j, ierr_j)
              else
                call op_solve_multi(oBI(vvol), NN_v, max_nadof, rhs_v, nb, idgetrs)
              end if
              if (oBI(vvol + 1)%backend == BK_ITER) then
                call lregion(vvol + 1, Igeometry, Nvol, Lcoord_v1, Lplasmaregion, Lvacuumregion)
                call op_solve_multi_iter(oBI(vvol + 1), mu(vvol + 1), Lcoord_v1, -1, max_nadof, rhs_v1, nb, &
                    pws(0), lm_jac, ngmresrestart, nitergmres, epsgmres_jac, its_j, ierr_j)
              else
                call op_solve_multi(oBI(vvol + 1), NN_v1, max_nadof, rhs_v1, nb, idgetrs)
              end if
            end if

            ! Loop 3: unpack each batched solution and consume it. Parallel over DOFs: each thread
            ! unpacks into its own storage slot aslot=-(tid+1) (allocated in preset_mod) so DOFs do not
            ! collide; the outputs are per-DOF-indexed (dmupfdx by idof, *_vec by idof_total). The
            ! running idof_total is derived deterministically from ic (loop-carried dep) and the shared
            ! counter advanced once after the loop.
            idof_total_base = idof_total
            !$OMP PARALLEL DO IF(nthreads > 1) SCHEDULE(dynamic) DEFAULT(shared) &
            !$OMP& PRIVATE(ic, tid, aslot, idof, ii, lvol, innout, idof_total, Lcoord_v, Lcoord_v1)
            do ic = 1, nb
              tid = omp_get_thread_num()
              aslot = -(tid + 1) ! this thread's private perturbed storage slot in Ate/Aze/Ato/Azo

              ! Unpacking of the batched solutions
              idof = dof_idof(ic)
              ii = dof_ii(ic)
              dBdX%vol = vvol
              dBdX%ii = ii
              dBdX%irz = dof_irz(ic)
              dBdX%issym = dof_issym(ic)
              dBdX%L = .true.

              dBdX%innout = 1
              call lregion(vvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
              call unpack_perturbed_solution(vvol, NN_v, rhs_v(1:NN_v, ic), aslot)

              dBdX%innout = 0
              call lregion(vvol + 1, Igeometry, Nvol, Lcoord_v1, Lplasmaregion, Lvacuumregion)
              call unpack_perturbed_solution(vvol + 1, NN_v1, rhs_v1(1:NN_v1, ic), aslot)

              ! Helicity multiplier and poloidal flux derivatives (single rank => inner volume).
              call evaluate_dmupfdx(1, idof, ii, Nt, Nz, aslot, Lcoord_v1)

              do lvol = vvol, vvol + 1
                if (lvol == vvol) then
                  innout = 1
                else
                  innout = 0
                end if
                dBdX%innout = innout
                ! Deterministic idof_total: 2 entries per ic (one per lvol), starting from the base.
                idof_total = idof_total_base + 2 * (ic - 1) + (lvol - vvol) + 1
                call lregion(lvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
                call calc_vecpot_deriv(lvol, idof, idof_total, Lcoord_v, aslot)
              end do
            end do
            !$OMP END PARALLEL DO
            idof_total = idof_total_base + 2 * nb

            cycle ! interface done; skip the per-DOF multi-rank path below
          end if

          do ii = 1, mn ! loop over deformations in Fourier harmonics; inside do vvol
            dBdX%ii = ii ! controls construction of derivatives in subroutines called below

            do irz = 0, 1 ! loop over deformations in R and Z; inside do vvol; inside do ii
              if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z
              dBdX%irz = irz ! controls construction of derivatives

              do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms
                if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
                if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
                if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}
                dBdX%issym = issym ! controls construction of derivatives

                idof = idof + 1 ! this labels the degree-of-freedom that the derivative is taken with respect to; this is outside do innout

                if (im(ii) > mpol_fjac .or. abs(in(ii)) > ntor_fjac) cycle

                do lvol = vvol, vvol + 1

                  ! Only interested in one interface perturbation
                  if (ncpu > 1) then
                    if (LinnerVolume .and. (lvol == vvol + 1) ) cycle
                    if (.not.LinnerVolume .and. (lvol == vvol) ) cycle
                  end if

                  ! Set up perturbation information
                  if (lvol == vvol) innout = 1 ! Perturb w.r.t outer interface
                  if (lvol == vvol + 1) innout = 0 ! Perturb w.r.t inner interface

                  dBdX%innout = innout
                  dBdX%L = .true.

                  ! Set up volume information
                  call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.
                  NN = NAdof(lvol)

                  call get_perturbed_solution(lvol, oBI(lvol), NN, Nt, Nz)

                  Lsavedguvij = .false.
                end do

                ! Now volumes neighbouring the interface share perturbed solution
                cpu_id = WhichCpuID(vvol, ncpu)
                cpu_id1 = WhichCpuID(vvol + 1, ncpu)

                ! TODO IMPROVE MPI COMMUNICATIONS. Could directly send dAt and dAz?
                ! Gather everything in inner volume
                if (ncpu > 1) then
                  if (LinnerVolume) then
                    do jj = 1, mn
                      tag = 1 ! Tags for MPI communications
                      tag2 = 2

                      call MPI_RECV(Ate(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION, cpu_id1,&
                          & tag, MPI_COMM_SPEC, stat, ierr)
                      call MPI_RECV(Aze(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION, cpu_id1,&
                          & tag2, MPI_COMM_SPEC, stat, ierr)
                    end do

                    ! Non-stellarator symmetric terms
                    if (.not.enforce_stell_sym) then
                      do jj = 1, mn
                        tag = 3
                        tag2 = 4

                        call MPI_RECV(Ato(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION, &
                            cpu_id1, tag, MPI_COMM_SPEC, stat, ierr)
                        call MPI_RECV(Azo(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION, &
                            cpu_id1, tag2, MPI_COMM_SPEC, stat, ierr)
                      end do
                    end if

                  else
                    do jj = 1, mn
                      tag = 1
                      tag2 = 2

                      call MPI_SEND(Ate(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION, cpu_id, &
                          tag, MPI_COMM_SPEC, ierr)
                      call MPI_SEND(Aze(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION, cpu_id, &
                          tag2, MPI_COMM_SPEC, ierr)
                    end do

                    if (.not.enforce_stell_sym) then
                      do jj = 1, mn
                        tag = 3
                        tag2 = 4

                        call MPI_SEND(Ato(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION, cpu_id&
                            , tag, MPI_COMM_SPEC, ierr)
                        call MPI_SEND(Azo(vvol + 1, -1, jj)%s(0:Lrad(vvol + 1)), Lrad(vvol + 1) + 1, MPI_DOUBLE_PRECISION, cpu_id&
                            , tag2, MPI_COMM_SPEC, ierr)
                      end do
                    end if
                  end if
                end if

                ! At this point, the inverted original Beltrami matrix oBI and the perturbed solution for each volume
                ! neighboring the perturbed interface in inner volume cpu. We now compute the derivatives of mu and
                ! psip w.r.t the position and dBB.
                if (LinnerVolume) then
                  ! Helicity multiplier and poloidal flux derivatives
                  call evaluate_dmupfdx(1, idof, ii, Nt, Nz, -1, Lcoordinatesingularity)
                else
                  ! Only inner volume computes the derivatives in all volumes (it will be broadcasted later)
                  dmupfdx(1:Mvol, vvol, 1:2, idof, 1) = 0.0_rkind
                end if

                do lvol = vvol, vvol + 1

                  call IsMyVolume(lvol)

                  if (IsMyVolumeValue == 0) then
                    cycle
                  else if (IsMyVolumeValue == -1) then
                    call fatal(.true., __FILE__, "Unassociated volume")
                  end if

                  if (lvol == vvol) then
                    innout = 1
                  else
                    innout = 0
                  end if
                  dBdX%innout = innout

                  idof_total = idof_total + 1

                  call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.

                  call calc_vecpot_deriv(lvol, idof, idof_total, Lcoordinatesingularity, -1)
                end do ! matches do lvol = vvol, vvol+1
              end do ! matches do issym
            end do ! matches do irz
          end do ! matches do ii
        end do ! matches do vvol
      end do ! matches do even_or_odd

      deallocate(dof_idof, dof_ii, dof_irz, dof_issym, rhs_v, rhs_v1, rhs_col)

      do tid = 0, nthreads - 1
        call dealloc_perturbed_workspace(pws(tid))
      end do
      deallocate(pws, rhs_col_t)
      call dealloc_matvec_multipliers(lm_jac)

      do vvol = 1, Mvol
        cpu_id = WhichCpuID(vvol, ncpu)

        if (vvol /= Mvol) then
          call MPI_BCAST(dmupfdx(1:Mvol, vvol, 1:2, 1:LGdof, 1), Mvol * 2 * LGdof, MPI_DOUBLE_PRECISION, cpu_id, MPI_COMM_SPEC, &
              ierr)
        end if
      end do

      do vvol = 1, Mvol
        call op_release(oBI(vvol))
      end do
    end if

  end subroutine solve_field_deriv

  subroutine solve_field_deriv_dir(position, ndirs, dirs, jvp, nt_dbb, nz_dbb, recompute_axis)

    ! Jacobian-vector products of the real-space force residual: jvp(:, k) = J . dirs(:, k),
    ! evaluated with ONE general-direction linearised solve per volume per direction instead of
    ! one per (interface, harmonic) column as in solve_field_deriv. The Beltrami operators are
    ! factorised once and shared by all directions, so batching directions amortises the fixed
    ! cost. The general-direction machinery lives in compute_coordinates/volume (Ldirgen mode);
    ! everything downstream of the displacement fields is linear and shared with the dense build.
    !
    ! Contract (same as solve_field_deriv): solve_field must have been run at this position, so
    ! Ate/Aze slots 0:2 hold the converged solution and its flux derivatives. Single MPI rank
    ! only. The adiabatic-pressure term uses each volume's own general dvolume (the per-DOF build
    ! reads a single module dvolume last set for the perturbed interface's volume); the two agree
    ! when adiabatic = 0.

    real(rkind), intent(in) :: position(0:NGdof)
    integer, intent(in) :: ndirs, nt_dbb, nz_dbb
    real(rkind), intent(in) :: dirs(0:NGdof, 1:ndirs)
    real(rkind), intent(out) :: jvp(1:((Mvol - 1) * nt_dbb * nz_dbb), 1:ndirs)
    logical, intent(in) :: recompute_axis

    integer :: vvol, lvol, ivol, idir, ll, NN, vflag, iocons, id, idgetrs, ideriv, id_lim
    integer :: max_iquad, max_lrad, max_nadof, max_lldof
    integer :: ntz_dbb, bknd, its_j, ierr_j
    real(rkind), allocatable :: dBB_base(:, :), rhs_col(:), rhs_one(:, :)
    real(rkind), allocatable :: dvolume_dir(:), dmupf_dir(:, :)
    type(matvec_multipliers) :: lm_jac
    type(perturbed_rhs_workspace) :: pws0
    type(beltrami_operator) :: oBI(1:Mvol)

    jvp = 0.0_rkind
    force_jvp_status = 0
    if (ncpu > 1) then
      force_jvp_status = -1
      return
    end if
    if (Lconstraint == 2) then
      force_jvp_status = -2
      return
    end if

    ntz_dbb = nt_dbb * nz_dbb
    mpol_fjac = maxval(im(1:mn))
    ntor_fjac = maxval(abs(in(1:mn)))

    if (Lconstraint == 1 .or. Lconstraint == 3) then
      id_lim = 2
    else
      id_lim = 0
    end if

    call sallocate(efmn_vec, 1, mn, -1, id_lim, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(cfmn_vec, 1, mn, -1, id_lim, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(ofmn_vec, 1, mn, -1, id_lim, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(sfmn_vec, 1, mn, -1, id_lim, 0, 1, 1, Mvol, 0.0_rkind)

    call invalidate_dmupfdx_lhs()

    call set_fftw_plan(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), &
      iRbs(1:mn, 0:Mvol), iZbc(1:mn, 0:Mvol), 'U', .true., recompute_axis)

    ! Virtual-casing source fields and the all-DOF (d dVC/dx).a0 batch: built once, serially, from
    ! the converged solution and geometry; every direction's vacuum-volume RHS is then served by
    ! the batch contraction in get_perturbed_rhs (vc_deriv_rhs_add_dir).
    if (lvcvacuum == 1 .and. Lfreebound == 1) call vc_prepare_deriv()

    Ldirgen = .false.
    dBdX%L = .false.

    max_iquad = 1
    max_lrad = 1
    max_nadof = 1
    max_lldof = 1

    allocate(dBB_base(1:Ntz, -1:2))
    dBB_base(1:Ntz, -1:2) = 0.0_rkind

    ! Base per-volume state and cached factors (mirrors solve_field_deriv's global-constraint
    ! preamble, minus the MPI distribution).
    do vvol = 1, Mvol
      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      ll = Lrad(vvol)
      NN = NAdof(vvol)

      vflag = 1
      call volume(vvol, vflag, Nt, Nz, Lcoordinatesingularity)

      do iocons = 0, 1
        if (vvol == 1 .and. iocons == 0) cycle
        if (vvol == Mvol .and. iocons == 1) cycle
        ideriv = 0
        call force_real_lforce(vvol, iocons, ideriv, Nt, Nz, dBB_base(1:Ntz, 0), Lcoordinatesingularity)
      end do

      if (Lcoordinatesingularity) then
        max_lldof = max(max_lldof, (ll - mod(ll, 2)) / 2)
      else
        max_lldof = max(max_lldof, ll)
      end if
      max_iquad = max(max_iquad, Iquad(vvol))
      max_lrad = max(max_lrad, ll)
      max_nadof = max(max_nadof, NN)

      ! Reuse the factorization the preceding solve_field deposited for this volume (moved out of
      ! the cache, matched on mu and NAdof). This skips the dominant fixed cost of a JVP batch --
      ! the metric-integral tensors and the dense LU (or preconditioner bind) per volume. On a
      ! miss (caching off, first call, or an unconverged mu) fall through and build it here.
      if (.not. op_cache_take(vvol, mu(vvol), NN, oBI(vvol))) then
        bknd = op_select_backend(vvol)
        if (bknd == BK_ITER) then
          oBI(vvol)%vvol = vvol
          oBI(vvol)%NN = NN
          call allocate_geometry_matrices(vvol, .false., Ntz)
          Lsavedguvij = .false.
          call volume_integrate_chebyshev(Iquad(vvol), mn, vvol, ll, Nt, Nz, diag_only=.true.)
          call op_bind_iter_sparse(oBI(vvol), NN, Lcoordinatesingularity, mu(vvol))
          call deallocate_geometry_matrices(.false.)
        else
          call op_alloc(oBI(vvol), vvol, NN, .false.)
          call allocate_geometry_matrices(vvol, .true., Ntz)
          call allocate_Beltrami_matrices(vvol, .true.)
          call get_LU_beltrami_matrices(vvol, oBI(vvol), NN, Nt, Nz)
          call deallocate_Beltrami_matrices(.true.)
          call deallocate_geometry_matrices(.true.)
        end if
      end if
    end do

    call nallocate(solution, 1, max_nadof, -1, 2)
    call nallocate(guvijsave, 1, Ntz, 1, 3, 1, 3, 1, max_iquad)
    call intghs_alloc_wk_perturbed(Ntz, mn, max_iquad, max_lrad, Mpol, .not.enforce_stell_sym)
    call nallocate(Tss, 0, max_lldof, 1, mn)
    call nallocate(Dtc, 0, max_lldof, 1, mn)
    call nallocate(Dzc, 0, max_lldof, 1, mn)
    if (.not.enforce_stell_sym) then
      call nallocate(Tsc, 0, max_lldof, 1, mn)
      call nallocate(Dts, 0, max_lldof, 1, mn)
      call nallocate(Dzs, 0, max_lldof, 1, mn)
    end if

    call alloc_perturbed_workspace(pws0, Ntz, mn, max_iquad, max_lrad, Mpol, max_lldof)
    call alloc_matvec_multipliers(lm_jac, mn)

    if (Lconstraint == 3) call ensure_dmupfdx_lhs(Nt, Nz)

    allocate(rhs_col(0:max_nadof), rhs_one(1:max_nadof, 1:1))
    allocate(dvolume_dir(1:Mvol), dmupf_dir(1:Mvol, 1:2))

    call sallocate(freal_jac, 1, ntz_dbb, 0, 1, 1, 1, 0, 0, 1, Mvol, 0.0_rkind)
    call sallocate(freal_dBBdmp, 1, ntz_dbb, 1, Mvol, 0, 1, 1, 2, 0.0_rkind)

    do idir = 1, ndirs

      ! ---- one linearised solve per volume, on the internal (Nt, Nz) grid
      call set_fftw_plan(Nt, Nz)
      call alloc_ntz_arrays(Nt, Nz)
      call calc_axis_deriv_grid(Ntz)

      Ldirgen = .true.
      dBdX%L = .true.

      do lvol = 1, Mvol
        call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
        NN = NAdof(lvol)

        call load_direction_coeffs(NGdof, dirs(0:NGdof, idir), lvol, Ntz)
        dBdX%vol = lvol
        Lsavedguvij = .false.

        call get_perturbed_rhs(lvol, NN, Nt, Nz, Lcoordinatesingularity, rhs_col(0:NN), pws0)

        if (oBI(lvol)%backend == BK_ITER) then
          rhs_one(1:NN, 1) = rhs_col(1:NN)
          call op_solve_multi_iter(oBI(lvol), mu(lvol), Lcoordinatesingularity, -1, max_nadof, rhs_one, 1, &
              pws0, lm_jac, ngmresrestart, nitergmres, epsgmres_jac, its_j, ierr_j)
          if (ierr_j /= 0) force_jvp_status(lvol) = max(force_jvp_status(lvol), 1)
          solution(1:NN, -1) = rhs_one(1:NN, 1)
        else
          solution(1:NN, -1) = rhs_col(1:NN)
          call op_solve_multi(oBI(lvol), NN, max_nadof, solution(1:max_nadof, -1:-1), 1, idgetrs)
          if (idgetrs /= 0) then
            force_jvp_status(lvol) = max(force_jvp_status(lvol), 2)
            solution(1:NN, -1) = 0.0_rkind
          end if
        end if

        call unpack_perturbed_solution(lvol, NN, solution(1:NN, -1), -1)

        ! this volume's general dvolume, for the adiabatic pressure term of the rows
        vflag = 1
        call volume(lvol, vflag, Nt, Nz, Lcoordinatesingularity)
        dvolume_dir(lvol) = dvolume
      end do

      ! ---- constraint-multiplier response (direction-contracted dmupfdx)
      dmupf_dir(1:Mvol, 1:2) = 0.0_rkind
      call evaluate_dmupfdx_dir(NGdof, dirs(0:NGdof, idir), Nt, Nz, dmupf_dir)

      ! ---- row responses on the force grid
      call set_fftw_plan(nt_dbb, nz_dbb)
      call alloc_ntz_arrays(nt_dbb, nz_dbb)
      call calc_axis_deriv_grid(ntz_dbb)

      do lvol = 1, Mvol
        call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
        call load_direction_coeffs(NGdof, dirs(0:NGdof, idir), lvol, ntz_dbb)
        dBdX%vol = lvol
        dvolume = dvolume_dir(lvol)
        Lsavedguvij = .false.

        call calc_vecpot_deriv(lvol, 1, lvol, Lcoordinatesingularity, -1)
        call force_real_evaldbb_dir(lvol, lvol, nt_dbb, nz_dbb, Lcoordinatesingularity)
      end do

      ! ---- assemble the rows (the per-DOF local and global assemblies contract to this form)
      do ivol = 1, Mvol - 1
        jvp((ivol - 1) * ntz_dbb + 1:ivol * ntz_dbb, idir) = &
            freal_jac(1:ntz_dbb, 0, 1, 0, ivol + 1) - freal_jac(1:ntz_dbb, 1, 1, 0, ivol)

        if ((LocalConstraint .and. Lconstraint == 1) .or. .not. LocalConstraint) then
          do id = 1, 2
            jvp((ivol - 1) * ntz_dbb + 1:ivol * ntz_dbb, idir) = jvp((ivol - 1) * ntz_dbb + 1:ivol * ntz_dbb, idir) &
                + freal_dBBdmp(1:ntz_dbb, ivol + 1, 0, id) * dmupf_dir(ivol + 1, id) &
                - freal_dBBdmp(1:ntz_dbb, ivol + 0, 1, id) * dmupf_dir(ivol + 0, id)
          end do
        end if
      end do

    end do ! idir

    Ldirgen = .false.
    dBdX%L = .false.

    deallocate(freal_jac)
    deallocate(freal_dBBdmp)
    deallocate(dBB_base, rhs_col, rhs_one, dvolume_dir, dmupf_dir)
    call dealloc_perturbed_workspace(pws0)
    call dealloc_matvec_multipliers(lm_jac)

    do vvol = 1, Mvol
      ! Return each factorization to the cache -- both the ones taken from it and any built here --
      ! so further JVP batches at this geometry skip the build too. op_cache_put is a no-op when
      ! caching is off and leaves the handle empty when on, so the release stays safe either way.
      call op_cache_put(vvol, mu(vvol), oBI(vvol))
      call op_release(oBI(vvol))
    end do

  end subroutine solve_field_deriv_dir

  subroutine force_real_evaldbb_dir(lvol, idof_total, nt_loc, nz_loc, Lcoordinatesingularity)

    ! General-direction counterpart of force_real_evaldbb: the dBB response of volume lvol to the
    ! loaded direction (both boundaries at once, through the Ldirgen metric response and the slot
    ! -1 vector potential). No psifactor here (folded into the direction coefficients) and no
    ! innout index (both boundaries merged); stores into freal_jac(:, iocons, 1, 0, lvol).

    integer, intent(in) :: lvol, idof_total, nt_loc, nz_loc
    logical, intent(in) :: Lcoordinatesingularity

    integer :: iocons, ntz_loc, Lcurvature, id
    real(rkind) :: lss
    real(rkind) :: dBB_id0(1:(nt_loc*nz_loc)), dBB_idm1(1:(nt_loc*nz_loc))
    real(rkind) :: dAt(1:(nt_loc*nz_loc), -1:2), dAz(1:(nt_loc*nz_loc), -1:2)

    ntz_loc = nt_loc * nz_loc

    dAt(1:ntz_loc, -1:2) = 0.0_rkind
    dAz(1:ntz_loc, -1:2) = 0.0_rkind

    do iocons = 0, 1

      if (lvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis); no constraints
      if (lvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary; no constraints

      lss = 2.0_rkind * iocons - 1.0_rkind
      Lcurvature = 1
      call compute_coordinates(lvol, lss, Lcurvature, nt_loc, nz_loc, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)

      call invfft(mn, im, in, efmn_vec(1:mn, 0, iocons, idof_total), ofmn_vec(1:mn, 0, iocons, idof_total), &
        cfmn_vec(1:mn, 0, iocons, idof_total), sfmn_vec(1:mn, 0, iocons, idof_total), dAt(1:ntz_loc, 0), dAz(1:ntz_loc, 0))

      dBB_id0(1:ntz_loc) = 0.5_rkind * (dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 0) &
        - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 0) &
        + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 0)) * inv_jac(1:ntz_loc) ** 2

      if (Lconstraint == 1 .or. Lconstraint == 3) then
        do id = 1, 2
          call invfft(mn, im, in, efmn_vec(1:mn, id, iocons, idof_total), ofmn_vec(1:mn, id, iocons, idof_total), &
          cfmn_vec(1:mn, id, iocons, idof_total), sfmn_vec(1:mn, id, iocons, idof_total), dAt(1:ntz_loc, id), dAz(1:ntz_loc, id))

          freal_dBBdmp(1:ntz_loc, lvol, iocons, id) = 0.5_rkind * (dAz(1:ntz_loc, id) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 0) &
              - 2.0_rkind * dAz(1:ntz_loc, id) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 0) &
              + dAt(1:ntz_loc, id) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 0) &
              + dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, id) * guvij(1:ntz_loc, 2, 2, 0) &
              - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, id) * guvij(1:ntz_loc, 2, 3, 0) &
              + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, id) * guvij(1:ntz_loc, 3, 3, 0)) * inv_jac(1:ntz_loc) ** 2
        end do
      end if

      call invfft(mn, im, in, efmn_vec(1:mn, -1, iocons, idof_total), ofmn_vec(1:mn, -1, iocons, idof_total), &
        cfmn_vec(1:mn, -1, iocons, idof_total), sfmn_vec(1:mn, -1, iocons, idof_total), dAt(1:ntz_loc, -1), dAz(1:ntz_loc, -1))

      Lcurvature = 4
      call compute_coordinates(lvol, lss, Lcurvature, nt_loc, nz_loc, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)

      dBB_idm1(1:ntz_loc) = 0.5_rkind * (dAz(1:ntz_loc, -1) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 0) &
        - 2.0_rkind * dAz(1:ntz_loc, -1) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 0) &
        + dAt(1:ntz_loc, -1) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 0) &
        + dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, -1) * guvij(1:ntz_loc, 2, 2, 0) &
        - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, -1) * guvij(1:ntz_loc, 2, 3, 0) &
        + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, -1) * guvij(1:ntz_loc, 3, 3, 0)) * inv_jac(1:ntz_loc) ** 2 &
        + & ! Now the metric derivatives
        0.5_rkind * (dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 1) &
        - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 1) &
        + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 1)) * inv_jac(1:ntz_loc) ** 2 &
        - dBB_id0(1:ntz_loc) * 2.0_rkind * sg(1:ntz_loc, 1) * inv_jac(1:ntz_loc)

      if (vvolume(lvol) < small) then
        force_jvp_status(lvol) = max(force_jvp_status(lvol), 3)
        freal_jac(1:ntz_loc, iocons, 1, 0, lvol) = dBB_idm1(1:ntz_loc)
      else
        freal_jac(1:ntz_loc, iocons, 1, 0, lvol) = -adiabatic(lvol) * pscale * gamma * dvolume &
            / vvolume(lvol) ** (gamma + 1.0_rkind) + dBB_idm1(1:ntz_loc)
      end if

    end do

  end subroutine force_real_evaldbb_dir

  subroutine calc_b2mag_jump_deriv(force_real_jac, nt_dbb, nz_dbb)
    real(rkind), intent(out) :: force_real_jac(1: (Mvol - 1) * (nt_dbb*nz_dbb), 0:NGdof)! Jacobian matrix to be calculated!       ! dimensions
    integer, intent(in) :: nt_dbb, nz_dbb
    
    logical :: LInnerVolume

    integer :: ntz_dbb, idof_total, ii, irz, issym, ivol, vvol, innout, NN, ll, tdoc_ntz, tdof, &
      even_or_odd, idof, lvol, vol_parity, cpu_id1, id, cpu_id, ierr

    ntz_dbb = nt_dbb * nz_dbb
    
    call sallocate(freal_jac, 1, ntz_dbb, 0, 1, 1, LGdof, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(freal_dBBdmp, 1, ntz_dbb, 1, Mvol, 0, 1, 1, 2, 0.0_rkind)

    idof_total = 0

    call set_fftw_plan(nt_dbb, nz_dbb)
    call alloc_ntz_arrays(nt_dbb, nz_dbb)

    call calc_axis_deriv_grid(ntz_dbb)

    if (LocalConstraint) then

      do vvol = 1, Mvol
        call IsMyVolume(vvol)

        if (IsMyVolumeValue == 0) then
          cycle
        else if (IsMyVolumeValue == -1) then
          call fatal(.true., __FILE__, "Unassociated volume")
        end if

        call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.

        dBdX%vol = vvol ! Label
        ll = Lrad(vvol)! Shorthand
        NN = NAdof(vvol)! shorthand  

        idof = 0 ! labels degree of freedom; local to interface

        do ii = 1, mn ! loop over deformations in Fourier harmonics; inside do vvol
          dBdX%ii = ii ! controls construction of derivatives in subroutines called below
          
          do irz = 0, 1 ! loop over deformations in R and Z; inside do vvol; inside do ii
            if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z
            
            dBdX%irz = irz ! controls construction of derivatives

            do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms
              if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
              if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
              if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

              dBdX%issym = issym ! controls construction of derivatives
              
              idof = idof + 1 ! this labels the degree-of-freedom that the derivative is taken with respect to; this is outside do innout

              do innout = 0, 1 ! loop over deformations to inner and outer interface; inside do vvol; inside do ii; inside do irz

                if (vvol == 1 .and. innout == 0) cycle ! no degrees of freedom at coordinate axis / fixed inner boundary
                if (vvol == Mvol .and. innout == 1) cycle ! no degress of freedom                      fixed outer boundary; for linearized displacement

                dBdX%innout = innout

                idof_total = idof_total + 1

                call force_real_evaldbb(vvol, idof, innout, ii, idof_total, nt_dbb, nz_dbb, Lcoordinatesingularity)

                Lsavedguvij = .false.
              end do ! matches do innout
            end do ! matches do issym
          end do ! matches do irz
        end do ! matches do ii
      end do ! matches do vvol = 1, Mvol
    
    else ! Global constraint

      ! Loop on perturbed interfaces
      do even_or_odd = 0, 1 ! First loop on even interfaces perturbation, then on odd interfaces. This allow efficient parallelization
        do vvol = 1, Mvol - 1 !labels which interface is perturbed
          ! Parallelization - parallelization on perturbed interface and not on volume (outermost loop)
          vol_parity = MODULO(vvol, 2)
          if ( (vol_parity == 0) .and. (even_or_odd == 1) ) cycle ! even_or_odd=1 thus perturb only odd  interfaces
          if ( (vol_parity == 1) .and. (even_or_odd == 0) ) cycle ! even_or_odd=0 thus perturb only even interfaces

          call IsMyVolume(vvol)

          if (IsMyVolumeValue == 0) then ! This CPU does not deal with interface's inner volume

            call IsMyVolume(vvol + 1)

            if (IsMyVolumeValue == 0) then ! This CPU does not deal with interface's outer volume either - cycle
              cycle
            else if (IsMyVolumeValue == -1) then
              call fatal(.true., __FILE__, "Unassociated volume")
            else
              LInnerVolume = .false.
            end if

          else if (IsMyVolumeValue == -1) then
            call fatal(.true., __FILE__, "Unassociated volume")
          else
            LinnerVolume = .true.
          end if

          idof = 0 ! labels degree of freedom of interface vvol

          do ii = 1, mn ! loop over deformations in Fourier harmonics; inside do vvol
            dBdX%ii = ii ! controls construction of derivatives in subroutines called below

            do irz = 0, 1 ! loop over deformations in R and Z; inside do vvol; inside do ii
              if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z
              dBdX%irz = irz ! controls construction of derivatives

              do issym = 0, 1 ! loop over stellarator and non-stellarator symmetric terms
                if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on non-stellarator symmetric harmonics
                if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
                if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}
                dBdX%issym = issym ! controls construction of derivatives
                
                idof = idof + 1 ! this labels the degree-of-freedom that the derivative is taken with respect to; this is outside do innout

                if (im(ii) > mpol_fjac .or. abs(in(ii)) > ntor_fjac) cycle

                do lvol = vvol, vvol + 1
                  call IsMyVolume(lvol)

                  if (IsMyVolumeValue == 0) then
                    cycle
                  else if (IsMyVolumeValue == -1) then
                    call fatal(.true., __FILE__, "Unassociated volume")
                  end if

                  if (lvol == vvol) then
                    innout = 1
                  else
                    innout = 0
                  end if
                  dBdX%innout = innout

                  idof_total = idof_total + 1

                  call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)! assigns Lcoordinatesingularity, Lplasmaregion, etc.

                  call force_real_evaldbb(lvol, idof, innout, ii, idof_total, nt_dbb, nz_dbb, Lcoordinatesingularity)
                end do ! matches do lvol = vvol, vvol+1
              end do ! matches do issym
            end do ! matches do irz
          end do ! matches do ii
        end do ! matches do vvol
      end do ! matches do even_or_odd

      do vvol = 1, Mvol
        cpu_id = WhichCpuID(vvol, ncpu)

        call MPI_BCAST(freal_jac(1:ntz_dbb, 0:1, 1:LGdof, 0:1, vvol), 2 * 2 * LGdof * ntz_dbb, MPI_DOUBLE_PRECISION, cpu_id, &
            MPI_COMM_SPEC, ierr)
        call MPI_BCAST(freal_dBBdmp(1:ntz_dbb, vvol, 0:1, 1:2), 2 * 2 * ntz_dbb, MPI_DOUBLE_PRECISION, cpu_id, MPI_COMM_SPEC, ierr)
      end do
    end if
    
    ! Broadcast information to all CPUs
    do vvol = 1, Mvol
      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      call bcast_freal_jac(vvol, ntz_dbb)
    end do

    force_real_jac(1:ntz_dbb * (Mvol - 1) , 1:NGdof) = 0.0_rkind

    do vvol = 1, Mvol - 1 ! loop over interior surfaces

      if (ImagneticOK(vvol) .and. ImagneticOK(vvol + 1)) then ! the magnetic fields in the volumes adjacent to this interface are valid

        idof = 0 ! labels degree-of-freedom = Fourier harmonic of surface geometry

        do ii = 1, mn ! loop over degrees-of-freedom

          do irz = 0, 1 ! Fourier harmonic of R, Fourier harmonic of Z

            if (irz == 1 .and. Igeometry < 3) cycle ! no dependence on Z

            do issym = 0, 1 ! stellarator symmetry

              if (issym == 1 .and. enforce_stell_sym) cycle ! no dependence on the non-stellarator symmetric harmonics

              if (ii == 1 .and. irz == 1 .and. issym == 0) cycle ! no dependence on Zbs_{m=0,n=0}
              if (ii == 1 .and. irz == 0 .and. issym == 1) cycle ! no dependence on Rbs_{m=0,n=0}

              idof = idof + 1 ! labels degree-of-freedom

              if (LocalConstraint) then
                ! Derivative with respect to previous interface
                if (vvol > 1) then
                  tdof = (vvol - 2) * LGdof + idof ! labels degree-of-freedom in internal interface geometry
                  tdoc_ntz = (vvol - 1) * ntz_dbb

                  force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = -freal_jac(1:ntz_dbb, 1, idof, 0, vvol + 0)
                  if (Lconstraint == 1) then
                    force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) &
                        - freal_dBBdmp(1:ntz_dbb, vvol + 0, 1, 1) * dmupfdx(vvol, 1, 1, idof, 0) &
                        - freal_dBBdmp(1:ntz_dbb, vvol + 0, 1, 2) * dmupfdx(vvol, 1, 2, idof, 0)
                  end if

                end if ! end of if( vvol.gt.1 )

                ! Derivative with respect to current interface
                tdof = (vvol - 1) * LGdof + idof
                tdoc_ntz = (vvol - 1) * ntz_dbb

                force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = freal_jac(1:ntz_dbb, 0, idof, 0, vvol + 1) &
                    - freal_jac(1:ntz_dbb, 1, idof, 1, vvol + 0)

                if (Lconstraint == 1) then
                  force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) &
                      + freal_dBBdmp(1:ntz_dbb, vvol + 1, 0, 1) * dmupfdx(vvol + 1, 1, 1, idof, 0) &
                      + freal_dBBdmp(1:ntz_dbb, vvol + 1, 0, 2) * dmupfdx(vvol + 1, 1, 2, idof, 0) &
                      - freal_dBBdmp(1:ntz_dbb, vvol + 0, 1, 1) * dmupfdx(vvol + 0, 1, 1, idof, 1) &
                      - freal_dBBdmp(1:ntz_dbb, vvol + 0, 1, 2) * dmupfdx(vvol + 0, 1, 2, idof, 1)
                end if

                ! Derivative with respect to next interface
                if (vvol < Mvol - 1) then
                  tdof = (vvol + 0) * LGdof + idof
                  tdoc_ntz = (vvol - 1) * ntz_dbb

                  force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = freal_jac(1:ntz_dbb, 0, idof, 1, vvol + 1)

                  if (Lconstraint == 1) then
                    force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) &
                        + freal_dBBdmp(1:ntz_dbb, vvol + 1, 0, 1) * dmupfdx(vvol + 1, 1, 1, idof, 1) &
                        + freal_dBBdmp(1:ntz_dbb, vvol + 1, 0, 2) * dmupfdx(vvol + 1, 1, 2, idof, 1)
                  end if

                end if ! end of if( vvol.lt.Mvol-1 )

              else ! Global constraint
                ! In the general case of global constraint, there are no 0.0_rkind element in the hessian. We thus loop on all volumes

                tdof = (vvol - 1) * LGdof + idof
                do ivol = 1, Mvol - 1
                  tdoc_ntz = (ivol - 1) * ntz_dbb

                  if (ivol == vvol - 1) then
                    force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = freal_jac(1:ntz_dbb, 0, idof, 1, ivol + 1)
                  elseif (ivol == vvol) then
                    force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = freal_jac(1:ntz_dbb, 0, idof, 0, ivol + 1) &
                        - freal_jac(1:ntz_dbb, 1, idof, 1, ivol)
                  elseif (ivol == vvol + 1) then
                    force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = -freal_jac(1:ntz_dbb, 1, idof, 0, ivol)
                  end if

                  force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) = force_real_jac(tdoc_ntz + 1:tdoc_ntz + ntz_dbb, tdof) &
                      + freal_dBBdmp(1:ntz_dbb, ivol + 1, 0, 1) * dmupfdx(ivol + 1, vvol, 1, idof, 1) &
                      + freal_dBBdmp(1:ntz_dbb, ivol + 1, 0, 2) * dmupfdx(ivol + 1, vvol, 2, idof, 1) &
                      - freal_dBBdmp(1:ntz_dbb, ivol + 0, 1, 1) * dmupfdx(ivol + 0, vvol, 1, idof, 1) &
                      - freal_dBBdmp(1:ntz_dbb, ivol + 0, 1, 2) * dmupfdx(ivol + 0, vvol, 2, idof, 1)
                end do
              end if ! matches if( LocalConstraint )
            end do ! matches do issym
          end do ! matches do irz
        end do ! matches do ii

      else ! matches if( ImagneticOK(vvol) .and. ImagneticOK(vvol+1) )
        call fatal(.true., __FILE__, "Need to provide suitable values for hessian in case of field failure")
      end if ! end of if( ImagneticOK(vvol) .and. ImagneticOK(vvol+1) )
    end do ! end of do vvol

    deallocate(freal_jac)
    deallocate(freal_dBBdmp)
  end subroutine calc_b2mag_jump_deriv

  subroutine alloc_ntz_arrays(nt_loc, nz_loc)
    integer, intent(in) :: nt_loc, nz_loc
    integer :: ntz_loc, kk, jj, ii, mi, ni, jk
    real(rkind) :: teta, zeta, arg

    ! Already allocated arrays for these dimensions, do not repeat work
    if(nt_loc == nt_alloc .and. nz_loc == nz_alloc) return 

    ntz_loc = nt_loc * nz_loc

    nt_alloc = nt_loc
    nz_alloc = nz_loc

    call sallocate(Rij, 1, ntz_loc, 0, 3, 0, 3, 0.0_rkind)
    call sallocate(Zij, 1, ntz_loc, 0, 3, 0, 3, 0.0_rkind)
    call sallocate(sg, 1, ntz_loc, 0, 3, 0.0_rkind)
    call sallocate(inv_jac, 1, ntz_loc, 0.0_rkind)
    call sallocate(guvij, 1, ntz_loc, 0, 3, 0, 3, -1, 3, 0.0_rkind)

    call sallocate(cosi, 1, ntz_loc, 1, mn, 0.0_rkind)
    call sallocate(sini, 1, ntz_loc, 1, mn, 0.0_rkind)

    do ii = 1, mn
      mi = im(ii)
      ni = in(ii)! loop over Fourier harmonics

      do kk = 0, nz_loc - 1
        zeta = kk * pi2nfp / nz_loc
        do jj = 0, nt_loc - 1
          teta = jj * pi2 / nt_loc
          jk = 1 + jj + kk * nt_loc
          arg = mi * teta - ni * zeta
          cosi(jk, ii) = cos(arg)
          sini(jk, ii) = sin(arg)
        end do
      end do

    end do

    call sallocate(ijreal, 1, ntz_loc, 0.0_rkind)
    call sallocate(ijimag, 1, ntz_loc, 0.0_rkind)
    call sallocate(jireal, 1, ntz_loc, 0.0_rkind)
    call sallocate(jiimag, 1, ntz_loc, 0.0_rkind)
    call sallocate(jkreal, 1, ntz_loc, 0.0_rkind)
    call sallocate(jkimag, 1, ntz_loc, 0.0_rkind)
    call sallocate(kjreal, 1, ntz_loc, 0.0_rkind)
    call sallocate(kjimag, 1, ntz_loc, 0.0_rkind)

    if (Igeometry == 3) then
      call sallocate(dRodR, 1, ntz_loc, 0, 3, 1, mn, 0.0_rkind)
      call sallocate(dRodZ, 1, ntz_loc, 0, 3, 1, mn, 0.0_rkind)
      call sallocate(dZodR, 1, ntz_loc, 0, 3, 1, mn, 0.0_rkind)
      call sallocate(dZodZ, 1, ntz_loc, 0, 3, 1, mn, 0.0_rkind)
    end if
    
  end subroutine alloc_ntz_arrays

  subroutine dealloc_ntz_arrays()

    deallocate(Rij)
    deallocate(Zij)
    deallocate(sg)
    deallocate(inv_jac)
    deallocate(guvij)

    deallocate(cosi)
    deallocate(sini)

    deallocate(ijreal)
    deallocate(ijimag)
    deallocate(jireal)
    deallocate(jiimag)
    deallocate(jkreal)
    deallocate(jkimag)
    deallocate(kjreal)
    deallocate(kjimag)

    if (Igeometry == 3) then
      deallocate(dRodR)
      deallocate(dRodZ)
      deallocate(dZodR)
      deallocate(dZodZ)
    end if
  
  end subroutine dealloc_ntz_arrays

  subroutine calc_vecpot_deriv(lvol, idof, idof_total, Lcoordinatesingularity, aslot)
    integer, intent(in) :: lvol, idof, idof_total
    logical, intent(in) :: Lcoordinatesingularity
    integer, intent(in) :: aslot ! perturbed vector-potential slot for the id=-1 read; -1 when serial

    integer :: id, id_lim, iocons, read_slot
    ! thread-local harmonics filled by build_vector_potential, copied into the per-DOF *_vec arrays
    real(rkind) :: lefmn(1:mn), lofmn(1:mn), lcfmn(1:mn), lsfmn(1:mn)

    if (Lconstraint == 1 .or. Lconstraint == 3) then
      id_lim = 2 ! evaluate B^2 and derivatives w.r.t interface geometry, mu, and pflux
    else
      id_lim = 0 ! evaluate B^2 and derivative wrt interface geometry
    end if

    do iocons = 0, 1
      if (lvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis)
      if (lvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary

      do id = -1, id_lim
        read_slot = merge(aslot, id, id == -1) ! -1 is the per-DOF perturbed slot; 0..2 are shared
        call build_vector_potential(lvol, iocons, read_slot, Lcoordinatesingularity, lefmn(1:mn), lofmn(1:mn), lcfmn(1:mn), &
            lsfmn(1:mn))

        efmn_vec(1:mn, id, iocons, idof_total) = lefmn(1:mn)
        cfmn_vec(1:mn, id, iocons, idof_total) = lcfmn(1:mn)
        if (.not.enforce_stell_sym) then
          sfmn_vec(1:mn, id, iocons, idof_total) = lsfmn(1:mn)
          ofmn_vec(1:mn, id, iocons, idof_total) = lofmn(1:mn)
        end if
      end do
    end do

  end subroutine calc_vecpot_deriv

  subroutine force_real_evaldbb(lvol, idof, innout, ii, idof_total, nt_loc, nz_loc, Lcoordinatesingularity)
    integer :: iocons, lvol, innout, ii, idof, nt_loc, nz_loc, idof_total
    logical, intent(in) :: Lcoordinatesingularity
    real(rkind) :: lss
    real(rkind) :: dBB_id0(1:(nt_loc*nz_loc)), dBB_idm1(1:(nt_loc*nz_loc))
    real(rkind) :: dAt(1:(nt_loc*nz_loc), -1:2), dAz(1:(nt_loc*nz_loc), -1:2)

    integer :: ntz_loc, Lcurvature, id

    call fatal(lvol - 1 + innout > Mvol, __FILE__, "psifactor needs attention")

    ntz_loc = nt_loc * nz_loc

    dAt(1:ntz_loc, -1:2) = 0.0_rkind ! initialize intent out; 01 Jul 14
    dAz(1:ntz_loc, -1:2) = 0.0_rkind ! initialize intent out; 01 Jul 14

    do iocons = 0, 1

      if (lvol == 1 .and. iocons == 0) cycle ! fixed inner boundary (or coordinate axis); no constraints
      if (lvol == Mvol .and. iocons == 1) cycle ! fixed outer boundary                     ; no constraints

      lss = 2.0_rkind * iocons - 1.0_rkind ! recall that iocons is effective local radial coordinate; 24 Apr 13
      Lcurvature = 1
      call compute_coordinates(lvol, lss, Lcurvature, nt_loc, nz_loc, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)! get coordinates and derivatives wrt Rj, Zj, at specific radial location

      call invfft(mn, im, in, efmn_vec(1:mn, 0, iocons, idof_total), ofmn_vec(1:mn, 0, iocons, idof_total), &
        cfmn_vec(1:mn, 0, iocons, idof_total), sfmn_vec(1:mn, 0, iocons, idof_total), dAt(1:ntz_loc, 0), dAz(1:ntz_loc, 0))

      dBB_id0(1:ntz_loc) = 0.5_rkind * (dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 0) &
        - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 0) &
        + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 0)) * inv_jac(1:ntz_loc) ** 2

      if (Lconstraint == 1 .or. Lconstraint == 3) then
        do id = 1, 2
          call invfft(mn, im, in, efmn_vec(1:mn, id, iocons, idof_total), ofmn_vec(1:mn, id, iocons, idof_total), &
          cfmn_vec(1:mn, id, iocons, idof_total), sfmn_vec(1:mn, id, iocons, idof_total), dAt(1:ntz_loc, id), dAz(1:ntz_loc, id))
  
          freal_dBBdmp(1:ntz_loc, lvol, iocons, id) = 0.5_rkind * (dAz(1:ntz_loc, id) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 0) &
              - 2.0_rkind * dAz(1:ntz_loc, id) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 0) &
              + dAt(1:ntz_loc, id) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 0) &
              + dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, id) * guvij(1:ntz_loc, 2, 2, 0) &
              - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, id) * guvij(1:ntz_loc, 2, 3, 0) &
              + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, id) * guvij(1:ntz_loc, 3, 3, 0)) * inv_jac(1:ntz_loc) ** 2
        end do
      end if
      
      call invfft(mn, im, in, efmn_vec(1:mn, -1, iocons, idof_total), ofmn_vec(1:mn, -1, iocons, idof_total), &
        cfmn_vec(1:mn, -1, iocons, idof_total), sfmn_vec(1:mn, -1, iocons, idof_total), dAt(1:ntz_loc, -1), dAz(1:ntz_loc, -1))
      
      ! If derivatives w.r.t geometry, take into account metric derivatives and their derivatives wrt Rj, Zj
      Lcurvature = 4
      call compute_coordinates(lvol, lss, Lcurvature, nt_loc, nz_loc, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)

      dBB_idm1(1:ntz_loc) = 0.5_rkind * (dAz(1:ntz_loc, -1) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 0) &
        - 2.0_rkind * dAz(1:ntz_loc, -1) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 0) &
        + dAt(1:ntz_loc, -1) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 0) &
        + dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, -1) * guvij(1:ntz_loc, 2, 2, 0) &
        - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, -1) * guvij(1:ntz_loc, 2, 3, 0) &
        + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, -1) * guvij(1:ntz_loc, 3, 3, 0)) * inv_jac(1:ntz_loc) ** 2 &
        + & ! Now the metric derivatives
        0.5_rkind * (dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 1) &
        - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 1) &
        + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 1)) * inv_jac(1:ntz_loc) ** 2 &
        - dBB_id0(1:ntz_loc) * 2.0_rkind * sg(1:ntz_loc, 1) * inv_jac(1:ntz_loc)
    
      ! Add derivatives of pressure as well
      call fatal(vvolume(lvol) < small, __FILE__, "shall divide by vvolume(lvol)**(gamma + 1.0_rkind)")

      ! Derivatives of force wrt geometry; In real(rkind) space.
      freal_jac(1:ntz_loc, iocons, idof, innout, lvol) = (-adiabatic(lvol) * pscale * gamma * dvolume &
          / vvolume(lvol) ** (gamma + 1.0_rkind) &
          + dBB_idm1(1:ntz_loc) ) * psifactor(ii, lvol - 1 + innout)
      
    end do

  end subroutine force_real_evaldbb

  subroutine force_real_lforce(lvol, iocons, ideriv, nt_loc, nz_loc, dBB, Lcoordinatesingularity)
    integer, intent(in) :: lvol, iocons, ideriv, nt_loc, nz_loc
    real(rkind) :: dAt(1:(nt_loc*nz_loc), -1:2), dAz(1:(nt_loc*nz_loc), -1:2)
    logical, intent(in) :: Lcoordinatesingularity

    integer :: Lcurvature, ii, jj, kk, ll, ifail, ivol, lnn, mi, id, ntz_loc
    real(rkind) :: dBB(1:(nt_loc*nz_loc), -1:2), lss

    ntz_loc = nt_loc * nz_loc

    dAt(1:ntz_loc, -1:2) = 0.0_rkind ! initialize intent out; 01 Jul 14
    dAz(1:ntz_loc, -1:2) = 0.0_rkind ! initialize intent out; 01 Jul 14

    lss = 2.0_rkind * iocons - 1.0_rkind ! recall that iocons is effective local radial coordinate; 24 Apr 13
    Lcurvature = 1
    call compute_coordinates(lvol, lss, Lcurvature, nt_loc, nz_loc, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)! get coordinates and derivatives wrt Rj, Zj, at specific radial location

    ! Compute covariant vector potential. Stored in efmn, ofmn, cfmn, sfmn.
    call build_vector_potential(lvol, iocons, 0, Lcoordinatesingularity, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn))

    ! Map to real(rkind) space
    call invfft(mn, im, in, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), dAt(1:ntz_loc, 0), dAz(1:ntz_loc, 0))

    id = ideriv
    if (id == 0) then

      dBB(1:ntz_loc, id) = 0.5_rkind * (dAz(1:ntz_loc, id) * dAz(1:ntz_loc, id) * guvij(1:ntz_loc, 2, 2, id) &
          - 2.0_rkind * dAz(1:ntz_loc, id) * dAt(1:ntz_loc, id) * guvij(1:ntz_loc, 2, 3, id) &
          + dAt(1:ntz_loc, id) * dAt(1:ntz_loc, id) * guvij(1:ntz_loc, 3, 3, id)) * inv_jac(1:ntz_loc) ** 2
    else
      ! Compute covariant vector potential. Stored in efmn, ofmn, cfmn, sfmn.
      call build_vector_potential(lvol, iocons, id, Lcoordinatesingularity, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn))

      ! Map to real(rkind) space
      call invfft(mn, im, in, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), dAt(1:ntz_loc, id), dAz(1:ntz_loc, id))

      dBB(1:ntz_loc, id) = 0.5_rkind * (dAz(1:ntz_loc, id) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 0) &
          - 2.0_rkind * dAz(1:ntz_loc, id) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 0) &
          + dAt(1:ntz_loc, id) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 0) &
          + dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, id) * guvij(1:ntz_loc, 2, 2, 0) &
          - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, id) * guvij(1:ntz_loc, 2, 3, 0) &
          + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, id) * guvij(1:ntz_loc, 3, 3, 0)) * inv_jac(1:ntz_loc) ** 2

    end if ! end of if( ideriv.gt.0 )

    ! If derivatives w.r.t geometry, take into account metric derivatives
    if (ideriv == -1) then
      ! Get coordinate metrics and their derivatives wrt Rj, Zj on interface
      lss = 2.0_rkind * iocons - 1.0_rkind
      Lcurvature = 4
      call compute_coordinates(lvol, lss, Lcurvature, nt_loc, nz_loc, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)

      dBB(1:ntz_loc, id) = dBB(1:ntz_loc, id) + &
          0.5_rkind * (dAz(1:ntz_loc, 0) * dAz(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 2, 1) &
          - 2.0_rkind * dAz(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 2, 3, 1) &
          + dAt(1:ntz_loc, 0) * dAt(1:ntz_loc, 0) * guvij(1:ntz_loc, 3, 3, 1)) * inv_jac(1:ntz_loc) ** 2 &
          - dBB(1:ntz_loc, 0) * 2.0_rkind * sg(1:ntz_loc, 1) * inv_jac(1:ntz_loc)
    end if
  end subroutine force_real_lforce

  subroutine bcast_freal_jac(lvol, ntz_loc)
    integer, intent(in) :: lvol, ntz_loc

    integer :: llmodnp, io, iRZl, ii, Nbc

    call fatal(lvol <= 0 .or. lvol > Mvol, __FILE__, "error")

    llmodnp = modulo(lvol - 1, ncpu)! identify which node contains data; this must be consistent with previous looping / parallelization

    if (LocalConstraint) then
      Nbc = ntz_loc * 2 * LGdof * 2
      call broadcast(freal_jac(1:ntz_loc, 0:1, 1:LGdof, 0:1, lvol), Nbc, llmodnp)

      Nbc = ntz_loc * 2 * 2
      call broadcast(freal_dBBdmp(1:ntz_loc, lvol, 0:1, 1:2), Nbc, llmodnp)

      Nbc = 2 * LGdof * 2
      call broadcast(dmupfdx(lvol, 1:1, 1:2, 1:LGdof, 0:1), Nbc, llmodnp)      
    end if

    call broadcast(ImagneticOK(lvol), llmodnp)
  end subroutine bcast_freal_jac

end module field_mod
