! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
! SPDX-FileCopyrightText: 2026 Proxima Fusion
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
      force_jac_status
  use constants_mod, only : pi2, mu0
  use errors_mod, only: fatal
  use force_jvp_context_mod, only: force_jvp_scope, jvp_context_close, jvp_context_open, &
      jvp_context_raise_status, jvp_context_reserve, jvp_context_reset, jvp_context_reset_status, &
      jvp_context_set_all_status, jvp_context_status
  use forces_mod, only: dfp100, get_perturbed_solution, evaluate_dmupfdx, get_LU_beltrami_matrices, &
      invalidate_dmupfdx_lhs, ensure_dmupfdx_lhs, get_perturbed_rhs, unpack_perturbed_solution, &
      load_direction_coeffs, evaluate_dmupfdx_dir
  use geometry_mod, only: compute_coordinates
  use intghs_mod, only: intghs_workspace_init, intghs_workspace_destroy, intghs_alloc_wk_perturbed
  use perturbed_workspace_mod, only: perturbed_rhs_workspace, alloc_perturbed_workspace, dealloc_perturbed_workspace, &
      matvec_multipliers, alloc_matvec_multipliers, dealloc_matvec_multipliers
  use input_list_mod, only : Nfp, Igeometry, Nvol, Lrad, tflux, pflux, phiedge, pscale, helicity, &
      adiabatic, gamma, Rbc, Zbs, Rbs, Zbc, Lconstraint, Lfreebound, &
      mu, Isurf, Ivolume, Mpol, enforce_stell_sym, ngmresrestart, nitergmres, epsgmres_jac, Lfreebsolver
  use io_mod, only: stderr
  use memory_allocation_mod, only: sallocate, nallocate
  use memory_mod, only: allocate_Beltrami_matrices, &
      allocate_geometry_matrices, &
      deallocate_Beltrami_matrices, &
      deallocate_geometry_matrices
  use mpi_collectives_mod, only: broadcast
  use fft_backend_mod, only: fft_init
  use numerical_mod, only: invfft
  use packing_mod, only: packxi, packab
  use precisions_mod, only: rkind, small
  use beltrami_operator_mod, only: beltrami_operator, op_alloc, op_release, op_bcast, &
      op_select_backend, op_bind_iter_identity, op_bind_iter_sparse, op_is_iterative, &
      op_solve_derivative_rhs_batch, op_apply_ad, op_dimension, op_prepare_metric, &
      op_solve_iter, op_solve_multi, BK_ITER
  use beltrami_factor_cache_mod, only: factor_cache_acquire, factor_cache_insert
  use metric_mod, only: compute_guvijsave
  use matrices_mod, only: matrix
  use chebyshev_mod, only: volume_integrate_chebyshev
  use volume_mod, only: lregion, volume
  use coordinate_mod, only: calc_axis_deriv_grid
  use vc_matrix_mod, only: vc_prepare_deriv, vc_attach_operator_correction

  implicit none

  private
  public :: solve_field, solve_field_deriv, calc_b2mag_jump, calc_b2mag_jump_deriv, force_real_lforce
  public :: get_force_jvp_status, solve_field_deriv_dir
  public :: alloc_ntz_arrays, dealloc_ntz_arrays

  real(rkind), allocatable :: freal_jac(:, :, :, :, :)
  real(rkind), allocatable :: freal_dBBdmp(:, :, :, :)

  real(rkind), allocatable :: efmn_vec(:, :, :, :), cfmn_vec(:, :, :, :), ofmn_vec(:, :, :, :), sfmn_vec(:, :, :, :)

  integer, public :: mpol_fjac, ntor_fjac, nt_alloc, nz_alloc
  integer, public :: last_deriv_response_count_expected = 0
  integer, public :: last_deriv_response_count_written = 0
  integer, public :: last_dense_jacobian_slices_touched = 0
  integer, public :: last_jacobian_max_chunk_columns = 0
  integer, public :: last_jacobian_max_active_stripes = 0
  integer, public :: last_jacobian_total_iterations = 0
  logical, public :: last_dense_jacobian_assembled = .false.
  real(rkind), public :: last_jacobian_phase_seconds(1:4) = 0.0_rkind

  integer, parameter :: JAC_CHUNK = 512
  integer, parameter :: JAC_ITER_STRIPES = 4

  !> Batch-owned storage for the JVP coordinator. Dense Jacobian storage remains separate.
  type :: force_jvp_batch_workspace
    integer :: max_iquad = 0
    integer :: max_lrad = 0
    integer :: max_nadof = 0
    integer :: max_lldof = 0
    integer :: ntz_force = 0
    real(rkind), allocatable :: rhs_col(:)
    real(rkind), allocatable :: dvolume_dir(:)
    real(rkind), allocatable :: dmupf_dir(:, :)
    ! Direction-independent helicity-constraint response data (Lconstraint==2 only). Per volume:
    ! hel_w = dMD . a0 (the bordered-system border vector), hel_z = (dMA - mu*dMD)^-1 . hel_w,
    ! hel_wz = <hel_w, hel_z> (= dK/dmu up to the constraint-row sign).
    real(rkind), allocatable :: hel_w(:, :)
    real(rkind), allocatable :: hel_z(:, :)
    real(rkind), allocatable :: hel_wz(:)
    type(perturbed_rhs_workspace) :: rhs_workspace
    type(matvec_multipliers) :: multipliers
    type(beltrami_operator), allocatable :: operators(:)
  end type force_jvp_batch_workspace

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

    call fft_init(Nt, Nz)
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

    call fft_init(nt_dbb, nz_dbb)
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

  !> Exact rank-local number of retained (interface DOF, volume) derivative responses.
  integer function count_local_deriv_dofs() result(ndof_loc)
    integer :: vvol, lvol, ii, irz, issym, nkept, nkept_trunc, nio

    nkept = 0
    nkept_trunc = 0
    do ii = 1, mn
      do irz = 0, 1
        if (irz == 1 .and. Igeometry < 3) cycle
        do issym = 0, 1
          if (issym == 1 .and. enforce_stell_sym) cycle
          if (ii == 1 .and. irz == 1 .and. issym == 0) cycle
          if (ii == 1 .and. irz == 0 .and. issym == 1) cycle
          nkept = nkept + 1
          if (im(ii) <= mpol_fjac .and. abs(in(ii)) <= ntor_fjac) then
            nkept_trunc = nkept_trunc + 1
          end if
        end do
      end do
    end do

    ndof_loc = 0
    if (LocalConstraint) then
      do vvol = 1, Mvol
        call IsMyVolume(vvol)
        if (IsMyVolumeValue == 0) cycle
        nio = 2
        if (vvol == 1) nio = nio - 1
        if (vvol == Mvol) nio = nio - 1
        ndof_loc = ndof_loc + nkept * nio
      end do
    else
      do vvol = 1, Mvol - 1
        do lvol = vvol, vvol + 1
          call IsMyVolume(lvol)
          if (IsMyVolumeValue == 0) cycle
          ndof_loc = ndof_loc + nkept_trunc
        end do
      end do
    end if
  end function count_local_deriv_dofs

  !> Solve a prepared chunk through immutable shared factors and per-stripe iterative scratch.
  subroutine solve_columns_striped(op, mu_v, Lcoord, ldr, rhs, ncolumns, pws_arr, lm_arr, &
      nthreads, iterations, status)
    type(beltrami_operator), intent(in) :: op
    real(rkind), intent(in) :: mu_v
    logical, intent(in) :: Lcoord
    integer, intent(in) :: ldr, ncolumns, nthreads
    real(rkind), intent(inout) :: rhs(1:ldr, 1:ncolumns)
    type(perturbed_rhs_workspace), intent(inout) :: pws_arr(0:nthreads - 1)
    type(matvec_multipliers), intent(inout) :: lm_arr(0:nthreads - 1)
    integer, intent(out) :: iterations, status

    integer :: NN, nstripes, columns_per_stripe, stripe, column, first_column, last_column
    integer :: its, ierr, thread_id
    integer :: stripe_iterations(0:JAC_ITER_STRIPES - 1)
    integer :: stripe_status(0:JAC_ITER_STRIPES - 1)
    real(rkind) :: residual
    real(rkind), allocatable :: warm_start(:)

    iterations = 0
    status = 0
    if (ncolumns < 1) return
    NN = op_dimension(op)

    if (.not. op_is_iterative(op)) then
      call op_solve_multi(op, NN, ldr, rhs, ncolumns, ierr)
      if (ierr /= 0) status = 2
      return
    end if

    stripe_iterations = 0
    stripe_status = 0
    nstripes = min(JAC_ITER_STRIPES, ncolumns)
    columns_per_stripe = (ncolumns + nstripes - 1) / nstripes
    last_jacobian_max_active_stripes = max(last_jacobian_max_active_stripes, &
        min(nthreads, nstripes))

    !$OMP PARALLEL DO IF(nthreads > 1) SCHEDULE(static, 1) DEFAULT(shared) &
    !$OMP& PRIVATE(stripe, column, first_column, last_column, its, ierr, thread_id, residual, warm_start)
    do stripe = 0, nstripes - 1
      thread_id = omp_get_thread_num()
      first_column = stripe * columns_per_stripe + 1
      last_column = min(ncolumns, first_column + columns_per_stripe - 1)
      if (first_column <= last_column) then
        call op_prepare_metric(op, Lcoord, pws_arr(thread_id))
        allocate(warm_start(1:NN))
        warm_start = 0.0_rkind
        do column = first_column, last_column
          call op_solve_iter(op, mu_v, Lcoord, -(thread_id + 1), rhs(1:NN, column), &
              warm_start, pws_arr(thread_id), lm_arr(thread_id), ngmresrestart, nitergmres, &
              epsgmres_jac, its, residual, ierr)
          rhs(1:NN, column) = warm_start
          stripe_iterations(stripe) = stripe_iterations(stripe) + its
          if (ierr /= 0) stripe_status(stripe) = ierr
        end do
        deallocate(warm_start)
      end if
    end do
    !$OMP END PARALLEL DO

    iterations = sum(stripe_iterations)
    if (any(stripe_status(0:nstripes - 1) /= 0)) status = 1
  end subroutine solve_columns_striped

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

    integer :: ndof_batch_max, nb, ic, solve_status, NN_v, NN_v1
    integer :: jc, ncb, icc, nad_buf, lvol_own, innout_own, NN_own
    integer, allocatable :: dof_idof(:), dof_ii(:), dof_irz(:), dof_issym(:)
    real(rkind), allocatable :: rhs_v(:, :), rhs_v1(:, :)

    type(perturbed_rhs_workspace), allocatable :: pws(:)
    real(rkind), allocatable :: rhs_col_t(:, :)
    integer :: nthreads, tid
    integer :: aslot, idof_total_base ! per-thread Ate storage slot; base of the idof_total counter
    logical :: Lcoord_v, Lcoord_v1, Lcoord_own ! per-iteration region flags
    integer :: bknd, its_j ! iterative-backend selection + per-interface GMRES bookkeeping
    type(matvec_multipliers), allocatable :: lm_t(:)
    real(rkind) :: phase_mark

    type(beltrami_operator) :: oBI(1:Mvol)

    dBB(1:Ntz, -1:2) = 0.0_rkind

    force_jac_status = 0
    last_jacobian_phase_seconds = 0.0_rkind
    last_jacobian_max_chunk_columns = 0
    last_jacobian_max_active_stripes = 0
    last_jacobian_total_iterations = 0

    mpol_fjac = mpol_rz
    ntor_fjac = ntor_rz

    if (Lconstraint == 1 .or. Lconstraint == 3 .or. (Lconstraint == 2 .and. Lfreebound == 1)) then
      id_lim = 2 ! evaluate B^2 and derivatives wrt interface geometry, mu (or tflux), and pflux
    else
      id_lim = 0 ! evaluate B^2 and derivative wrt interface geometry
    end if

    last_deriv_response_count_expected = count_local_deriv_dofs()
    last_deriv_response_count_written = 0
    ndof_total = max(last_deriv_response_count_expected, 1)

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

    call fft_init(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), &
      iRbs(1:mn, 0:Mvol), iZbc(1:mn, 0:Mvol), 'U', .true., recompute_axis)

    ! Build the virtual-casing derivative batch once from the converged solution and geometry.
    ! The per-column path consumes it through get_perturbed_rhs as a thread-safe lookup.
    if (Lfreebsolver == 1 .and. Lfreebound == 1) call vc_prepare_deriv()

    idof_total = 0

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
        ! helicity constraint, so dimension is NN+1 -> 0:NN). The free-boundary vacuum volume
        ! carries no helicity constraint and stays unbordered.
        call op_alloc(oBI(vvol), vvol, NN, Lconstraint == 2 .and. Lplasmaregion)

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
          call op_bind_iter_identity(oBI(vvol), vvol, NN)
          ! Set the tag on every rank: non-owners consult it to skip the
          ! (dense-only) factor broadcast below; only the owning rank ever solves with it
        else
          call op_alloc(oBI(vvol), vvol, NN, Lconstraint == 2)
        end if

        ! Parallelization
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
          call op_bind_iter_sparse(oBI(vvol), vvol, NN, Lcoordinatesingularity, mu(vvol))
          if (Lfreebound == 1 .and. Lfreebsolver == 1 .and. vvol == Mvol) then
            call vc_attach_operator_correction(oBI(vvol), NN)
          end if
          call deallocate_geometry_matrices(.false.)
          dBdX%L = .true.
        else ! Dense backend
          call allocate_geometry_matrices(vvol, .true., Ntz)
          call allocate_Beltrami_matrices(vvol, .true.)

          call get_LU_beltrami_matrices(vvol, oBI(vvol), NN, Nt, Nz)

          call deallocate_Beltrami_matrices(.true.)
          call deallocate_geometry_matrices(.true.)
        end if
      end do ! end of vvol = 1, Mvol

      ! Broadcast the dense factors to all CPUs. Iterative volumes are skipped: their operator
      ! lives on the owning rank only (op_bcast ships the dense LU, which they do not have), and
      ! the per-DOF loop below only ever solves a volume on the rank that owns it, so no other
      ! rank needs the factorization.
      if (ncpu > 1) then
        do vvol = 1, Mvol
          if (op_is_iterative(oBI(vvol))) cycle
          NN = NAdof(vvol)
          cpu_id = WhichCpuID(vvol, ncpu)
          call op_bcast(oBI(vvol), NN, cpu_id, MPI_COMM_SPEC)
        end do
      end if

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
      nad_buf = maxval(NAdof(1:Mvol))
      allocate(rhs_v(1:nad_buf, 1:min(JAC_CHUNK, ndof_batch_max)))
      allocate(rhs_v1(1:nad_buf, 1:min(JAC_CHUNK, ndof_batch_max)))

      nthreads = omp_get_max_threads()
      allocate(pws(0:nthreads - 1))
      do tid = 0, nthreads - 1
        call alloc_perturbed_workspace(pws(tid), Ntz, mn, max_iquad, max_lrad, mpol, max_lldof)
      end do
      allocate(rhs_col_t(0:max_nadof, 0:nthreads - 1))
      allocate(lm_t(0:nthreads - 1))
      do tid = 0, nthreads - 1
        call alloc_matvec_multipliers(lm_t(tid), mn)
      end do

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

            idof_total_base = idof_total

            do jc = 1, nb, JAC_CHUNK
              ncb = min(JAC_CHUNK, nb - jc + 1)
              last_jacobian_max_chunk_columns = max(last_jacobian_max_chunk_columns, ncb)

              ! Phase 1: build this chunk's RHS columns for both volumes adjoining the interface.
              ! Each OpenMP thread owns pws(tid) and rhs_col_t(:,tid); dBdX and Lsavedguvij are
              ! THREADPRIVATE, and the region flag is returned through the loop-private Lcoord values.
              phase_mark = omp_get_wtime()
              !$OMP PARALLEL DO IF(nthreads > 1) SCHEDULE(dynamic) DEFAULT(shared) &
              !$OMP& PRIVATE(icc, ic, tid, Lcoord_v, Lcoord_v1)
              do icc = 1, ncb
                ic = jc + icc - 1
                tid = omp_get_thread_num()
                dBdX%vol = vvol
                dBdX%ii = dof_ii(ic)
                dBdX%irz = dof_irz(ic)
                dBdX%issym = dof_issym(ic)
                dBdX%L = .true.

                ! Inner volume: perturb its outer interface.
                dBdX%innout = 1
                call lregion(vvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
                Lsavedguvij = .false.
                call get_perturbed_rhs(vvol, NN_v, Nt, Nz, Lcoord_v, rhs_col_t(0:NN_v, tid), pws(tid))
                rhs_v(1:NN_v, icc) = rhs_col_t(1:NN_v, tid)

                ! Outer volume: perturb its inner interface.
                dBdX%innout = 0
                call lregion(vvol + 1, Igeometry, Nvol, Lcoord_v1, Lplasmaregion, Lvacuumregion)
                Lsavedguvij = .false.
                call get_perturbed_rhs(vvol + 1, NN_v1, Nt, Nz, Lcoord_v1, rhs_col_t(0:NN_v1, tid), pws(tid))
                rhs_v1(1:NN_v1, icc) = rhs_col_t(1:NN_v1, tid)
              end do
              !$OMP END PARALLEL DO
              last_jacobian_phase_seconds(1) = last_jacobian_phase_seconds(1) &
                  + omp_get_wtime() - phase_mark

              ! Phase 2: solve the chunk against both unperturbed operators. A dense operator uses
              ! one multi-RHS back-substitution; an iterative operator distributes warm-started
              ! GMRES column sweeps over thread-private stripes.
              phase_mark = omp_get_wtime()
              Lcoord_v = .false.
              if (op_is_iterative(oBI(vvol))) then
                call lregion(vvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
              end if
              call solve_columns_striped(oBI(vvol), mu(vvol), Lcoord_v, nad_buf, &
                  rhs_v(:, 1:ncb), ncb, pws, lm_t, nthreads, its_j, solve_status)
              last_jacobian_total_iterations = last_jacobian_total_iterations + its_j
              if (solve_status /= 0) force_jac_status(vvol) = max(force_jac_status(vvol), solve_status)

              Lcoord_v1 = .false.
              if (op_is_iterative(oBI(vvol + 1))) then
                call lregion(vvol + 1, Igeometry, Nvol, Lcoord_v1, Lplasmaregion, Lvacuumregion)
              end if
              call solve_columns_striped(oBI(vvol + 1), mu(vvol + 1), Lcoord_v1, nad_buf, &
                  rhs_v1(:, 1:ncb), ncb, pws, lm_t, nthreads, its_j, solve_status)
              last_jacobian_total_iterations = last_jacobian_total_iterations + its_j
              if (solve_status /= 0) then
                force_jac_status(vvol + 1) = max(force_jac_status(vvol + 1), solve_status)
              end if
              last_jacobian_phase_seconds(2) = last_jacobian_phase_seconds(2) &
                  + omp_get_wtime() - phase_mark

              ! Phase 3: unpack and consume the solved columns. Each thread writes through its own
              ! aslot=-(tid+1), while the output indices are derived from ic rather than advanced
              ! inside the parallel loop. The shared idof_total is advanced once after all chunks.
              phase_mark = omp_get_wtime()
              !$OMP PARALLEL DO IF(nthreads > 1) SCHEDULE(dynamic) DEFAULT(shared) &
              !$OMP& PRIVATE(icc, ic, tid, aslot, idof, ii, lvol, innout, idof_total, Lcoord_v, Lcoord_v1)
              do icc = 1, ncb
                ic = jc + icc - 1
                tid = omp_get_thread_num()
                aslot = -(tid + 1)
                idof = dof_idof(ic)
                ii = dof_ii(ic)
                dBdX%vol = vvol
                dBdX%ii = ii
                dBdX%irz = dof_irz(ic)
                dBdX%issym = dof_issym(ic)
                dBdX%L = .true.

                dBdX%innout = 1
                call lregion(vvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
                call unpack_perturbed_solution(vvol, NN_v, rhs_v(1:NN_v, icc), aslot)
                dBdX%innout = 0
                call lregion(vvol + 1, Igeometry, Nvol, Lcoord_v1, Lplasmaregion, Lvacuumregion)
                call unpack_perturbed_solution(vvol + 1, NN_v1, rhs_v1(1:NN_v1, icc), aslot)
                call evaluate_dmupfdx(1, idof, ii, Nt, Nz, aslot, Lcoord_v1)

                do lvol = vvol, vvol + 1
                  innout = merge(1, 0, lvol == vvol)
                  dBdX%innout = innout
                  idof_total = idof_total_base + 2 * (ic - 1) + (lvol - vvol) + 1
                  call lregion(lvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
                  call calc_vecpot_deriv(lvol, idof, idof_total, Lcoord_v, aslot)
                end do
              end do
              !$OMP END PARALLEL DO
              last_jacobian_phase_seconds(3) = last_jacobian_phase_seconds(3) &
                  + omp_get_wtime() - phase_mark
            end do
            idof_total = idof_total_base + 2 * nb

            cycle ! interface done; skip the per-DOF multi-rank path below
          end if

          ! Under MPI each neighboring owner builds and solves its own volume columns. The outer
          ! owner sends one solved contiguous slab per bounded chunk to the inner owner, which
          ! alone evaluates the replicated constraint response. All MPI calls remain outside the
          ! OpenMP regions. The bordered Lconstraint==2 path stays on its established algorithm.
          if (ncpu > 1 .and. Lconstraint /= 2) then
            if (LinnerVolume) then
              lvol_own = vvol
              innout_own = 1
            else
              lvol_own = vvol + 1
              innout_own = 0
            end if
            NN_v = NAdof(vvol)
            NN_v1 = NAdof(vvol + 1)
            NN_own = NAdof(lvol_own)
            cpu_id = WhichCpuID(vvol, ncpu)
            cpu_id1 = WhichCpuID(vvol + 1, ncpu)

            ! Pass 1a: enumerate the kept interface DOFs identically on both neighboring owners.
            nb = 0
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

            if (.not. LinnerVolume) then
              dmupfdx(1:Mvol, vvol, 1:2, 1:LGdof, 1) = 0.0_rkind
            end if
            idof_total_base = idof_total

            do jc = 1, nb, JAC_CHUNK
              ncb = min(JAC_CHUNK, nb - jc + 1)
              last_jacobian_max_chunk_columns = max(last_jacobian_max_chunk_columns, ncb)

              ! Phase 1: each owner builds the RHS columns only for its own adjoining volume.
              ! Thread-private pws(tid), rhs_col_t(:,tid), dBdX and Lsavedguvij keep the parallel
              ! column construction independent.
              phase_mark = omp_get_wtime()
              !$OMP PARALLEL DO IF(nthreads > 1) SCHEDULE(dynamic) DEFAULT(shared) &
              !$OMP& PRIVATE(icc, ic, tid, Lcoord_own)
              do icc = 1, ncb
                ic = jc + icc - 1
                tid = omp_get_thread_num()
                dBdX%vol = vvol
                dBdX%ii = dof_ii(ic)
                dBdX%irz = dof_irz(ic)
                dBdX%issym = dof_issym(ic)
                dBdX%L = .true.
                dBdX%innout = innout_own
                call lregion(lvol_own, Igeometry, Nvol, Lcoord_own, Lplasmaregion, Lvacuumregion)
                Lsavedguvij = .false.
                call get_perturbed_rhs(lvol_own, NN_own, Nt, Nz, Lcoord_own, &
                    rhs_col_t(0:NN_own, tid), pws(tid))
                rhs_v(1:NN_own, icc) = rhs_col_t(1:NN_own, tid)
              end do
              !$OMP END PARALLEL DO
              last_jacobian_phase_seconds(1) = last_jacobian_phase_seconds(1) &
                  + omp_get_wtime() - phase_mark

              ! Phase 2: solve the locally owned volume's chunk, dispatching to one dense
              ! multi-RHS solve or to thread-striped iterative solves.
              phase_mark = omp_get_wtime()
              Lcoord_own = .false.
              if (op_is_iterative(oBI(lvol_own))) then
                call lregion(lvol_own, Igeometry, Nvol, Lcoord_own, Lplasmaregion, Lvacuumregion)
              end if
              call solve_columns_striped(oBI(lvol_own), mu(lvol_own), Lcoord_own, nad_buf, &
                  rhs_v(:, 1:ncb), ncb, pws, lm_t, nthreads, its_j, solve_status)
              last_jacobian_total_iterations = last_jacobian_total_iterations + its_j
              if (solve_status /= 0) then
                force_jac_status(lvol_own) = max(force_jac_status(lvol_own), solve_status)
              end if
              last_jacobian_phase_seconds(2) = last_jacobian_phase_seconds(2) &
                  + omp_get_wtime() - phase_mark

              ! Phase 3: send the outer-volume solutions to the inner owner. The inner owner then
              ! evaluates the coupled constraint response; both owners unpack and consume their
              ! own volume response through thread-private aslot storage. Rank-local output indices
              ! are derived from ic, and the shared idof_total advances once after all chunks.
              phase_mark = omp_get_wtime()
              if (LinnerVolume) then
                call MPI_RECV(rhs_v1(:, 1:ncb), nad_buf * ncb, MPI_DOUBLE_PRECISION, &
                    cpu_id1, vvol, MPI_COMM_SPEC, stat, ierr)
              else
                call MPI_SEND(rhs_v(:, 1:ncb), nad_buf * ncb, MPI_DOUBLE_PRECISION, &
                    cpu_id, vvol, MPI_COMM_SPEC, ierr)
              end if

              !$OMP PARALLEL DO IF(nthreads > 1) SCHEDULE(dynamic) DEFAULT(shared) &
              !$OMP& PRIVATE(icc, ic, tid, aslot, idof, ii, idof_total, Lcoord_v, Lcoord_v1, Lcoord_own)
              do icc = 1, ncb
                ic = jc + icc - 1
                tid = omp_get_thread_num()
                aslot = -(tid + 1)
                idof = dof_idof(ic)
                ii = dof_ii(ic)
                dBdX%vol = vvol
                dBdX%ii = ii
                dBdX%irz = dof_irz(ic)
                dBdX%issym = dof_issym(ic)
                dBdX%L = .true.

                if (LinnerVolume) then
                  dBdX%innout = 1
                  call lregion(vvol, Igeometry, Nvol, Lcoord_v, Lplasmaregion, Lvacuumregion)
                  call unpack_perturbed_solution(vvol, NN_v, rhs_v(1:NN_v, icc), aslot)
                  dBdX%innout = 0
                  call lregion(vvol + 1, Igeometry, Nvol, Lcoord_v1, Lplasmaregion, Lvacuumregion)
                  call unpack_perturbed_solution(vvol + 1, NN_v1, rhs_v1(1:NN_v1, icc), aslot)
                  call evaluate_dmupfdx(1, idof, ii, Nt, Nz, aslot, Lcoord_v1)
                else
                  dBdX%innout = 0
                  call lregion(vvol + 1, Igeometry, Nvol, Lcoord_v1, Lplasmaregion, Lvacuumregion)
                  call unpack_perturbed_solution(vvol + 1, NN_v1, rhs_v(1:NN_v1, icc), aslot)
                end if

                dBdX%innout = innout_own
                idof_total = idof_total_base + ic
                call lregion(lvol_own, Igeometry, Nvol, Lcoord_own, Lplasmaregion, Lvacuumregion)
                call calc_vecpot_deriv(lvol_own, idof, idof_total, Lcoord_own, aslot)
              end do
              !$OMP END PARALLEL DO
              last_jacobian_phase_seconds(3) = last_jacobian_phase_seconds(3) &
                  + omp_get_wtime() - phase_mark
            end do
            idof_total = idof_total_base + nb

            cycle
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

      deallocate(dof_idof, dof_ii, dof_irz, dof_issym, rhs_v, rhs_v1)

      do tid = 0, nthreads - 1
        call dealloc_perturbed_workspace(pws(tid))
        call dealloc_matvec_multipliers(lm_t(tid))
      end do
      deallocate(pws, rhs_col_t, lm_t)

      if (ncpu > 1 .and. Lconstraint /= 2) then
        call MPI_ALLREDUCE(MPI_IN_PLACE, last_jacobian_phase_seconds(1:3), 3, &
            MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_SPEC, ierr)
        call MPI_ALLREDUCE(MPI_IN_PLACE, last_jacobian_total_iterations, 1, &
            MPI_INTEGER, MPI_SUM, MPI_COMM_SPEC, ierr)
      end if

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

    last_deriv_response_count_written = idof_total

  end subroutine solve_field_deriv

  subroutine solve_field_deriv_dir(position, ndirs, dirs, jvp, nt_dbb, nz_dbb, recompute_axis)

    ! Jacobian-vector products of the real-space force residual: jvp(:, k) = J . dirs(:, k),
    ! evaluated with ONE general-direction linearised solve per volume per direction instead of
    ! one per (interface, harmonic) column as in solve_field_deriv. The Beltrami operators are
    ! factorised once and shared by all directions, so batching directions amortises the fixed
    ! cost. The general-direction machinery lives in the private JVP context and is consumed by
    ! compute_coordinates/volume. Everything downstream of the displacement fields is linear and
    ! shared with the dense build.
    !
    ! Contract (same as solve_field_deriv): solve_field must have been run at this position, so
    ! Ate/Aze slots 0:2 hold the converged solution and its flux derivatives (on each volume's
    ! owning rank; they are broadcast below). Under MPI the call is collective with identical
    ! position and dirs on every rank: linearised solves and row responses run on volume owners,
    ! then their compact outputs are combined so every rank returns identical products and
    ! statuses. The adiabatic-pressure term uses each volume's own general dvolume (the per-DOF
    ! build reads a single module dvolume last set for the perturbed interface's volume); the two
    ! agree when adiabatic = 0.

    real(rkind), intent(in) :: position(0:NGdof)
    integer, intent(in) :: ndirs, nt_dbb, nz_dbb
    real(rkind), intent(in) :: dirs(0:NGdof, 1:ndirs)
    real(rkind), intent(out) :: jvp(1:((Mvol - 1) * nt_dbb * nz_dbb), 1:ndirs)
    logical, intent(in) :: recompute_axis

    integer :: configuration_status, cpu_id, ideriv, idir, ierr, ii, vvol
    integer :: status_merge(1:Mvol)
    type(force_jvp_batch_workspace) :: workspace
    type(force_jvp_scope) :: context_scope

    call jvp_context_open(context_scope)
    jvp = 0.0_rkind
    call jvp_context_reset_status()
    call validate_force_jvp_configuration(configuration_status)
    if (configuration_status /= 0) then
      call jvp_context_set_all_status(configuration_status)
      return
    end if

    ! Replicated constraint and row assembly needs the converged solution and flux-derivative
    ! slots for every volume. Publish them from each volume owner before preparing the batch.
    if (ncpu > 1) then
      do vvol = 1, Mvol
        cpu_id = WhichCpuID(vvol, ncpu)
        call broadcast(ImagneticOK(vvol), cpu_id)
        do ideriv = 0, 2
          do ii = 1, mn
            call broadcast(Ate(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
            call broadcast(Aze(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
          end do
        end do
        if (.not.enforce_stell_sym) then
          do ideriv = 0, 2
            do ii = 1, mn
              call broadcast(Ato(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
              call broadcast(Azo(vvol, ideriv, ii)%s(0:Lrad(vvol)), Lrad(vvol) + 1, cpu_id)
            end do
          end do
        end if
      end do
    end if ! ncpu > 1: broadcast occurred

    call initialize_force_jvp_batch(position, nt_dbb, nz_dbb, recompute_axis, workspace)
    do idir = 1, ndirs
      call solve_force_jvp_volume_responses(dirs(0:NGdof, idir), workspace)
      call evaluate_force_jvp_constraint_response(dirs(0:NGdof, idir), workspace)
      call assemble_force_jvp_rows(dirs(0:NGdof, idir), nt_dbb, nz_dbb, workspace, jvp(:, idir))
      ! MPI reductions need not produce the same last bits on every rank. Publish the root's
      ! assembled column so the collective public API returns an identical result everywhere.
      if (ncpu > 1) call broadcast(jvp(:, idir), size(jvp, 1), 0)
      call jvp_context_reset()
    end do ! idir

    call release_force_jvp_batch(workspace)

    ! Solver diagnostics are raised only on the volume owners. Merge them so the collective API
    ! returns the same status array on every rank.
    if (ncpu > 1) then
      do vvol = 1, Mvol
        status_merge(vvol) = jvp_context_status(vvol)
      end do
      call MPI_ALLREDUCE(MPI_IN_PLACE, status_merge, Mvol, MPI_INTEGER, MPI_MAX, MPI_COMM_SPEC, ierr)
      do vvol = 1, Mvol
        call jvp_context_raise_status(vvol, status_merge(vvol))
      end do
    end if

    call jvp_context_close(context_scope)
  end subroutine solve_field_deriv_dir

  !> Validate only the established JVP configuration contract.
  subroutine validate_force_jvp_configuration(status)
    integer, intent(out) :: status

    status = 0
  end subroutine validate_force_jvp_configuration

  !> Initialize base geometry, operators, and reusable storage for one direction batch.
  subroutine initialize_force_jvp_batch(position, nt_dbb, nz_dbb, recompute_axis, workspace)
    real(rkind), intent(in) :: position(0:NGdof)
    integer, intent(in) :: nt_dbb, nz_dbb
    logical, intent(in) :: recompute_axis
    type(force_jvp_batch_workspace), intent(inout) :: workspace

    integer :: id_lim, ideriv, iocons, ll, NN, vflag, vvol
    real(rkind), allocatable :: dBB_base(:, :)

    workspace%ntz_force = nt_dbb * nz_dbb
    mpol_fjac = maxval(im(1:mn))
    ntor_fjac = maxval(abs(in(1:mn)))

    if (Lconstraint == 1 .or. Lconstraint == 3 .or. (Lconstraint == 2 .and. Lfreebound == 1)) then
      id_lim = 2 ! the free-boundary vacuum volume's flux response needs the dA/dflux slots
    else
      id_lim = 0
    end if

    call sallocate(efmn_vec, 1, mn, -1, id_lim, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(cfmn_vec, 1, mn, -1, id_lim, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(ofmn_vec, 1, mn, -1, id_lim, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(sfmn_vec, 1, mn, -1, id_lim, 0, 1, 1, Mvol, 0.0_rkind)

    call invalidate_dmupfdx_lhs()
    call fft_init(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)
    call packxi(NGdof, position(0:NGdof), Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), &
        iRbs(1:mn, 0:Mvol), iZbc(1:mn, 0:Mvol), 'U', .true., recompute_axis)

    call jvp_context_reset()
    dBdX%L = .false.

    ! Only the vacuum-volume owner consumes the direct-VC derivative batch.
    if (Lfreebsolver == 1 .and. Lfreebound == 1) then
      if (ncpu > 1) call IsMyVolume(Mvol)
      if (ncpu == 1 .or. IsMyVolumeValue /= 0) call vc_prepare_deriv()
    end if

    workspace%max_iquad = 1
    workspace%max_lrad = 1
    workspace%max_nadof = 1
    workspace%max_lldof = 1
    allocate(workspace%operators(1:Mvol))
    allocate(dBB_base(1:Ntz, -1:2))
    dBB_base(1:Ntz, -1:2) = 0.0_rkind

    do vvol = 1, Mvol
      if (ncpu > 1) then
        call IsMyVolume(vvol)
        if (IsMyVolumeValue == 0) cycle
      end if
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
        workspace%max_lldof = max(workspace%max_lldof, (ll - mod(ll, 2)) / 2)
      else
        workspace%max_lldof = max(workspace%max_lldof, ll)
      end if
      workspace%max_iquad = max(workspace%max_iquad, Iquad(vvol))
      workspace%max_lrad = max(workspace%max_lrad, ll)
      workspace%max_nadof = max(workspace%max_nadof, NN)
      call acquire_force_jvp_operator(vvol, NN, ll, Lcoordinatesingularity, workspace%operators(vvol))
    end do
    deallocate(dBB_base)

    call nallocate(solution, 1, workspace%max_nadof, -1, 2)
    call nallocate(guvijsave, 1, Ntz, 1, 3, 1, 3, 1, workspace%max_iquad)
    call intghs_alloc_wk_perturbed(Ntz, mn, workspace%max_iquad, workspace%max_lrad, &
        Mpol, .not.enforce_stell_sym)
    call nallocate(Tss, 0, workspace%max_lldof, 1, mn)
    call nallocate(Dtc, 0, workspace%max_lldof, 1, mn)
    call nallocate(Dzc, 0, workspace%max_lldof, 1, mn)
    if (.not.enforce_stell_sym) then
      call nallocate(Tsc, 0, workspace%max_lldof, 1, mn)
      call nallocate(Dts, 0, workspace%max_lldof, 1, mn)
      call nallocate(Dzs, 0, workspace%max_lldof, 1, mn)
    end if

    call alloc_perturbed_workspace(workspace%rhs_workspace, Ntz, mn, workspace%max_iquad, &
        workspace%max_lrad, Mpol, workspace%max_lldof)
    call alloc_matvec_multipliers(workspace%multipliers, mn)
    if (Lconstraint == 3) call ensure_dmupfdx_lhs(Nt, Nz)

    allocate(workspace%rhs_col(0:workspace%max_nadof))
    allocate(workspace%dvolume_dir(1:Mvol), workspace%dmupf_dir(1:Mvol, 1:2))
    call sallocate(freal_jac, 1, workspace%ntz_force, 0, 1, 1, 1, 0, 0, 1, Mvol, 0.0_rkind)
    call sallocate(freal_dBBdmp, 1, workspace%ntz_force, 1, Mvol, 0, 1, 1, 2, 0.0_rkind)
    call jvp_context_reserve(mn, max(Nt * Nz, workspace%ntz_force))
    if (Lconstraint == 2) call initialize_helicity_constraint_response(workspace)
  end subroutine initialize_force_jvp_batch

  !> Build the direction-independent helicity-constraint response data for one JVP batch.
  !>
  !> The dense Jacobian path enforces helicity conservation by bordering the Beltrami system
  !> with the dMD.solution row/column (get_LU_beltrami_matrices). The JVP instead solves the
  !> bordered system through its Schur complement over the unbordered operator, which both
  !> backends already expose: per volume, w = dMD . a0 (matrix-free, op_apply_ad; the helicity
  !> matrix has zero Lagrange rows/columns, so the zero-filled multiplier entries of a0 are
  !> exact) and z = (dMA - mu*dMD)^-1 . w. The per-direction solve then closes the constraint
  !> row <w, da> = 0 (helicity conservation, since dMD carries no geometry dependence) with
  !> da = y + dmu*z, dmu = -<w, y>/<w, z>.
  subroutine initialize_helicity_constraint_response(workspace)
    type(force_jvp_batch_workspace), intent(inout) :: workspace

    integer :: vvol, NN, iterations, solve_status
    real(rkind) :: a0(1:workspace%max_nadof), adx(1:workspace%max_nadof)

    call sallocate(workspace%hel_w, 1, workspace%max_nadof, 1, Mvol, 0.0_rkind)
    call sallocate(workspace%hel_z, 1, workspace%max_nadof, 1, Mvol, 0.0_rkind)
    call sallocate(workspace%hel_wz, 1, Mvol, 0.0_rkind)

    do vvol = 1, Mvol
      if (ncpu > 1) then
        call IsMyVolume(vvol)
        if (IsMyVolumeValue == 0) cycle
      end if
      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      if (Lvacuumregion) cycle ! the free-boundary vacuum volume carries no helicity constraint
      NN = NAdof(vvol)

      a0(1:NN) = 0.0_rkind
      call packab('P', vvol, NN, a0(1:NN), 0)
      Lsavedguvij = .false.
      call op_apply_ad(workspace%operators(vvol), Lcoordinatesingularity, -1, a0(1:NN), &
          adx(1:NN), workspace%hel_w(1:NN, vvol), workspace%rhs_workspace, workspace%multipliers)

      workspace%hel_z(1:NN, vvol) = workspace%hel_w(1:NN, vvol)
      call op_solve_derivative_rhs_batch(workspace%operators(vvol), mu(vvol), Lcoordinatesingularity, -1, &
          workspace%max_nadof, workspace%hel_z(1:workspace%max_nadof, vvol:vvol), 1, workspace%rhs_workspace, &
          workspace%multipliers, ngmresrestart, nitergmres, epsgmres_jac, iterations, solve_status)
      if (solve_status /= 0) call jvp_context_raise_status(vvol, solve_status)
      if (solve_status == 2) workspace%hel_z(1:NN, vvol) = 0.0_rkind

      workspace%hel_wz(vvol) = dot_product(workspace%hel_w(1:NN, vvol), workspace%hel_z(1:NN, vvol))
    end do
  end subroutine initialize_helicity_constraint_response

  !> Acquire a matching cached factor or prepare the configured backend for one JVP volume.
  subroutine acquire_force_jvp_operator(vvol, NN, ll, Lcoord, op)
    integer, intent(in) :: vvol, NN, ll
    logical, intent(in) :: Lcoord
    type(beltrami_operator), intent(inout) :: op

    integer :: backend

    if (factor_cache_acquire(vvol, mu(vvol), NN, op)) return

    backend = op_select_backend(vvol)
    if (backend == BK_ITER) then
      call allocate_geometry_matrices(vvol, .false., Ntz)
      Lsavedguvij = .false.
      call volume_integrate_chebyshev(Iquad(vvol), mn, vvol, ll, Nt, Nz, diag_only=.true.)
      call op_bind_iter_sparse(op, vvol, NN, Lcoord, mu(vvol))
      ! A cache hit above already carries the correction with its move-only operator handle.
      if (Lfreebound == 1 .and. Lfreebsolver == 1 .and. vvol == Mvol) then
        call vc_attach_operator_correction(op, NN)
      end if
      call deallocate_geometry_matrices(.false.)
    else
      call op_alloc(op, vvol, NN, .false.)
      call allocate_geometry_matrices(vvol, .true., Ntz)
      call allocate_Beltrami_matrices(vvol, .true.)
      call get_LU_beltrami_matrices(vvol, op, NN, Nt, Nz)
      call deallocate_Beltrami_matrices(.true.)
      call deallocate_geometry_matrices(.true.)
    end if
  end subroutine acquire_force_jvp_operator

  !> Build and solve one directional RHS for every volume on the internal field grid.
  subroutine solve_force_jvp_volume_responses(direction, workspace)
    real(rkind), intent(in) :: direction(0:NGdof)
    type(force_jvp_batch_workspace), intent(inout) :: workspace

    integer :: iterations, lvol, NN, solve_status, vflag
    real(rkind) :: dmu_dir

    call fft_init(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)
    call calc_axis_deriv_grid(Ntz)

    do lvol = 1, Mvol
      if (ncpu > 1) then
        call IsMyVolume(lvol)
        if (IsMyVolumeValue == 0) cycle
      end if
      call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      NN = NAdof(lvol)
      call load_direction_coeffs(NGdof, direction, lvol, Ntz)
      ! Per-volume perturbation descriptor for the RHS build and the volume() derivative below
      ! (the operator solve does not touch it).
      dBdX%L = .true.
      dBdX%vol = lvol
      Lsavedguvij = .false.
      call get_perturbed_rhs(lvol, NN, Nt, Nz, Lcoordinatesingularity, &
          workspace%rhs_col(0:NN), workspace%rhs_workspace)

      solution(1:NN, -1) = workspace%rhs_col(1:NN)
      call op_solve_derivative_rhs_batch(workspace%operators(lvol), mu(lvol), Lcoordinatesingularity, -1, &
          workspace%max_nadof, solution, 1, workspace%rhs_workspace, &
          workspace%multipliers, ngmresrestart, nitergmres, epsgmres_jac, iterations, solve_status)
      if (solve_status /= 0) call jvp_context_raise_status(lvol, solve_status)
      if (solve_status == 2) solution(1:NN, -1) = 0.0_rkind ! hard fail, discard to prevent re-use

      if (Lconstraint == 2 .and. Lplasmaregion) then
        ! Helicity constraint: close the bordered system's Schur complement. The unconstrained
        ! response y (solved above) gains the mu response dmu*z so that <w, da> = 0, i.e. the
        ! perturbed field conserves this volume's helicity. The free-boundary vacuum volume is
        ! unconstrained here; its flux response enters through dmupf_dir instead.
        if (abs(workspace%hel_wz(lvol)) < small) then
          call jvp_context_raise_status(lvol, 4)
        else
          dmu_dir = -dot_product(workspace%hel_w(1:NN, lvol), solution(1:NN, -1)) / workspace%hel_wz(lvol)
          solution(1:NN, -1) = solution(1:NN, -1) + dmu_dir * workspace%hel_z(1:NN, lvol)
        end if
      end if

      call unpack_perturbed_solution(lvol, NN, solution(1:NN, -1), -1)
      vflag = 1
      call volume(lvol, vflag, Nt, Nz, Lcoordinatesingularity)
      workspace%dvolume_dir(lvol) = dvolume
    end do
  end subroutine solve_force_jvp_volume_responses

  !> Evaluate the direction-contracted constraint response through the existing JVP helper.
  subroutine evaluate_force_jvp_constraint_response(direction, workspace)
    real(rkind), intent(in) :: direction(0:NGdof)
    type(force_jvp_batch_workspace), intent(inout) :: workspace

    workspace%dmupf_dir(1:Mvol, 1:2) = 0.0_rkind
    call evaluate_dmupfdx_dir(NGdof, direction, Nt, Nz, workspace%dmupf_dir)
  end subroutine evaluate_force_jvp_constraint_response

  !> Evaluate force-grid row responses and assemble one JVP output column.
  subroutine assemble_force_jvp_rows(direction, nt_dbb, nz_dbb, workspace, jvp_column)
    real(rkind), intent(in) :: direction(0:NGdof)
    integer, intent(in) :: nt_dbb, nz_dbb
    type(force_jvp_batch_workspace), intent(inout) :: workspace
    real(rkind), contiguous, intent(out) :: jvp_column(:)

    integer :: id, ierr, ivol, lvol

    call fft_init(nt_dbb, nz_dbb)
    call alloc_ntz_arrays(nt_dbb, nz_dbb)
    call calc_axis_deriv_grid(workspace%ntz_force)

    ! These arrays are summed across ranks below. Clear non-owned volume surfaces for each new
    ! direction so a previous direction's merged values cannot be counted again.
    freal_jac = 0.0_rkind
    freal_dBBdmp = 0.0_rkind

    do lvol = 1, Mvol
      if (ncpu > 1) then
        call IsMyVolume(lvol)
        if (IsMyVolumeValue == 0) cycle
      end if
      call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      call load_direction_coeffs(NGdof, direction, lvol, workspace%ntz_force)
      dBdX%vol = lvol
      dvolume = workspace%dvolume_dir(lvol)
      Lsavedguvij = .false.
      call calc_vecpot_deriv(lvol, 1, lvol, Lcoordinatesingularity, -1)
      call force_real_evaldbb_dir(lvol, lvol, nt_dbb, nz_dbb, Lcoordinatesingularity)
    end do

    if (ncpu > 1) then
      call MPI_ALLREDUCE(MPI_IN_PLACE, freal_jac, workspace%ntz_force * 2 * Mvol, MPI_DOUBLE_PRECISION, &
          MPI_SUM, MPI_COMM_SPEC, ierr)
      call MPI_ALLREDUCE(MPI_IN_PLACE, freal_dBBdmp, workspace%ntz_force * Mvol * 4, MPI_DOUBLE_PRECISION, &
          MPI_SUM, MPI_COMM_SPEC, ierr)
    end if

    do ivol = 1, Mvol - 1
      jvp_column((ivol - 1) * workspace%ntz_force + 1:ivol * workspace%ntz_force) = &
          freal_jac(1:workspace%ntz_force, 0, 1, 0, ivol + 1) - &
          freal_jac(1:workspace%ntz_force, 1, 1, 0, ivol)
      ! For Lconstraint==2 only the free-boundary vacuum volume has a (flux) response in
      ! dmupf_dir; the plasma volumes' mu response lives inside the constrained solve.
      if ((LocalConstraint .and. (Lconstraint == 1 .or. Lconstraint == 2)) .or. .not. LocalConstraint) then
        do id = 1, 2
          jvp_column((ivol - 1) * workspace%ntz_force + 1:ivol * workspace%ntz_force) = &
              jvp_column((ivol - 1) * workspace%ntz_force + 1:ivol * workspace%ntz_force) &
              + freal_dBBdmp(1:workspace%ntz_force, ivol + 1, 0, id) * workspace%dmupf_dir(ivol + 1, id) &
              - freal_dBBdmp(1:workspace%ntz_force, ivol + 0, 1, id) * workspace%dmupf_dir(ivol + 0, id)
        end do
      end if
    end do
  end subroutine assemble_force_jvp_rows

  !> Return factors to their cache owner and release all batch-owned JVP storage.
  subroutine release_force_jvp_batch(workspace)
    type(force_jvp_batch_workspace), intent(inout) :: workspace

    integer :: vvol

    dBdX%L = .false.
    deallocate(freal_jac, freal_dBBdmp)
    deallocate(workspace%rhs_col, workspace%dvolume_dir, workspace%dmupf_dir)
    if (allocated(workspace%hel_w)) deallocate(workspace%hel_w, workspace%hel_z, workspace%hel_wz)
    call dealloc_perturbed_workspace(workspace%rhs_workspace)
    call dealloc_matvec_multipliers(workspace%multipliers)

    do vvol = 1, Mvol
      if (ncpu > 1) then
        call IsMyVolume(vvol)
        if (IsMyVolumeValue == 0) cycle
      end if
      ! Both acquired and freshly prepared factors return through the same move-only cache seam.
      call factor_cache_insert(vvol, mu(vvol), workspace%operators(vvol))
      call op_release(workspace%operators(vvol))
    end do
    deallocate(workspace%operators)
  end subroutine release_force_jvp_batch

  !> Copy the private JVP-context diagnostics across the stable Python boundary.
  subroutine get_force_jvp_status(values)
    integer, intent(out) :: values(1:Mvol)

    integer :: vvol

    do vvol = 1, Mvol
      values(vvol) = jvp_context_status(vvol)
    end do
  end subroutine get_force_jvp_status

  subroutine force_real_evaldbb_dir(lvol, idof_total, nt_loc, nz_loc, Lcoordinatesingularity)

    ! General-direction counterpart of force_real_evaldbb: the dBB response of volume lvol to the
    ! loaded direction (both boundaries at once, through the JVP-context metric response and slot
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

      if (Lconstraint == 1 .or. Lconstraint == 3 .or. (Lconstraint == 2 .and. Lfreebound == 1)) then
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
        call jvp_context_raise_status(lvol, 3)
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
    real(rkind) :: row_phase_start

    ntz_dbb = nt_dbb * nz_dbb
    row_phase_start = omp_get_wtime()

    last_dense_jacobian_assembled = .false.
    last_dense_jacobian_slices_touched = 0
    if (myid == 0) then
      call sallocate(freal_jac, 1, ntz_dbb, 0, 1, 1, LGdof, 0, 1, 1, Mvol, 0.0_rkind)
      last_dense_jacobian_slices_touched = Mvol
    else
      ! Workers reserve the address range but commit only the volume slices they own. Rank 0
      ! receives those contiguous slices and alone needs the initialized all-volume mirror.
      allocate(freal_jac(1:ntz_dbb, 0:1, 1:LGdof, 0:1, 1:Mvol))
      do vvol = 1, Mvol
        call IsMyVolume(vvol)
        if (IsMyVolumeValue == 0) cycle
        freal_jac(1:ntz_dbb, 0:1, 1:LGdof, 0:1, vvol) = 0.0_rkind
        last_dense_jacobian_slices_touched = last_dense_jacobian_slices_touched + 1
      end do
    end if
    call sallocate(freal_dBBdmp, 1, ntz_dbb, 1, Mvol, 0, 1, 1, 2, 0.0_rkind)

    idof_total = 0

    call fft_init(nt_dbb, nz_dbb)
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

    end if

    ! Gather the large owned force-response slices to rank 0. Only the small constraint
    ! responses remain broadcasts because replicated constraint bookkeeping still consumes them.
    do vvol = 1, Mvol
      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
      call bcast_freal_jac(vvol, ntz_dbb)
    end do

    if (myid /= 0) then
      last_jacobian_phase_seconds(4) = omp_get_wtime() - row_phase_start
      deallocate(freal_jac)
      deallocate(freal_dBBdmp)
      return
    end if

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
                  ! For Lconstraint==2 only the free-boundary vacuum volume has a (flux) response
                  ! in dmupfdx; the plasma volumes' mu response lives inside the bordered solve.
                  if (Lconstraint == 1 .or. Lconstraint == 2) then
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

                if (Lconstraint == 1 .or. Lconstraint == 2) then
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

                  if (Lconstraint == 1 .or. Lconstraint == 2) then
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

    last_dense_jacobian_assembled = .true.
    last_jacobian_phase_seconds(4) = omp_get_wtime() - row_phase_start
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

    if (Lconstraint == 1 .or. Lconstraint == 3 .or. (Lconstraint == 2 .and. Lfreebound == 1)) then
      id_lim = 2 ! evaluate B^2 and derivatives w.r.t interface geometry, mu (or tflux), and pflux
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

      if (Lconstraint == 1 .or. Lconstraint == 3 .or. (Lconstraint == 2 .and. Lfreebound == 1)) then
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

    integer :: llmodnp, Nbc, ierr
    MPI_STATUS_TYPE :: stat

    call fatal(lvol <= 0 .or. lvol > Mvol, __FILE__, "error")

    llmodnp = modulo(lvol - 1, ncpu)! identify which node contains data; this must be consistent with previous looping / parallelization

    Nbc = ntz_loc * 2 * LGdof * 2
    if (ncpu > 1 .and. llmodnp /= 0) then
      if (myid == llmodnp) then
        call MPI_SEND(freal_jac(1:ntz_loc, 0:1, 1:LGdof, 0:1, lvol), Nbc, &
            MPI_DOUBLE_PRECISION, 0, lvol, MPI_COMM_SPEC, ierr)
      else if (myid == 0) then
        call MPI_RECV(freal_jac(1:ntz_loc, 0:1, 1:LGdof, 0:1, lvol), Nbc, &
            MPI_DOUBLE_PRECISION, llmodnp, lvol, MPI_COMM_SPEC, stat, ierr)
      end if
    end if

    Nbc = ntz_loc * 2 * 2
    call broadcast(freal_dBBdmp(1:ntz_loc, lvol, 0:1, 1:2), Nbc, llmodnp)

    if (LocalConstraint) then
      Nbc = 2 * LGdof * 2
      call broadcast(dmupfdx(lvol, 1:1, 1:2, 1:LGdof, 0:1), Nbc, llmodnp)
    end if

    call broadcast(ImagneticOK(lvol), llmodnp)
  end subroutine bcast_freal_jac

end module field_mod
