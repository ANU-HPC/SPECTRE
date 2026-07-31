! SPDX-FileCopyrightText: 2026 Proxima Fusion
! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module beltrami_solver_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind, machprec
  use io_mod, only: ounit
  use input_list_mod, only: Wmacros, Wsolve_beltrami_system, Wcompute_rotational_transform, Wcompute_plasma_current, &
      Wconstruct_beltrami_field, mu, helicity, &
      iota, oita, curtor, curpol, Lrad, Ntor, Lconstraint, mupftol, enforce_stell_sym, &
      ngmresrestart, nitergmres, epsgmres, Mpol
  use allglobal_mod, only: myid, cpus, ivol, Lcoordinatesingularity, Lplasmaregion, mn, mns, Nt, Nz, Ntz, Iquad, NAdof, &
      dMA, dMB, dMD, dMG, solution, dtflux, dpflux, diotadxup, dItGpdxtp, lBBintegral, lABintegral, xoffset, ImagneticOK, &
      mvol, guvijsave, niters_gmres_total, beltrami_gmres_status, beltrami_gmres_iterations, &
      beltrami_gmres_residual
  use memory_allocation_mod, only: sallocate
  use errors_mod, only: fatal
  use plasma_current_mod, only: compute_plasma_current
  use packing_mod, only: packab
  use magnetic_field_mod, only: compute_rotational_transform
  use beltrami_operator_mod, only: beltrami_operator, op_alloc, op_release, op_factor_copy, op_solve_refine, &
      op_select_backend, op_bind_iter_sparse, op_solve_iter, op_apply_ad, op_prepare_metric, BK_ITER
  use beltrami_factor_cache_mod, only: factor_cache_insert
  use perturbed_workspace_mod, only: perturbed_rhs_workspace, alloc_perturbed_workspace, dealloc_perturbed_workspace, &
      matvec_multipliers, alloc_matvec_multipliers, dealloc_matvec_multipliers

  implicit none(type, external)

  private
  public :: solve_beltrami_system

  external :: DCOPY, DAXPY, DGEMV
  real(rkind), external :: DDOT

contains

  subroutine solve_beltrami_system(Ndof, Xdof, Fdof, Ddof, Ldfjac, iflag)
    integer, intent(in) :: Ndof
    integer, intent(in) :: Ldfjac
    real(rkind), intent(in) :: Xdof(1:Ndof)
    real(rkind), intent(inout) :: Fdof(1:Ndof)
    real(rkind), intent(inout) :: Ddof(1:Ldfjac, 1:Ndof)
    integer, intent(inout) :: iflag

    real(rkind) :: cput, resid_gmres
    integer :: lvol, NN, MM, ideriv, lmns, jd
    integer :: idgetrf(0:1), idgetrs(0:1), idgerfs(0:1), idgecon(0:1)
    real(rkind) :: lmu, dpf, dtf, dpsi(1:2), tpsi(1:2), ppsi(1:2), lcpu
    real(rkind) :: rcond
    character :: packorunpack

    ! Beltrami operator handle: owns the LU factor and LAPACK workspace
    type(beltrami_operator) :: fop
    real(rkind), allocatable :: matrix(:, :), rhs(:, :)
    ! scratch for the DGEMV+DDOT bilinear forms; outlives matrix/rhs (the helicity Ddof reads it)
    real(rkind), allocatable :: wtmp(:)

    ! iterative (matrix-free GMRES) backend, selected per volume by the input flags
    integer :: backend, aslot_fwd, its_gmres, ierr_gmres, max_lldof
    type(perturbed_rhs_workspace) :: pws_fwd
    type(matvec_multipliers) :: lm_fwd
    ! dMA.solution(:,0) and dMD.solution(:,0), evaluated matrix-free for the iterative backend so the
    ! mu-derivative RHS and the energy/helicity integrals need not read the dense dMA/dMD
    real(rkind), allocatable :: adx_sol(:), ddx_sol(:)
    real(rkind) :: dhel_dmu

    lvol = ivol ! recall that ivol is global

    if (Lplasmaregion) then
      lmu = Xdof(1) - xoffset
      dtf = dtflux(lvol)
      if (Ndof == 2) then
        dpf = Xdof(2) - xoffset
      else
        dpf = dpflux(lvol)
      end if

      ! only toroidal flux is modified
      if (Lconstraint == -2) then
        lmu = mu(lvol)
        dpf = dpflux(lvol)
        if (lvol == mvol) then
          dtf = Xdof(1) - xoffset
        else
          dtf = dtflux(lvol)
        end if
      end if

    else ! Lvacuumregion
      lmu = 0.0_rkind ! restrict attention to strict vacuum field
      dtf = Xdof(1) - xoffset
      if (Ndof == 2) then
        dpf = Xdof(2) - xoffset
      else
        dpf = dpflux(lvol)
      end if

    end if

    dpsi(1:2) = [dtf, dpf] ! enclosed poloidal fluxes and their derivatives
    tpsi(1:2) = [1.0_rkind, 0.0_rkind] ! enclosed toroidal fluxes and their derivatives
    ppsi(1:2) = [0.0_rkind, 1.0_rkind] ! enclosed toroidal fluxes and their derivatives

    diotadxup(0:1, -1:2, lvol) = 0.0_rkind ! rotational-transform, and its derivatives with respect to lmu and dpf, or
    ! toroidal current, on the inner/outer interface
    dItGpdxtp(0:1, -1:2, lvol) = 0.0_rkind ! plasma and linking currents

    NN = NAdof(lvol)! shorthand

    ! Select the linear-solver backend for this volume. The dense path below is unchanged (and
    ! bit-identical); the iterative branch replaces the factor+solve with matrix-free GMRES and
    ! never forms the dense operator, so its NN x NN `matrix` copy is allocated at minimal size.
    backend = op_select_backend(lvol)

    call sallocate(rhs, 1, NN, 0, 2, 0.0_rkind)
    allocate(wtmp(1:NN))
    if (backend == BK_ITER) then
      call sallocate(matrix, 1, 1, 1, 1, 0.0_rkind)  ! dummy not needed on matrix-free path
    else
      call sallocate(matrix, 1, NN, 1, NN, 0.0_rkind)
    end if

    solution(1:NN, -1:2) = 0.0_rkind ! this is a global array allocated in dforce

    aslot_fwd = -1
    if (backend == BK_ITER) then
      ! no dense NN x NN factor storage: op_bind_iter_sparse allocates only the O(mn*Lrad^2) factor
      if (Lcoordinatesingularity) then
        max_lldof = (Lrad(lvol) - mod(Lrad(lvol), 2)) / 2
      else ! Dense backend
        max_lldof = Lrad(lvol)
      end if
      call alloc_perturbed_workspace(pws_fwd, Ntz, mn, Iquad(lvol), Lrad(lvol), Mpol, max_lldof)
      call alloc_matvec_multipliers(lm_fwd, mn)
      allocate(adx_sol(1:NN), ddx_sol(1:NN))
      adx_sol(1:NN) = 0.0_rkind
      ddx_sol(1:NN) = 0.0_rkind
    else
      call op_alloc(fop, lvol, NN, .false.)
    end if

    idgetrf(0:1) = 0 ! error flags
    idgetrs(0:1) = 0 ! error flags
    idgerfs(0:1) = 0 ! error flags
    idgecon(0:1) = 0 ! error flags
    beltrami_gmres_status(lvol) = 0
    beltrami_gmres_iterations(lvol) = 0
    beltrami_gmres_residual(lvol) = 0.0_rkind

    ! ideriv labels derivative as follows:
    ! ideriv = 0 : compute Beltrami field: ideriv = 1 : compute d Beltrami field / d \mu           : ideriv = 2 : compute
    ! d Beltrami field / d \Delta \psi_p
    ! ideriv = 0 : compute Vacuum   field: ideriv = 1 : compute d Vacuum   field / d \Delta \psi_t : ideriv = 2 : compute
    ! d Vacuum   field / d \Delta \psi_p

    do ideriv = 0, 1 ! loop over derivatives
      if (iflag == 1 .and. ideriv == 1) cycle ! only need to return function: recall the derivative estimate requires
      ! function evaluation

      if (Lcoordinatesingularity) then
        if (backend /= BK_ITER) then
          do jd = 1, NN ! matrix = dMA - lmu*dMD, column-wise (lda NN vs NN+1); BLAS version
            call DCOPY(NN, dMA(1, jd), 1, matrix(1, jd), 1)
            call DAXPY(NN, -lmu, dMD(1, jd), 1, matrix(1, jd), 1)
          end do
        end if

        select case (ideriv)
        case (0)
          call DGEMV('N', NN, 2, -1.0_rkind, dMB(1, 1), NN + 1, dpsi(1), 1, 0.0_rkind, rhs(1, 0), 1)! BLAS version
        case (1) ! construct dMD*solution
          if (backend == BK_ITER) then
            rhs(1:NN, 1) = ddx_sol(1:NN) ! = dMD . solution(:,0), evaluated matrix-free
          else
            ! the signs cancel: no -1.0*dMD temporary, DGEMV's alpha handles it; BLAS version
            call DGEMV('N', NN, NN, 1.0_rkind, dMD(1, 1), NN + 1, solution(1, 0), 1, 0.0_rkind, rhs(1, 1), 1)
          end if
          call DGEMV('N', NN, 2, -1.0_rkind, dMB(1, 1), NN + 1, ppsi(1), 1, 0.0_rkind, rhs(1, 2), 1)! BLAS version
        end select

        if (Lconstraint == -2) then
          select case (ideriv)
          case (0)
            rhs(1:NN, 0) = -dMG(1:NN)
            call DGEMV('N', NN, 2, -1.0_rkind, dMB(1, 1), NN + 1, dpsi(1), 1, 1.0_rkind, rhs(1, 0), 1)! BLAS version
          case (1)
            call DGEMV('N', NN, 2, -1.0_rkind, dMB(1, 1), NN + 1, tpsi(1), 1, 0.0_rkind, rhs(1, 1), 1)! BLAS version
          end select
        end if

      else ! .not.Lcoordinatesingularity

        if (Lplasmaregion) then
          if (backend /= BK_ITER) then
          do jd = 1, NN ! matrix = dMA - lmu*dMD, column-wise (lda NN vs NN+1); BLAS version
            call DCOPY(NN, dMA(1, jd), 1, matrix(1, jd), 1)
            call DAXPY(NN, -lmu, dMD(1, jd), 1, matrix(1, jd), 1)
          end do
        end if

          select case (ideriv)
          case (0)
            rhs(1:NN, 0) = -matmul(dMB(1:NN, 1:2), dpsi(1:2))
          case (1) ! construct dMD*solution
            if (backend == BK_ITER) then
              rhs(1:NN, 1) = ddx_sol(1:NN) ! = dMD . solution(:,0), evaluated matrix-free
            else
              rhs(1:NN, 1) = -matmul(-1.0_rkind * dMD(1:NN, 1:NN), solution(1:NN, 0))
            end if
            rhs(1:NN, 2) = -matmul(dMB(1:NN, 1:2), ppsi(1:2))
          end select

        else ! Lvacuumregion
          if (backend /= BK_ITER) then
            do jd = 1, NN ! matrix = dMA (no -lmu*dMD in vacuum), column-wise; BLAS version
              call DCOPY(NN, dMA(1, jd), 1, matrix(1, jd), 1)
            end do
          end if

          select case (ideriv)! perhaps there is an lmu term missing in these rhs
          case (0)
            rhs(1:NN, 0) = -dMG(1:NN)
            call DGEMV('N', NN, 2, -1.0_rkind, dMB(1, 1), NN + 1, dpsi(1), 1, 1.0_rkind, rhs(1, 0), 1)! BLAS version
          case (1)
            call DGEMV('N', NN, 2, -1.0_rkind, dMB(1, 1), NN + 1, tpsi(1), 1, 0.0_rkind, rhs(1, 1), 1)! BLAS version
            call DGEMV('N', NN, 2, -1.0_rkind, dMB(1, 1), NN + 1, ppsi(1), 1, 0.0_rkind, rhs(1, 2), 1)! BLAS version
          end select
        end if
      end if

      select case (ideriv)
      case (0) ! ideriv=0
        MM = 1
        if (backend == BK_ITER) then
          ! bind the exact same-(m,n) preconditioner for the current mu, then matrix-free GMRES.
          ! The diagonal metric integrals it reads were filled by the caller (dfp100).
          call op_bind_iter_sparse(fop, lvol, NN, Lcoordinatesingularity, lmu)
          call op_prepare_metric(fop, Lcoordinatesingularity, pws_fwd) ! compute metric once; reused by all solves below
          ! Warm start: seed from the previous converged solution in the persistent Ate/Aze store
          ! (slot 0), written by packab('U') at the end of every call -- so this seeds across force
          ! evaluations AND across mu-iterations within one. Seed-only semantics: GMRES judges
          ! convergence on the true residual of the current system, so a stale seed costs iterations,
          ! never correctness. On a fresh instance the store is zero => today's cold start. The
          ! Lagrange-multiplier DOFs are not covered by packab and stay at the zero fill.
          solution(1:NN, 0) = 0.0_rkind
          call packab('P', lvol, NN, solution(1:NN, 0), 0)
          call op_solve_iter(fop, lmu, Lcoordinatesingularity, aslot_fwd, rhs(1:NN, 0), solution(1:NN, 0), &
              pws_fwd, lm_fwd, ngmresrestart, nitergmres, epsgmres, its_gmres, resid_gmres, ierr_gmres)
          niters_gmres_total = niters_gmres_total + its_gmres
          beltrami_gmres_status(lvol) = max(beltrami_gmres_status(lvol), ierr_gmres)
          beltrami_gmres_iterations(lvol) = beltrami_gmres_iterations(lvol) + its_gmres
          beltrami_gmres_residual(lvol) = max(beltrami_gmres_residual(lvol), resid_gmres)
          ! A stalled solve silently degrades the force rather than failing, so say so loudly. The
          ! usual cause is running the coordinate-singularity volume iteratively (see nadofgmres_sing).
          if (ierr_gmres /= 0) then
            write(ounit, '("solve_beltrami_system : WARNING : GMRES did not converge in volume ",i3, &
                &" (NAdof=",i7,") after ",i7," iterations; the field for this volume is not converged.", &
                &" Raise nitergmres, or keep this volume on the dense path.")') lvol, NN, its_gmres
          end if
          ! Reaching the iteration limit is recoverable: GMRES leaves its best finite iterate in
          ! solution. Keep the volume usable and expose the degraded status separately.
          idgetrs(ideriv) = 0
          rcond = 1.0_rkind ! GMRES has no condition estimate; convergence is judged by ierr_gmres
          ! dMA.sol and dMD.sol for the mu-derivative RHS (ideriv=1) and the energy/helicity integrals
          call op_apply_ad(fop, Lcoordinatesingularity, aslot_fwd, solution(1:NN, 0), adx_sol, ddx_sol, pws_fwd, lm_fwd)
        else  !Dense backend
          call op_factor_copy(fop, NN, matrix, rcond, idgetrf(ideriv), idgecon(ideriv))! LU factorization + condition estimate
          call op_solve_refine(fop, NN, matrix, rhs(1:NN, 0:0), solution(1:NN, 0:0), MM, idgetrs(ideriv), idgerfs(ideriv))
        end if

      case (1) ! ideriv=1
        MM = 2
        if (backend == BK_ITER) then
          ! the two derivative RHS share the operator (and its preconditioner) from ideriv=0;
          ! warm-started from the previous derivative solutions (slots 1:2), seed-only as above
          solution(1:NN, 1) = 0.0_rkind
          call packab('P', lvol, NN, solution(1:NN, 1), 1)
          call op_solve_iter(fop, lmu, Lcoordinatesingularity, aslot_fwd, rhs(1:NN, 1), solution(1:NN, 1), &
              pws_fwd, lm_fwd, ngmresrestart, nitergmres, epsgmres, its_gmres, resid_gmres, ierr_gmres)
          niters_gmres_total = niters_gmres_total + its_gmres
          beltrami_gmres_status(lvol) = max(beltrami_gmres_status(lvol), ierr_gmres)
          beltrami_gmres_iterations(lvol) = beltrami_gmres_iterations(lvol) + its_gmres
          beltrami_gmres_residual(lvol) = max(beltrami_gmres_residual(lvol), resid_gmres)
          idgetrs(ideriv) = 0
          solution(1:NN, 2) = 0.0_rkind
          call packab('P', lvol, NN, solution(1:NN, 2), 2)
          call op_solve_iter(fop, lmu, Lcoordinatesingularity, aslot_fwd, rhs(1:NN, 2), solution(1:NN, 2), &
              pws_fwd, lm_fwd, ngmresrestart, nitergmres, epsgmres, its_gmres, resid_gmres, ierr_gmres)
          niters_gmres_total = niters_gmres_total + its_gmres
          beltrami_gmres_status(lvol) = max(beltrami_gmres_status(lvol), ierr_gmres)
          beltrami_gmres_iterations(lvol) = beltrami_gmres_iterations(lvol) + its_gmres
          beltrami_gmres_residual(lvol) = max(beltrami_gmres_residual(lvol), resid_gmres)
          rcond = 1.0_rkind
        else ! Dense backend
          call op_solve_refine(fop, NN, matrix, rhs(1:NN, 1:2), solution(1:NN, 1:2), MM, idgetrs(ideriv), idgerfs(ideriv))
        end if
      end select ! ideriv

      cput = MPI_Wtime()

      if (idgetrf(ideriv) == 0 .and. idgetrs(ideriv) == 0 .and. idgerfs(ideriv) == 0 .and. rcond >= machprec) then
        if (Wsolve_beltrami_system) write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), &
            idgetrs(ideriv), idgetrf(ideriv), "success", cput - lcpu
      elseif (idgetrf(ideriv) < 0 .or. idgetrs(ideriv) < 0 .or. idgerfs(ideriv) < 0) then
        write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), idgetrs(ideriv), &
            idgetrf(ideriv), "input error"
      elseif (idgetrf(ideriv) > 0) then
        write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), idgetrs(ideriv), &
            idgetrf(ideriv), "singular"
      elseif (rcond <= machprec) then
        write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), idgetrs(ideriv), &
            idgetrf(ideriv), "ill conditioned"
      else
        write(ounit, 1010)cput - cpus, myid, lvol, ideriv, "idgetrf idgetrs idgerfs", idgetrf(ideriv), idgetrs(ideriv), &
            idgetrf(ideriv), "invalid error"
      end if

      1010 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; ideriv=", i2, " ; ", a23, "=", i3, ' ', &
          i3, ' ', i3, " ;", a34, :, " time=", f10.2, " ")
    end do

    do ideriv = 0, 2
      if (iflag == 1 .and. ideriv > 0) cycle

      packorunpack = 'U'
      call packab(packorunpack, lvol, NN, solution(1:NN, ideriv), ideriv)! unpacking: this assigns oAt, oAz through
      ! common
    end do ! do ideriv = 0, 2

    ! can compute the energy and helicity integrals: easiest to do this with solution in packed format
    ! (the iterative backend uses the matrix-free dMA.sol / dMD.sol computed above instead of dMA/dMD)
    call DGEMV('N', NN, 2, 1.0_rkind, dMB(1, 1), NN + 1, dpsi(1), 1, 0.0_rkind, wtmp(1), 1)! wtmp = dMB.dpsi; BLAS version
    if (backend == BK_ITER) then
      lBBintegral(lvol) = 0.5_rkind * DDOT(NN, solution(1, 0), 1, adx_sol(1), 1) &
          + DDOT(NN, solution(1, 0), 1, wtmp(1), 1)
      lABintegral(lvol) = 0.5_rkind * DDOT(NN, solution(1, 0), 1, ddx_sol(1), 1)
      ! dK/dmu = (dsol/dmu) . dMD . sol with dMD symmetric; the helicity constraint (Lconstraint==2)
      ! reads it below, after adx_sol/ddx_sol are released. Stale slot-1 data when the derivative
      ! solve was skipped (iflag==1) is harmless: Ddof is only consumed on iflag==2.
      dhel_dmu = DDOT(NN, solution(1, 1), 1, ddx_sol(1), 1)
    else ! Dense backend
      lBBintegral(lvol) = DDOT(NN, solution(1, 0), 1, wtmp(1), 1)
      call DGEMV('N', NN, NN, 1.0_rkind, dMA(1, 1), NN + 1, solution(1, 0), 1, 0.0_rkind, wtmp(1), 1)! wtmp = dMA.sol
      lBBintegral(lvol) = lBBintegral(lvol) + 0.5_rkind * DDOT(NN, solution(1, 0), 1, wtmp(1), 1)
      call DGEMV('N', NN, NN, 1.0_rkind, dMD(1, 1), NN + 1, solution(1, 0), 1, 0.0_rkind, wtmp(1), 1)! wtmp = dMD.sol
      lABintegral(lvol) = 0.5_rkind * DDOT(NN, solution(1, 0), 1, wtmp(1), 1)
    end if

    deallocate(matrix)
    deallocate(rhs)
    ! Deposit a successful factorization for reuse by a following JVP batch (opt-in, no-op unless
    ! Lbeltrami_factor_cache): the last call of the mu-iteration is the accepted one, so the cached
    ! factor is the operator at the converged (geometry, mu). factor_cache_insert moves the storage,
    ! so the release below stays a safe no-op.
    if (all(idgetrf(0:1) == 0) .and. all(idgetrs(0:1) == 0) &
        .and. all(idgerfs(0:1) == 0) .and. all(idgecon(0:1) == 0)) then
      call factor_cache_insert(lvol, lmu, fop)
    end if
    call op_release(fop)
    if (backend == BK_ITER) then
      call dealloc_perturbed_workspace(pws_fwd)
      call dealloc_matvec_multipliers(lm_fwd)
      deallocate(adx_sol, ddx_sol)
    end if

    idgetrf(0:1) = abs(idgetrf(0:1)) + abs(idgetrs(0:1)) + abs(idgerfs(0:1)) + abs(idgecon(0:1))
    if (idgetrf(0)/= 0 .or. idgetrf(1)/= 0) then ! failed to construct Beltrami/vacuum field and/or derivatives

      ImagneticOK(lvol) = .false. ! set error flag

      if (iflag == 1) Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out
      if (iflag == 2) Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      iflag = -1 ! this value will be returned by C05PCF to construct_beltrami_field

      return

    else

      ImagneticOK(lvol) = .true. ! set error flag: used in dforce

    end if

    if (enforce_stell_sym) then
      lmns = 1 + (mns - 1) ! number of independent degrees of freedom in angle transformation
    else
      lmns = 1 + (mns - 1) + (mns - 1) ! only required for dense, Fourier angle transformation
    end if

    select case (Lconstraint)

    case (-2 ) ! Lconstraint=-2
      call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)

      ! Constraint on curpol only
      if ( iflag == 1 ) Fdof(1) = dItGpdxtp(1, 0, lvol) - curpol

      ! Derivative w.r.t. toroidal flux only
      if ( iflag == 2 ) Ddof(1, 1) = dItGpdxtp(1, 1, lvol)
    case (-1) ! Lconstraint=-1

      if (Lplasmaregion) then

        if (Wcompute_rotational_transform) then ! compute rotational transform only for diagnostic purposes
          call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)
        end if

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out: Lconstraint=-1 indicates no iterations over mu   , dpflux
        ! are required
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      else ! Lvacuumregion

        if (Wcompute_rotational_transform) then ! compute rotational transform only for diagnostic purposes
          call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)
        end if

        if (Wcompute_plasma_current) then ! compute enclosed currents    only for diagnostic purposes
          call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)
          curtor = dItGpdxtp(0, 0, lvol)! icurrent(0) ! update input variables
          curpol = dItGpdxtp(1, 0, lvol)! gcurrent(0)
        end if

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out:Lconstraint=-1 indicates no iterations over dtflux, dpflux
        ! are required
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      end if

    case (0) ! Lconstraint= 0

      if (Lplasmaregion) then

        if (Wcompute_rotational_transform) then ! compute rotational transform only for diagnostic purposes
          call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)
        end if

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out: Lconstraint= 0 indicates no iterations over mu, dpflux are
        ! required
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      else ! Lvacuumregion

        call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)

        if (iflag == 1) Fdof(1:2) = [dItGpdxtp(0, 0, lvol) - curtor, dItGpdxtp(1, 0, lvol) - curpol]
        if (iflag == 2) Ddof(1:2, 1) = [dItGpdxtp(0, 1, lvol), dItGpdxtp(1, 1, lvol)]
        !if( iflag.eq.2 ) Ddof(1:2,2) = (/ dItGpdxtp(0,0,lvol)         , dItGpdxtp(1,2,lvol)          /)
        if (iflag == 2) Ddof(1:2, 2) = [dItGpdxtp(0, 2, lvol), dItGpdxtp(1, 2, lvol)]

      end if

    case (1) ! Lconstraint= 1

      call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)! required for both plasma and vacuum region

      if (Lplasmaregion) then

        if (Lcoordinatesingularity) then ! Ndof = 1
          if (iflag == 1) Fdof(1) = diotadxup(1, 0, lvol) - iota(lvol)
          if (iflag == 2) Ddof(1, 1) = diotadxup(1, 1, lvol)! derivative of outer rotational-transform wrt helicity
          ! multiplier
        end if

        if (Ndof == 2) then
          if (iflag == 1) Fdof(1:2) = diotadxup(0:1, 0, lvol) - [oita(lvol - 1), iota(lvol)]
          if (iflag == 2) Ddof(1:2, 1) = diotadxup(0:1, 1, lvol)
          if (iflag == 2) Ddof(1:2, 2) = diotadxup(0:1, 2, lvol)
        end if

      else ! Lvacuumregion

        call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)

        curtor = dItGpdxtp(0, 0, lvol)! update input variables
        !curpol = dItGpdxtp(1,0,lvol)

        if (iflag == 1) Fdof(1:2) = [diotadxup(0, 0, lvol) - oita(lvol - 1), dItGpdxtp(1, 0, lvol) - curpol]
        if (iflag == 2) Ddof(1:2, 1) = [diotadxup(0, 1, lvol), dItGpdxtp(1, 1, lvol)]
        if (iflag == 2) Ddof(1:2, 2) = [diotadxup(0, 2, lvol), dItGpdxtp(1, 2, lvol)]

      end if

    case (2)

      if (Lplasmaregion) then

        if (iflag == 1) Fdof(1) = lABintegral(lvol) - helicity(lvol)

        if (iflag == 2) then
          if (backend == BK_ITER) then
            Ddof(1, 1) = dhel_dmu ! evaluated matrix-free above, while ddx_sol was still allocated
          else
            ! dMD is symmetric, so 0.5*(x1.dMD.x0 + x0.dMD.x1) = x1.(dMD.x0) — one DGEMV instead
            ! of two NN x NN matmuls; mirrors dhel_dmu on the matrix-free branch. BLAS version.
            call DGEMV('N', NN, NN, 1.0_rkind, dMD(1, 1), NN + 1, solution(1, 0), 1, 0.0_rkind, wtmp(1), 1)
            Ddof(1, 1) = DDOT(NN, solution(1, 1), 1, wtmp(1), 1)
          end if
        end if

      else ! Lvacuumregion: the fluxes are varied to match the currents, exactly as for
        ! Lconstraint=0; the vacuum helicity is unconstrained (mu=0 leaves no freedom to
        ! impose it) and stays whatever the current-matched field gives.

        call compute_plasma_current(lvol, mn, Nt, Nz, Lcoordinatesingularity, iflag, dItGpdxtp(0:1, -1:2, lvol), -1)

        if (iflag == 1) Fdof(1:2) = [dItGpdxtp(0, 0, lvol) - curtor, dItGpdxtp(1, 0, lvol) - curpol]
        if (iflag == 2) Ddof(1:2, 1) = [dItGpdxtp(0, 1, lvol), dItGpdxtp(1, 1, lvol)]
        if (iflag == 2) Ddof(1:2, 2) = [dItGpdxtp(0, 2, lvol), dItGpdxtp(1, 2, lvol)]

      end if

    case (3)

      if (Lplasmaregion) then

        if (Wcompute_rotational_transform) then ! compute rotational transform only for diagnostic purposes
          call compute_rotational_transform(lvol, mn, lmns, Nt, Nz, Lcoordinatesingularity, iflag, diotadxup(0:1, -1:2, lvol), -1)
        end if

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out: no iteration other mu and psip locally
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      else ! Lvacuumregion

        !WCALL( solve_beltrami_system, compute_plasma_current,( lvol, mn, Nt, Nz, iflag, dItGpdxtp(0:1,-1:2,lvol) ) )

        ! Iteration only on toroidal flux.
        ! if( iflag.eq.1 ) Fdof(1:Ndof  ) = dItGpdxtp(1,0,lvol) - curpol
        ! if( iflag.eq.2 ) Ddof(1:Ndof,1:Ndof) = dItGpdxtp(1,1,lvol)

        Fdof(1:Ndof) = 0.0_rkind ! provide dummy intent out: no iteration other mu and psip locally
        Ddof(1:Ndof, 1:Ndof) = 0.0_rkind ! provide dummy intent out

      end if

    end select

    if (Wsolve_beltrami_system .or. Wconstruct_beltrami_field) then ! the following is screen output

      cput = MPI_Wtime()

      if (Lplasmaregion) then
        select case (iflag)
        case (0)
          write(ounit, 3000)cput - cpus, myid, lvol, lmu, dpf, iflag ! this is impossible by above logic
        case (1)
          write(ounit, 3000)cput - cpus, myid, lvol, lmu, dpf, iflag, Fdof(1:Ndof)
        case (2)
          write(ounit, 3010)cput - cpus, myid, lvol, lmu, dpf, iflag, Ddof(1:Ndof, 1:Ndof)
        case default
          call fatal(.true., __FILE__, "illegal iflag on entry ")

        end select
      else ! Lvacuumregion
        select case (iflag)
        case (0)
          write(ounit, 3001)cput - cpus, myid, lvol, dtf, dpf, iflag ! this is impossible by above logic
        case (1)
          write(ounit, 3001)cput - cpus, myid, lvol, dtf, dpf, iflag, Fdof(1:Ndof)
        case (2)
          write(ounit, 3011)cput - cpus, myid, lvol, dtf, dpf, iflag, Ddof(1:Ndof, 1:Ndof)
        case default
          call fatal(.true., __FILE__, "illegal iflag on entry ")

        end select
      end if

      3000 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (mu,dp)=(", es23.15, " ,", es23.15, &
          " ) ; iflag=", i2, " ;", :, " F=", 2es13.05, " ")
      3010 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (mu,dp)=(", es23.15, " ,", es23.15, &
          " ) ; iflag=", i2, " ;", :, " D=", 4es13.05, " ")
      3001 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (dt,dp)=(", es23.15, " ,", es23.15, &
          " ) ; iflag=", i2, " ;", :, " F=", 2es13.05, " ")
      3011 format("solve_beltrami_system : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; (dt,dp)=(", es23.15, " ,", es23.15, &
          " ) ; iflag=", i2, " ;", :, " D=", 4es13.05, " ")

    end if

    if (iflag == 1) then ! only in this case is Fdof defined

      if (sum(abs(Fdof(1:Ndof))) / Ndof < mupftol) then ! satisfactory

        if (Lplasmaregion) then
          mu(lvol) = lmu

          dpflux(lvol) = dpf
        else
          mu(lvol) = 0.0_rkind
          dtflux(lvol) = dtf
          dpflux(lvol) = dpf
        end if

        if ( Lconstraint == -2) then
          !mu(lvol) = lmu
          dtflux(lvol) = dtf
        end if

        iflag = -2 ! return "acceptance" flag through to construct_beltrami_field via ifail: early termination
      end if
    end if
  end subroutine solve_beltrami_system
end module beltrami_solver_mod
