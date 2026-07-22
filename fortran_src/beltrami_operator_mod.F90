! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> \brief Per-volume Beltrami linear-operator seam: one handle that owns the factorization
!> storage and exposes bind/solve operations, so the dense LAPACK path and the matrix-free
!> GMRES backend are interchangeable behind one interface.
module beltrami_operator_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind
  use allglobal_mod, only: MPI_COMM_SPEC, mn, Nt, Nz, Iquad, dBdX, Lsavedguvij, NAdof, Mvol, Lbeltrami_factor_cache
  use input_list_mod, only: Lrad, Lbeltramisolver, nadofgmres, nadofgmres_sing, Lconstraint, &
      enforce_stell_sym, Igeometry, Lfreebound, lvcvacuum
  use memory_allocation_mod, only: sallocate
  use perturbed_workspace_mod, only: perturbed_rhs_workspace, matvec_multipliers
  use intghs_mod, only: intghs
  use matrices_mod, only: mtrxhs_full
  use packing_mod, only: packab_matvec
  use metric_mod, only: compute_guvijsave
  use beltrami_sparse_mod, only: build_dof_mode, samemn_csr_count, build_samemn_csr, build_samemn_csr_diag
  use vc_matrix_mod, only: vc_matvec_sub
  use errors_mod, only: fatal
  use gmres_mod, only: gmres_solve, block_lu, block_lu_factor, block_lu_apply, block_lu_release

  implicit none(type, external)

  private
  public :: beltrami_operator, BK_DENSE, BK_ITER
  public :: op_alloc, op_release, op_factor_prepared, op_factor_copy
  public :: op_solve, op_solve_refine, op_solve_multi, op_bcast, op_apply
  public :: op_bind_iter, op_bind_iter_sparse, op_solve_iter, op_select_backend, op_apply_ad
  public :: op_prepare_metric, op_solve_multi_iter
  public :: op_cache_invalidate, op_cache_put, op_cache_take

  external :: DCOPY, DGETRF, DGETRS, DGERFS, DGECON

  integer, parameter :: BK_DENSE = 1 !< dense LU backend (LAPACK)
  integer, parameter :: BK_ITER = 2 !< iterative backend (matrix-free GMRES + same-(m,n) block-LU preconditioner)

  !> \brief Handle for one volume's Beltrami linear operator A - mu*D (A only in vacuum).
  !> Factor storage is (0:NN,0:NN): the bordered (Lconstraint==2) system uses the full block,
  !> the standard system the (1:NN,1:NN) sub-block with leading dimension NN+1.
  type beltrami_operator
    integer :: vvol = 0 !< volume this operator is bound to
    integer :: NN = 0 !< NAdof(vvol)
    integer :: backend = BK_DENSE
    logical :: bordered = .false. !< Lconstraint==2 augmented (NN+1)x(NN+1) system
    real(rkind), allocatable :: mat(:, :) !< dense factor storage
    integer, allocatable :: ipivot(:) !< pivots from DGETRF
    real(rkind), allocatable :: rwork(:) !< real workspace for DGECON/DGERFS
    integer, allocatable :: iwork(:) !< integer workspace for DGECON/DGERFS

    ! iterative (BK_ITER) backend: same-(m,n) sparse preconditioner (CSR) + its exact block-diagonal
    ! LU factor (one dense pivoted LU per Fourier-mode block; handles the saddle point exactly)
    real(rkind), allocatable :: pval(:) !< CSR values of the same-(m,n) approximation
    integer, allocatable :: pja(:), pia(:) !< CSR columns, row pointers
    integer, allocatable :: dof_mode(:) !< DOF -> Fourier-mode map
    type(block_lu) :: blu !< exact block-diagonal LU of the same-(m,n) preconditioner
  end type beltrami_operator

  ! Cross-call factor cache (opt-in via allglobal Lbeltrami_factor_cache): the forward field solve
  ! deposits each volume's final factorization here (op_cache_put in solve_beltrami_system), and
  ! the Jacobian-vector-product batch takes it back (op_cache_take in solve_field_deriv_dir)
  ! instead of refactoring the very same operator at the very same geometry. Handles are moved,
  ! never copied. A slot is trusted only if the stored (mu, NAdof) match the request; dfp100 drops
  ! each volume's slot before re-solving it, so a slot can never outlive its geometry through the
  ! normal force-evaluation flow. Single MPI rank per slot (each rank caches its own volumes; the
  ! only consumer is single-rank).
  type(beltrami_operator), allocatable :: op_cache(:)
  real(rkind), allocatable :: op_cache_mu(:) !< mu each cached factorization was built at
  logical, allocatable :: op_cache_ok(:) !< slot holds a live factorization

contains

  !> \brief Select the linear-solver backend for volume vvol from the input flags. Returns BK_DENSE
  !> unless the matrix-free GMRES path is enabled (Lbeltramisolver /= 0), the volume is large enough
  !> (Lbeltramisolver==1 uses the nadofgmres threshold; ==2 forces iterative), the constraint is not
  !> the bordered helicity system (Lconstraint==2 stays dense), and the case is stellarator-symmetric
  !> (the sparse preconditioner is stell-sym only).
  integer function op_select_backend(vvol) result(backend)
    integer, intent(in) :: vvol

    backend = BK_DENSE
    if (Lbeltramisolver == 0) return
    if (Lconstraint == 2) return
    if (.not. enforce_stell_sym) return
    ! The innermost (coordinate-singularity, Zernike) volume is kept dense unless explicitly opted in:
    ! the same-(m,n) block is a poor preconditioner there and GMRES stalls, which degrades the force
    ! rather than failing outright (measured at m21n17: no convergence in 4000 iterations, force_norm
    ! 2.5947 instead of 2.4638). Every other volume converges in tens of iterations.
    if (vvol == 1 .and. Igeometry /= 1 .and. NAdof(vvol) < nadofgmres_sing) return
    if (Lbeltramisolver == 2) then
      backend = BK_ITER
    else if (Lbeltramisolver == 1 .and. NAdof(vvol) >= nadofgmres) then
      backend = BK_ITER
    end if
  end function op_select_backend

  !> \brief Allocate the operator's factor storage for volume vvol with NN = NAdof(vvol).
  subroutine op_alloc(op, vvol, NN, bordered)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: vvol, NN
    logical, intent(in) :: bordered

    op%vvol = vvol
    op%NN = NN
    op%bordered = bordered

    call sallocate(op%mat, 0, NN, 0, NN, 0.0_rkind)
    call sallocate(op%ipivot, 0, NN, 0)
    call sallocate(op%rwork, 1, 4 * NN, 0.0_rkind)
    call sallocate(op%iwork, 1, NN, 0)
  end subroutine op_alloc

  !> \brief Release the operator's storage.
  subroutine op_release(op)
    type(beltrami_operator), intent(inout) :: op

    if (allocated(op%mat)) deallocate(op%mat)
    if (allocated(op%ipivot)) deallocate(op%ipivot)
    if (allocated(op%rwork)) deallocate(op%rwork)
    if (allocated(op%iwork)) deallocate(op%iwork)
    if (allocated(op%pval)) deallocate(op%pval)
    if (allocated(op%pja)) deallocate(op%pja)
    if (allocated(op%pia)) deallocate(op%pia)
    if (allocated(op%dof_mode)) deallocate(op%dof_mode)
    call block_lu_release(op%blu)
    op%vvol = 0
    op%NN = 0
  end subroutine op_release

  !> \brief Factor a caller-prepared matrix (already A - mu*D, plus border row/column when
  !> bordered) in place and keep the factor + pivots in the operator. Amat is destroyed.
  subroutine op_factor_prepared(op, NN, Amat, info)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN
    real(rkind), intent(inout) :: Amat(0:NN, 0:NN)
    integer, intent(out) :: info

    if (op%bordered) then
      call DGETRF(NN + 1, NN + 1, Amat(0, 0), NN + 1, op%ipivot, info)
      call DCOPY( (NN + 1) * (NN + 1) , Amat, 1, op%mat, 1)
    else
      call DGETRF(NN, NN, Amat(1, 1), NN + 1, op%ipivot(1:NN), info)
      op%mat(1:NN, 1:NN) = Amat(1:NN, 1:NN)
    end if
  end subroutine op_factor_prepared

  !> \brief Copy the caller's matrix into the operator, factor there (the caller's matrix is
  !> preserved for iterative refinement) and estimate the condition number.
  subroutine op_factor_copy(op, NN, matrix, rcond, info_trf, info_con)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN
    real(rkind), intent(in) :: matrix(1:NN, 1:NN)
    real(rkind), intent(out) :: rcond
    integer, intent(out) :: info_trf, info_con

    real(rkind) :: anorm

    op%mat(1:NN, 1:NN) = matrix(1:NN, 1:NN)
    call DGETRF(NN, NN, op%mat(1, 1), NN + 1, op%ipivot(1:NN), info_trf)

    anorm = maxval(sum(abs(matrix), 1))
    call DGECON('I', NN, op%mat(1, 1), NN + 1, anorm, rcond, op%rwork, op%iwork, info_con)
  end subroutine op_factor_copy

  !> \brief Back-substitute one right-hand side in place. x has length NN+1 for the bordered
  !> system and NN otherwise.
  subroutine op_solve(op, n, x, info)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: n
    real(rkind), intent(inout) :: x(1:n)
    integer, intent(out) :: info

    if (op%bordered) then
      call DGETRS('N', n, 1, op%mat(0, 0), n, op%ipivot(0:n - 1), x(1), n, info)
    else
      call DGETRS('N', n, 1, op%mat(1, 1), n + 1, op%ipivot(1:n), x(1), n, info)
    end if
  end subroutine op_solve

  !> \brief Back-substitute nrhs right-hand sides and refine against the caller's unfactored
  !> matrix (DGETRS + DGERFS, as the forward field solve requires).
  subroutine op_solve_refine(op, NN, matrix, rhs, sol, nrhs, info_trs, info_rfs)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN, nrhs
    real(rkind), intent(in) :: matrix(1:NN, 1:NN)
    real(rkind), intent(in) :: rhs(1:NN, 1:nrhs)
    real(rkind), intent(inout) :: sol(1:NN, 1:nrhs)
    integer, intent(out) :: info_trs, info_rfs

    real(rkind) :: ferr(1:nrhs), berr(1:nrhs)

    sol(1:NN, 1:nrhs) = rhs(1:NN, 1:nrhs)
    call DGETRS('N', NN, nrhs, op%mat(1, 1), NN + 1, op%ipivot(1:NN), sol(1, 1), NN, info_trs)
    call DGERFS('N', NN, nrhs, matrix(1, 1), NN, op%mat(1, 1), NN + 1, op%ipivot(1:NN), rhs(1, 1), NN, &
        sol(1, 1), NN, ferr, berr, op%rwork, op%iwork, info_rfs)
  end subroutine op_solve_refine

  !> \brief Back-substitute a batch of nb right-hand sides in place; rhs holds one column of
  !> length ldr (the caller's leading dimension) per right-hand side.
  subroutine op_solve_multi(op, NN, ldr, rhs, nb, info)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN, ldr, nb
    real(rkind), intent(inout) :: rhs(1:ldr, 1:nb)
    integer, intent(out) :: info

    call DGETRS('N', NN, nb, op%mat(1, 1), NN + 1, op%ipivot(1:NN), rhs(1, 1), ldr, info)
  end subroutine op_solve_multi

  !> \brief Matrix-free application of the Beltrami operator: ax = (dMA - mu*dMD) . x, computed
  !> without forming the matrix. The input x (length NN, augmented interior + Lagrange DOFs) is
  !> unpacked into vector-potential slot aslot, projected by intghs, and assembled by mtrxhs_full.
  !> Vacuum region has mu = 0, reproducing the dMA-only operator. Caller supplies per-thread pws
  !> (intghs/mtrxhs scratch) and lm (multiplier scratch); aslot is the per-thread Ate/Aze/Ato/Azo
  !> storage slot. This sets dBdX%L=.false. and Lsavedguvij=.false. (intghs recomputes the metric).
  subroutine op_apply(op, mu, Lcoord, aslot, x, ax, pws, lm)
    type(beltrami_operator), intent(in) :: op
    real(rkind), intent(in) :: mu
    logical, intent(in) :: Lcoord
    integer, intent(in) :: aslot
    real(rkind), intent(in) :: x(1:op%NN)
    real(rkind), intent(out) :: ax(1:op%NN)
    type(perturbed_rhs_workspace), intent(inout) :: pws
    type(matvec_multipliers), intent(inout) :: lm

    real(rkind) :: dax(1:op%NN), ddx(1:op%NN)

    call op_apply_ad(op, Lcoord, aslot, x, dax, ddx, pws, lm)
    ax(1:op%NN) = dax(1:op%NN) - mu * ddx(1:op%NN)
  end subroutine op_apply

  !> \brief Apply the two component operators separately: adx = dMA . x and ddx = dMD . x, without
  !> forming either matrix. The forward field solve needs them apart to build the mu-derivative RHS
  !> (dMD . solution) and the energy/helicity integrals; op_apply combines them as dMA - mu*dMD.
  subroutine op_apply_ad(op, Lcoord, aslot, x, adx, ddx, pws, lm)
    type(beltrami_operator), intent(in) :: op
    logical, intent(in) :: Lcoord
    integer, intent(in) :: aslot
    real(rkind), intent(in) :: x(1:op%NN)
    real(rkind), intent(out) :: adx(1:op%NN), ddx(1:op%NN)
    type(perturbed_rhs_workspace), intent(inout) :: pws
    type(matvec_multipliers), intent(inout) :: lm

    real(rkind) :: dax0(0:op%NN), ddx0(0:op%NN), x0(0:op%NN)

    dBdX%L = .false. ! matvec needs the full operator incl. dMD and Lagrange coupling
    dBdX%vol = op%vvol
    ! Lsavedguvij is left to the caller: op_solve_iter computes the (fixed) metric once into
    ! pws%guvijsave and sets Lsavedguvij=.true. so intghs reuses it across all matvecs of a solve.
    ! A caller that has not prepared the metric must set Lsavedguvij=.false. (intghs then computes it).

    call packab_matvec(op%vvol, op%NN, x, aslot, lm)
    call intghs(Iquad(op%vvol), mn, op%vvol, Lrad(op%vvol), aslot, Nt, Nz, Lcoord, pws)
    call mtrxhs_full(op%vvol, mn, Lrad(op%vvol), dax0, ddx0, aslot, Lcoord, pws, lm)

    ! Direct free-boundary solve: the vacuum operator carries the Laplace-virtual-casing rows
    ! (subtracted in place from the dense dMA by vc_subtract_dvc); subtract their action here so
    ! the matvec reproduces the corrected dense operator without forming it.
    if (Lfreebound == 1 .and. lvcvacuum == 1 .and. op%vvol == Mvol) then
      x0(0) = 0.0_rkind
      x0(1:op%NN) = x(1:op%NN)
      call vc_matvec_sub(op%NN, x0, dax0(1:op%NN))
    end if

    adx(1:op%NN) = dax0(1:op%NN)
    ddx(1:op%NN) = ddx0(1:op%NN)
  end subroutine op_apply_ad

  !> \brief Bind the iterative backend by EXTRACTING the same-(m,n) preconditioner block from the
  !> dense dMA/dMD and factoring it with the exact per-mode block LU. Production never forms those
  !> dense matrices and uses op_bind_iter_sparse instead; this extraction path is kept as the
  !> independent oracle for the self-tests (test_beltrami_solve / test_beltrami_precond).
  subroutine op_bind_iter(op, NN, dMA, dMD, mu)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN
    real(rkind), intent(in) :: dMA(0:NN, 0:NN), dMD(0:NN, 0:NN), mu
    integer :: nnz, ierr

    op%backend = BK_ITER
    op%NN = NN

    if (allocated(op%dof_mode)) deallocate(op%dof_mode)
    allocate(op%dof_mode(NN))
    call build_dof_mode(op%vvol, NN, op%dof_mode)

    call samemn_csr_count(NN, dMA, dMD, mu, op%dof_mode, nnz)
    if (allocated(op%pval)) deallocate(op%pval, op%pja, op%pia)
    allocate(op%pval(nnz), op%pja(nnz), op%pia(NN + 1))
    call build_samemn_csr(NN, dMA, dMD, mu, op%dof_mode, op%pval, op%pja, op%pia)

    call block_lu_release(op%blu)
    call block_lu_factor(NN, op%pval, op%pja, op%pia, op%dof_mode, mn, op%blu, ierr)
  end subroutine op_bind_iter

  !> \brief Bind the iterative backend WITHOUT forming the dense dMA/dMD: build the EXACT same-(m,n)
  !> preconditioner from the diagonal metric integrals (build_samemn_csr_diag) and factor it with an
  !> exact per-mode block LU. This is the memory-optimal path — only the diagonal metric-integral
  !> tensors (O(mn*Lrad^2)) and the block factor are stored, never the O(NAdof^2) dense matrix.
  !>
  !> The caller must have filled the diagonal tensors for this volume first, via
  !> volume_integrate_chebyshev(..., diag_only=.true.). Stellarator-symmetric only
  !> (op_select_backend keeps non-symmetric cases on the dense path).
  subroutine op_bind_iter_sparse(op, NN, Lcoord, mu)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN
    logical, intent(in) :: Lcoord
    real(rkind), intent(in) :: mu
    integer :: nnz, nnzmax, ierr, ii, mode_size

    op%backend = BK_ITER
    op%NN = NN

    if (allocated(op%dof_mode)) deallocate(op%dof_mode)
    allocate(op%dof_mode(NN))
    call build_dof_mode(op%vvol, NN, op%dof_mode)

    ! Exact structural upper bound: every retained entry couples degrees of freedom in the same
    ! Fourier-mode block. This avoids both guessed capacities and an abort on larger blocks.
    nnzmax = count(op%dof_mode == 0)
    do ii = 1, mn
      mode_size = count(op%dof_mode == ii)
      nnzmax = nnzmax + mode_size * mode_size
    end do
    if (allocated(op%pval)) deallocate(op%pval, op%pja, op%pia)
    allocate(op%pval(nnzmax), op%pja(nnzmax), op%pia(NN + 1))

    call build_samemn_csr_diag(op%vvol, NN, Lcoord, mu, op%pval, op%pja, op%pia, nnz, ierr)
    if (ierr /= 0) then
      call block_lu_release(op%blu)
      op%NN = 0
      return
    end if

    call block_lu_release(op%blu)
    call block_lu_factor(NN, op%pval, op%pja, op%pia, op%dof_mode, mn, op%blu, ierr)
  end subroutine op_bind_iter_sparse

  !> \brief Iterative solve A x = b (A = dMA - mu*dMD) via matrix-free right-preconditioned GMRES,
  !> using op_apply for the matvec and the operator's same-(m,n) block LU as the preconditioner.
  !> On entry x is the initial guess (warm start); on exit the solution. Caller supplies the
  !> per-thread matvec scratch (pws, lm) and storage slot (aslot). its/ierr report GMRES progress.
  subroutine op_solve_iter(op, mu, Lcoord, aslot, b, x, pws, lm, restart, maxiter, rtol, its, resid, ierr)
    type(beltrami_operator), intent(inout) :: op
    real(rkind), intent(in) :: mu, rtol
    logical, intent(in) :: Lcoord
    integer, intent(in) :: aslot, restart, maxiter
    real(rkind), intent(in) :: b(1:op%NN)
    real(rkind), intent(inout) :: x(1:op%NN)
    type(perturbed_rhs_workspace), intent(inout) :: pws
    type(matvec_multipliers), intent(inout) :: lm
    integer, intent(out) :: its, ierr
    real(rkind), intent(out) :: resid

    ! The (geometry-fixed) metric must already be in pws%guvijsave with Lsavedguvij=.true. — call
    ! op_prepare_metric(op, Lcoord, pws) once per volume before solving. intghs then reuses it
    ! across every matvec instead of recomputing the FFT-heavy metric each application.
    call gmres_solve(op%NN, b, x, mv, pc, restart, maxiter, rtol, 0.0_rkind, its, resid, ierr)

  contains

    subroutine mv(v, av)
      real(rkind), intent(in) :: v(:)
      real(rkind), intent(out) :: av(:)
      call op_apply(op, mu, Lcoord, aslot, v, av, pws, lm)
    end subroutine mv

    subroutine pc(v, z)
      real(rkind), intent(in) :: v(:)
      real(rkind), intent(out) :: z(:)
      call block_lu_apply(op%blu, v, z)
    end subroutine pc

  end subroutine op_solve_iter

  !> \brief Compute the geometry-fixed metric once into pws%guvijsave and mark it saved, so that
  !> subsequent intghs calls (matvecs) reuse it instead of recomputing the FFT-heavy metric. Must
  !> be called once per volume before op_solve_iter / op_solve_multi_iter / op_apply.
  subroutine op_prepare_metric(op, Lcoord, pws)
    type(beltrami_operator), intent(in) :: op
    logical, intent(in) :: Lcoord
    type(perturbed_rhs_workspace), intent(inout) :: pws

    dBdX%L = .false.
    dBdX%vol = op%vvol
    call compute_guvijsave(Iquad(op%vvol), op%vvol, 0, 1, Nt, Nz, Lcoord, &
        pws%guvijsave, pws%Rij, pws%Zij, pws%sg, pws%inv_jac, pws%guvij)
    Lsavedguvij = .true.
  end subroutine op_prepare_metric

  !> \brief Solve nb right-hand sides (columns of rhs, leading dim ldr) against the iterative
  !> operator, in place (rhs column -> its solution). The metric is prepared once; each column
  !> warm-starts from the previous column's solution, so after the first (cold) column the spectrally
  !> similar neighbours converge in far fewer iterations — the reuse needed to keep the Jacobian's
  !> many-RHS solve affordable. its_tot sums the iterations; ierr_any is nonzero if any column failed.
  subroutine op_solve_multi_iter(op, mu, Lcoord, aslot, ldr, rhs, nb, pws, lm, restart, maxiter, rtol, its_tot, ierr_any)
    type(beltrami_operator), intent(inout) :: op
    real(rkind), intent(in) :: mu, rtol
    logical, intent(in) :: Lcoord
    integer, intent(in) :: aslot, ldr, nb, restart, maxiter
    real(rkind), intent(inout) :: rhs(1:ldr, 1:nb)
    type(perturbed_rhs_workspace), intent(inout) :: pws
    type(matvec_multipliers), intent(inout) :: lm
    integer, intent(out) :: its_tot, ierr_any

    real(rkind) :: xcol(1:op%NN), resid
    integer :: ic, its, ierr

    call op_prepare_metric(op, Lcoord, pws)

    xcol(1:op%NN) = 0.0_rkind ! first column: cold start
    its_tot = 0
    ierr_any = 0
    do ic = 1, nb
      call gmres_solve(op%NN, rhs(1:op%NN, ic), xcol, mv, pc, restart, maxiter, rtol, 0.0_rkind, its, resid, ierr)
      rhs(1:op%NN, ic) = xcol(1:op%NN) ! store solution; xcol carries to the next column as warm start
      its_tot = its_tot + its
      if (ierr /= 0) ierr_any = ierr
    end do

  contains

    subroutine mv(v, av)
      real(rkind), intent(in) :: v(:)
      real(rkind), intent(out) :: av(:)
      call op_apply(op, mu, Lcoord, aslot, v, av, pws, lm)
    end subroutine mv

    subroutine pc(v, z)
      real(rkind), intent(in) :: v(:)
      real(rkind), intent(out) :: z(:)
      call block_lu_apply(op%blu, v, z)
    end subroutine pc

  end subroutine op_solve_multi_iter

  !> \brief Publish the factor from the owning rank to all ranks.
  subroutine op_bcast(op, NN, cpu_id)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN, cpu_id

    integer :: ierr

    call MPI_BCAST(op%mat(0:NN, 0:NN), (NN + 1) ** 2, MPI_DOUBLE_PRECISION, cpu_id, MPI_COMM_SPEC, ierr)
    call MPI_BCAST(op%ipivot(0:NN), NN + 1, MPI_INTEGER, cpu_id, MPI_COMM_SPEC, ierr)
  end subroutine op_bcast

  !> \brief Move src's storage into dst without copying; dst is released first, src comes back empty.
  subroutine op_move(src, dst)
    type(beltrami_operator), intent(inout) :: src, dst

    call op_release(dst)
    dst%vvol = src%vvol
    dst%NN = src%NN
    dst%backend = src%backend
    dst%bordered = src%bordered
    if (allocated(src%mat)) call move_alloc(src%mat, dst%mat)
    if (allocated(src%ipivot)) call move_alloc(src%ipivot, dst%ipivot)
    if (allocated(src%rwork)) call move_alloc(src%rwork, dst%rwork)
    if (allocated(src%iwork)) call move_alloc(src%iwork, dst%iwork)
    if (allocated(src%pval)) call move_alloc(src%pval, dst%pval)
    if (allocated(src%pja)) call move_alloc(src%pja, dst%pja)
    if (allocated(src%pia)) call move_alloc(src%pia, dst%pia)
    if (allocated(src%dof_mode)) call move_alloc(src%dof_mode, dst%dof_mode)
    dst%blu%n = src%blu%n
    dst%blu%nb = src%blu%nb
    if (allocated(src%blu%bsize)) call move_alloc(src%blu%bsize, dst%blu%bsize)
    if (allocated(src%blu%doff)) call move_alloc(src%blu%doff, dst%blu%doff)
    if (allocated(src%blu%dofs)) call move_alloc(src%blu%dofs, dst%blu%dofs)
    if (allocated(src%blu%foff)) call move_alloc(src%blu%foff, dst%blu%foff)
    if (allocated(src%blu%poff)) call move_alloc(src%blu%poff, dst%blu%poff)
    if (allocated(src%blu%fac)) call move_alloc(src%blu%fac, dst%blu%fac)
    if (allocated(src%blu%piv)) call move_alloc(src%blu%piv, dst%blu%piv)
    call op_release(src) ! storage is already moved out; this just resets the scalars
    src%backend = BK_DENSE
    src%bordered = .false.
  end subroutine op_move

  !> \brief Drop volume vvol's cached factorization. Called by dfp100 before re-solving the volume,
  !> so a stale (previous-geometry) factor can never survive into a JVP batch. No-op if the cache
  !> was never populated.
  subroutine op_cache_invalidate(vvol)
    integer, intent(in) :: vvol

    if (.not. allocated(op_cache)) return
    if (vvol < 1 .or. vvol > size(op_cache)) return
    call op_release(op_cache(vvol))
    op_cache_ok(vvol) = .false.
  end subroutine op_cache_invalidate

  !> \brief Deposit op's factorization (built at multiplier lmu) in volume vvol's cache slot.
  !> Storage is moved, so op comes back empty (a following op_release is a harmless no-op).
  !> Does nothing unless Lbeltrami_factor_cache is set.
  subroutine op_cache_put(vvol, lmu, op)
    integer, intent(in) :: vvol
    real(rkind), intent(in) :: lmu
    type(beltrami_operator), intent(inout) :: op

    integer :: i

    if (.not. Lbeltrami_factor_cache) return
    if (allocated(op_cache)) then
      if (size(op_cache) /= Mvol) then ! a different equilibrium was loaded into this process
        do i = 1, size(op_cache)
          call op_release(op_cache(i))
        end do
        deallocate(op_cache, op_cache_mu, op_cache_ok)
      end if
    end if
    if (.not. allocated(op_cache)) then
      allocate(op_cache(Mvol), op_cache_mu(Mvol), op_cache_ok(Mvol))
      op_cache_ok(1:Mvol) = .false.
      op_cache_mu(1:Mvol) = 0.0_rkind
    end if
    call op_move(op, op_cache(vvol))
    op_cache_mu(vvol) = lmu
    op_cache_ok(vvol) = .true.
  end subroutine op_cache_put

  !> \brief Take volume vvol's cached factorization into op, if a live one matching (lmu, NN)
  !> exists. Storage is moved and the slot marked consumed -- the caller returns the handle with
  !> op_cache_put when done. Returns .false. (and the caller builds the factorization itself) when
  !> the cache is off, empty, or stale; a (mu, NAdof) mismatch drops the stale slot.
  logical function op_cache_take(vvol, lmu, NN, op) result(hit)
    integer, intent(in) :: vvol, NN
    real(rkind), intent(in) :: lmu
    type(beltrami_operator), intent(inout) :: op

    hit = .false.
    if (.not. Lbeltrami_factor_cache) return
    if (.not. allocated(op_cache)) return
    if (vvol < 1 .or. vvol > size(op_cache)) return
    if (.not. op_cache_ok(vvol)) return
    if (op_cache(vvol)%NN /= NN .or. op_cache_mu(vvol) /= lmu) then
      call op_cache_invalidate(vvol)
      return
    end if
    call op_move(op_cache(vvol), op)
    op_cache_ok(vvol) = .false.
    hit = .true.
  end function op_cache_take

end module beltrami_operator_mod
