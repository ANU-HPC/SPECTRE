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
  use errors_mod, only: fatal
  use allglobal_mod, only: MPI_COMM_SPEC, mn, Nt, Nz, Iquad, dBdX, Lsavedguvij, NAdof
  use input_list_mod, only: Lrad, Lbeltramisolver, nadofgmres, nadofgmres_sing, &
      enforce_stell_sym, Igeometry
  use memory_allocation_mod, only: sallocate
  use perturbed_workspace_mod, only: perturbed_rhs_workspace, matvec_multipliers
  use intghs_mod, only: intghs
  use matrices_mod, only: mtrxhs_full
  use packing_mod, only: packab_matvec
  use metric_mod, only: compute_guvijsave
  use beltrami_sparse_mod, only: build_dof_mode, samemn_csr_count, build_samemn_csr, build_samemn_csr_diag
  use gmres_mod, only: gmres_solve, block_lu, block_lu_factor, block_lu_apply

  implicit none(type, external)

  private
  public :: beltrami_operator, BK_DENSE, BK_ITER
  public :: op_alloc, op_release, op_factor_prepared, op_factor_copy
  public :: op_solve, op_solve_refine, op_solve_multi, op_bcast, op_apply
  public :: op_solve_derivative_rhs_batch
  public :: op_bind_iter, op_bind_iter_sparse, op_solve_iter, op_select_backend, op_apply_ad
  public :: op_prepare_metric, op_solve_multi_iter
  public :: op_bind_iter_identity, op_is_iterative, op_is_bordered, op_is_empty, op_dimension, op_bound_to
  public :: op_move, op_set_row_correction, op_has_row_correction

  external :: DCOPY, DGETRF, DGETRS, DGERFS, DGECON

  integer, parameter :: BK_DENSE = 1 !< dense LU backend (LAPACK)
  integer, parameter :: BK_ITER = 2 !< iterative backend (matrix-free GMRES + same-(m,n) block-LU preconditioner)

  !> Dense LAPACK factor and refinement work arrays.
  type :: dense_factor_payload
    logical :: bordered = .false. !< Lconstraint==2 augmented (NN+1)x(NN+1) system
    real(rkind), allocatable :: mat(:, :) !< dense factor storage
    integer, allocatable :: ipivot(:) !< pivots from DGETRF
    real(rkind), allocatable :: rwork(:) !< real workspace for DGECON/DGERFS
    integer, allocatable :: iwork(:) !< integer workspace for DGECON/DGERFS
  end type dense_factor_payload

  !> Matrix-free backend's same-(m,n) CSR preconditioner and exact block-diagonal LU.
  type :: iterative_preconditioner_payload
    real(rkind), allocatable :: pval(:) !< CSR values of the same-(m,n) approximation
    integer, allocatable :: pja(:), pia(:) !< CSR columns, row pointers
    integer, allocatable :: dof_mode(:) !< DOF -> Fourier-mode map
    type(block_lu) :: blu !< exact block-diagonal LU of the same-(m,n) preconditioner
  end type iterative_preconditioner_payload

  !> Immutable compact correction subtracted from selected rows of the A-action.
  !>
  !> The payload is an owned snapshot. It is read-only during applications and carries no
  !> information about the correction's physical origin.
  type :: row_correction_payload
    integer, allocatable :: rows(:) !< corrected row indices in 1:NN
    real(rkind), allocatable :: values(:, :) !< compact corrected rows, shaped (nrows, NN)
  end type row_correction_payload

  !> \brief Opaque handle for one volume's Beltrami linear operator A - mu*D (A only in vacuum).
  !>
  !> Exactly one payload is allocated for a bound operator. Callers borrow the handle through
  !> solve/apply procedures, transfer ownership only with op_move, and finish with op_release.
  type beltrami_operator
    private
    integer :: vvol = 0 !< volume this operator is bound to
    integer :: NN = 0 !< NAdof(vvol)
    integer :: backend = BK_DENSE
    type(dense_factor_payload), allocatable :: dense
    type(iterative_preconditioner_payload), allocatable :: iterative
    type(row_correction_payload), allocatable :: correction
  end type beltrami_operator

contains

  !> \brief Select the linear-solver backend for volume vvol from the input flags. Returns BK_DENSE
  !> unless the matrix-free GMRES path is enabled (Lbeltramisolver /= 0), the volume is large enough
  !> (Lbeltramisolver==1 uses the nadofgmres threshold; ==2 forces iterative), and the case is
  !> stellarator-symmetric (the sparse preconditioner is stell-sym only). The helicity constraint
  !> (Lconstraint==2) iterates mu against the unbordered operator A - mu*D, so it is backend-generic
  !> here; only the dense Jacobian path forms the bordered system, and it allocates its operators
  !> dense directly (never through this selector).
  integer function op_select_backend(vvol) result(backend)
    integer, intent(in) :: vvol

    backend = BK_DENSE
    if (Lbeltramisolver == 0) return
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

  !> \brief Bind only the volume identity and backend tag, without allocating a payload.
  !>
  !> This is used by matrix-free apply tests and by non-owning MPI ranks that only need the tag.
  subroutine op_bind_identity(op, vvol, NN, backend)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: vvol, NN, backend

    call op_release(op)
    op%vvol = vvol
    op%NN = NN
    op%backend = backend
  end subroutine op_bind_identity

  !> Bind a matrix-free identity/tag without allocating its preconditioner payload.
  subroutine op_bind_iter_identity(op, vvol, NN)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: vvol, NN

    call op_bind_identity(op, vvol, NN, BK_ITER)
  end subroutine op_bind_iter_identity

  !> Return whether the handle selects the matrix-free backend.
  logical function op_is_iterative(op) result(is_iterative)
    type(beltrami_operator), intent(in) :: op

    is_iterative = op%backend == BK_ITER
  end function op_is_iterative

  !> Return whether the handle owns a dense factor of the bordered (helicity) system.
  logical function op_is_bordered(op) result(is_bordered)
    type(beltrami_operator), intent(in) :: op

    is_bordered = .false.
    if (allocated(op%dense)) is_bordered = op%dense%bordered
  end function op_is_bordered

  !> Return whether the handle owns no payload and has no bound identity.
  logical function op_is_empty(op) result(is_empty)
    type(beltrami_operator), intent(in) :: op

    is_empty = op%vvol == 0 .and. op%NN == 0 &
        .and. .not. allocated(op%dense) .and. .not. allocated(op%iterative) &
        .and. .not. allocated(op%correction)
  end function op_is_empty

  !> Return whether the handle owns a compact row-correction payload.
  logical function op_has_row_correction(op) result(has_correction)
    type(beltrami_operator), intent(in) :: op

    has_correction = allocated(op%correction)
  end function op_has_row_correction

  !> Return the bound operator dimension, or zero for an empty handle.
  integer function op_dimension(op) result(NN)
    type(beltrami_operator), intent(in) :: op

    NN = op%NN
  end function op_dimension

  !> Return whether the handle is bound to this volume and dimension.
  logical function op_bound_to(op, vvol, NN) result(matches)
    type(beltrami_operator), intent(in) :: op
    integer, intent(in) :: vvol, NN

    matches = op%vvol == vvol .and. op%NN == NN
  end function op_bound_to

  !> \brief Allocate the operator's factor storage for volume vvol with NN = NAdof(vvol).
  subroutine op_alloc(op, vvol, NN, bordered)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: vvol, NN
    logical, intent(in) :: bordered

    call op_bind_identity(op, vvol, NN, BK_DENSE)
    allocate(op%dense)
    op%dense%bordered = bordered

    call sallocate(op%dense%mat, 0, NN, 0, NN, 0.0_rkind)
    call sallocate(op%dense%ipivot, 0, NN, 0)
    call sallocate(op%dense%rwork, 1, 4 * NN, 0.0_rkind)
    call sallocate(op%dense%iwork, 1, NN, 0)
  end subroutine op_alloc

  !> \brief Release the operator's storage.
  subroutine op_release(op)
    type(beltrami_operator), intent(inout) :: op

    if (allocated(op%dense)) deallocate(op%dense)
    if (allocated(op%iterative)) deallocate(op%iterative)
    if (allocated(op%correction)) deallocate(op%correction)
    op%vvol = 0
    op%NN = 0
    op%backend = BK_DENSE
  end subroutine op_release

  !> Attach an immutable compact row correction to an already bound operator.
  !>
  !> Each application subsequently subtracts ``values(i,:) . x`` from A-row ``rows(i)``.
  !> The inputs are copied, so callers may release or modify their arrays after this call.
  !> With ``status`` present, invalid input is reported as 1 (unbound), 2 (empty), or 3
  !> (out-of-range row), or 4 (shape mismatch); without it, invalid input follows the repository's
  !> fatal validation path.
  subroutine op_set_row_correction(op, nrows, rows, values, status)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: nrows
    integer, intent(in) :: rows(:)
    real(rkind), intent(in) :: values(:, :)
    integer, optional, intent(out) :: status

    if (present(status)) status = 0
    if (op%NN == 0) then
      if (present(status)) then
        status = 1
      else
        call fatal(.true., __FILE__, "op_set_row_correction: operator is not bound")
      end if
      return
    end if
    if (nrows < 1) then
      if (present(status)) then
        status = 2
      else
        call fatal(.true., __FILE__, "op_set_row_correction: correction is empty")
      end if
      return
    end if
    if (size(rows) /= nrows .or. size(values, 1) /= nrows .or. size(values, 2) /= op%NN) then
      if (present(status)) then
        status = 4
      else
        call fatal(.true., __FILE__, "op_set_row_correction: payload shape mismatch")
      end if
      return
    end if
    if (minval(rows) < 1 .or. maxval(rows) > op%NN) then
      if (present(status)) then
        status = 3
      else
        call fatal(.true., __FILE__, "op_set_row_correction: row index is outside 1:NN")
      end if
      return
    end if

    if (allocated(op%correction)) deallocate(op%correction)
    allocate(op%correction)
    op%correction%rows = rows(1:nrows)
    op%correction%values = values(1:nrows, 1:op%NN)
  end subroutine op_set_row_correction

  !> \brief Factor a caller-prepared matrix (already A - mu*D, plus border row/column when
  !> bordered) in place and keep the factor + pivots in the operator. Amat is destroyed.
  subroutine op_factor_prepared(op, NN, Amat, info)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN
    real(rkind), intent(inout) :: Amat(0:NN, 0:NN)
    integer, intent(out) :: info

    if (op%dense%bordered) then
      call DGETRF(NN + 1, NN + 1, Amat(0, 0), NN + 1, op%dense%ipivot, info)
      call DCOPY( (NN + 1) * (NN + 1) , Amat, 1, op%dense%mat, 1)
    else
      call DGETRF(NN, NN, Amat(1, 1), NN + 1, op%dense%ipivot(1:NN), info)
      op%dense%mat(1:NN, 1:NN) = Amat(1:NN, 1:NN)
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

    op%dense%mat(1:NN, 1:NN) = matrix(1:NN, 1:NN)
    call DGETRF(NN, NN, op%dense%mat(1, 1), NN + 1, op%dense%ipivot(1:NN), info_trf)

    anorm = maxval(sum(abs(matrix), 1))
    call DGECON('I', NN, op%dense%mat(1, 1), NN + 1, anorm, rcond, &
        op%dense%rwork, op%dense%iwork, info_con)
  end subroutine op_factor_copy

  !> \brief Back-substitute one right-hand side in place. x has length NN+1 for the bordered
  !> system and NN otherwise.
  subroutine op_solve(op, n, x, info)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: n
    real(rkind), intent(inout) :: x(1:n)
    integer, intent(out) :: info

    if (op%dense%bordered) then
      call DGETRS('N', n, 1, op%dense%mat(0, 0), n, op%dense%ipivot(0:n - 1), x(1), n, info)
    else
      call DGETRS('N', n, 1, op%dense%mat(1, 1), n + 1, op%dense%ipivot(1:n), x(1), n, info)
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
    call DGETRS('N', NN, nrhs, op%dense%mat(1, 1), NN + 1, op%dense%ipivot(1:NN), &
        sol(1, 1), NN, info_trs)
    call DGERFS('N', NN, nrhs, matrix(1, 1), NN, op%dense%mat(1, 1), NN + 1, &
        op%dense%ipivot(1:NN), rhs(1, 1), NN, sol(1, 1), NN, ferr, berr, &
        op%dense%rwork, op%dense%iwork, info_rfs)
  end subroutine op_solve_refine

  !> \brief Back-substitute a batch of nb right-hand sides in place; rhs holds one column of
  !> length ldr (the caller's leading dimension) per right-hand side.
  subroutine op_solve_multi(op, NN, ldr, rhs, nb, info)
    type(beltrami_operator), intent(in) :: op
    integer, intent(in) :: NN, ldr, nb
    real(rkind), intent(inout) :: rhs(1:ldr, 1:nb)
    integer, intent(out) :: info

    call DGETRS('N', NN, nb, op%dense%mat(1, 1), NN + 1, op%dense%ipivot(1:NN), &
        rhs(1, 1), ldr, info)
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

    real(rkind) :: dax0(0:op%NN), ddx0(0:op%NN)
    integer :: irow

    dBdX%L = .false. ! matvec needs the full operator incl. dMD and Lagrange coupling
    dBdX%vol = op%vvol
    ! Lsavedguvij is left to the caller: op_solve_iter computes the (fixed) metric once into
    ! pws%guvijsave and sets Lsavedguvij=.true. so intghs reuses it across all matvecs of a solve.
    ! A caller that has not prepared the metric must set Lsavedguvij=.false. (intghs then computes it).

    call packab_matvec(op%vvol, op%NN, x, aslot, lm)
    call intghs(Iquad(op%vvol), mn, op%vvol, Lrad(op%vvol), aslot, Nt, Nz, Lcoord, pws)
    call mtrxhs_full(op%vvol, mn, Lrad(op%vvol), dax0, ddx0, aslot, Lcoord, pws, lm)

    adx(1:op%NN) = dax0(1:op%NN)
    ddx(1:op%NN) = ddx0(1:op%NN)

    ! Apply only to A. The D-action and the block-LU preconditioner deliberately remain
    ! unchanged, while GMRES convergence is measured with this corrected true operator.
    if (allocated(op%correction)) then
      do irow = 1, size(op%correction%rows)
        adx(op%correction%rows(irow)) = adx(op%correction%rows(irow)) &
            - dot_product(op%correction%values(irow, 1:op%NN), x(1:op%NN))
      end do
    end if
  end subroutine op_apply_ad

  !> \brief Bind the iterative backend by EXTRACTING the same-(m,n) preconditioner block from the
  !> dense dMA/dMD and factoring it with the exact per-mode block LU. Production never forms those
  !> dense matrices and uses op_bind_iter_sparse instead; this extraction path is kept as the
  !> independent oracle for the self-tests (test_beltrami_solve / test_beltrami_precond).
  subroutine op_bind_iter(op, vvol, NN, dMA, dMD, mu)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: vvol, NN
    real(rkind), intent(in) :: dMA(0:NN, 0:NN), dMD(0:NN, 0:NN), mu
    integer :: nnz, ierr

    call op_bind_identity(op, vvol, NN, BK_ITER)
    allocate(op%iterative)

    allocate(op%iterative%dof_mode(NN))
    call build_dof_mode(vvol, NN, op%iterative%dof_mode)

    call samemn_csr_count(NN, dMA, dMD, mu, op%iterative%dof_mode, nnz)
    allocate(op%iterative%pval(nnz), op%iterative%pja(nnz), op%iterative%pia(NN + 1))
    call build_samemn_csr(NN, dMA, dMD, mu, op%iterative%dof_mode, &
        op%iterative%pval, op%iterative%pja, op%iterative%pia)

    call block_lu_factor(NN, op%iterative%pval, op%iterative%pja, op%iterative%pia, &
        op%iterative%dof_mode, mn, op%iterative%blu, ierr)
  end subroutine op_bind_iter

  !> \brief Bind the iterative backend WITHOUT forming the dense dMA/dMD: build the EXACT same-(m,n)
  !> preconditioner from the diagonal metric integrals (build_samemn_csr_diag) and factor it with an
  !> exact per-mode block LU. This is the memory-optimal path — only the diagonal metric-integral
  !> tensors (O(mn*Lrad^2)) and the block factor are stored, never the O(NAdof^2) dense matrix.
  !>
  !> The caller must have filled the diagonal tensors for this volume first, via
  !> volume_integrate_chebyshev(..., diag_only=.true.). Stellarator-symmetric only
  !> (op_select_backend keeps non-symmetric cases on the dense path).
  subroutine op_bind_iter_sparse(op, vvol, NN, Lcoord, mu)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: vvol, NN
    logical, intent(in) :: Lcoord
    real(rkind), intent(in) :: mu
    integer :: nnz, nnzmax, ierr, ii, mode_size

    call op_bind_identity(op, vvol, NN, BK_ITER)
    allocate(op%iterative)

    allocate(op%iterative%dof_mode(NN))
    call build_dof_mode(vvol, NN, op%iterative%dof_mode)

    ! Exact structural upper bound: every retained entry couples degrees of freedom in the same
    ! Fourier-mode block. This avoids both guessed capacities and an abort on larger blocks.
    nnzmax = count(op%iterative%dof_mode == 0)
    do ii = 1, mn
      mode_size = count(op%iterative%dof_mode == ii)
      nnzmax = nnzmax + mode_size * mode_size
    end do
    allocate(op%iterative%pval(nnzmax), op%iterative%pja(nnzmax), &
        op%iterative%pia(NN + 1))

    call build_samemn_csr_diag(vvol, NN, Lcoord, mu, op%iterative%pval, &
        op%iterative%pja, op%iterative%pia, nnz, ierr)
    if (ierr /= 0) then
      op%NN = 0
      return
    end if

    call block_lu_factor(NN, op%iterative%pval, op%iterative%pja, op%iterative%pia, &
        op%iterative%dof_mode, mn, op%iterative%blu, ierr)
  end subroutine op_bind_iter_sparse

  !> \brief Iterative solve A x = b (A = dMA - mu*dMD) via matrix-free right-preconditioned GMRES,
  !> using op_apply for the matvec and the operator's same-(m,n) block LU as the preconditioner.
  !> On entry x is the initial guess (warm start); on exit the solution. Caller supplies the
  !> per-thread matvec scratch (pws, lm) and storage slot (aslot). its/ierr report GMRES progress.
  subroutine op_solve_iter(op, mu, Lcoord, aslot, b, x, pws, lm, restart, maxiter, rtol, its, resid, ierr)
    type(beltrami_operator), intent(in) :: op
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
    call op_solve_iter_prepared(op, mu, Lcoord, aslot, b, x, pws, lm, restart, maxiter, rtol, &
        its, resid, ierr)
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
    type(beltrami_operator), intent(in) :: op
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
      call op_solve_iter_prepared(op, mu, Lcoord, aslot, rhs(1:op%NN, ic), xcol, &
          pws, lm, restart, maxiter, rtol, its, resid, ierr)
      rhs(1:op%NN, ic) = xcol(1:op%NN) ! store solution; xcol carries to the next column as warm start
      its_tot = its_tot + its
      if (ierr /= 0) ierr_any = ierr
    end do
  end subroutine op_solve_multi_iter

  !> Run GMRES against an already prepared metric; the shared kernel behind both iterative solves.
  subroutine op_solve_iter_prepared(op, mu, Lcoord, aslot, b, x, pws, lm, restart, maxiter, rtol, &
      its, resid, ierr)
    type(beltrami_operator), intent(in) :: op
    real(rkind), intent(in) :: mu, rtol
    logical, intent(in) :: Lcoord
    integer, intent(in) :: aslot, restart, maxiter
    real(rkind), intent(in) :: b(1:op%NN)
    real(rkind), intent(inout) :: x(1:op%NN)
    type(perturbed_rhs_workspace), intent(inout) :: pws
    type(matvec_multipliers), intent(inout) :: lm
    integer, intent(out) :: its, ierr
    real(rkind), intent(out) :: resid
    real(rkind) :: preconditioner_work(1:op%NN)

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
      call block_lu_apply(op%iterative%blu, v, z, preconditioner_work)
    end subroutine pc

  end subroutine op_solve_iter_prepared

  !> Solve one prepared derivative RHS batch through its operator's selected backend.
  !>
  !> The explicit-shape RHS is the caller's existing contiguous column storage. This adapter
  !> contains no allocation, copying, reshaping, factor preparation, communication, force
  !> assembly, or optimizer policy. It adds one normal procedure call around the established
  !> dense DGETRS or iterative GMRES batch. Status retains the post-review convention (0 success,
  !> 1 degraded iterative solve, 2 dense back-substitution failure); iterations is zero for the
  !> dense backend.
  subroutine op_solve_derivative_rhs_batch(op, mu, Lcoord, aslot, ldr, rhs, nrhs, pws, lm, &
      restart, maxiter, rtol, iterations, status)
    type(beltrami_operator), intent(in) :: op
    real(rkind), intent(in) :: mu, rtol
    logical, intent(in) :: Lcoord
    integer, intent(in) :: aslot, ldr, nrhs, restart, maxiter
    real(rkind), intent(inout) :: rhs(1:ldr, 1:nrhs)
    type(perturbed_rhs_workspace), intent(inout) :: pws
    type(matvec_multipliers), intent(inout) :: lm
    integer, intent(out) :: iterations, status

    integer :: ierr

    if (op%backend == BK_ITER) then
      call op_solve_multi_iter(op, mu, Lcoord, aslot, ldr, rhs, nrhs, pws, lm, &
          restart, maxiter, rtol, iterations, ierr)
      if (ierr == 0) then
        status = 0
      else
        status = 1
      end if
    else
      call op_solve_multi(op, op%NN, ldr, rhs, nrhs, ierr)
      iterations = 0
      if (ierr == 0) then
        status = 0
      else
        status = 2
      end if
    end if
  end subroutine op_solve_derivative_rhs_batch

  !> \brief Publish the factor from the owning rank to all ranks.
  subroutine op_bcast(op, NN, cpu_id)
    type(beltrami_operator), intent(inout) :: op
    integer, intent(in) :: NN, cpu_id

    integer :: ierr

    call MPI_BCAST(op%dense%mat(0:NN, 0:NN), (NN + 1) ** 2, &
        MPI_DOUBLE_PRECISION, cpu_id, MPI_COMM_SPEC, ierr)
    call MPI_BCAST(op%dense%ipivot(0:NN), NN + 1, MPI_INTEGER, cpu_id, MPI_COMM_SPEC, ierr)
  end subroutine op_bcast

  !> \brief Move src's complete payload into dst without copying.
  !>
  !> dst is released first and src comes back empty. This is the only supported ownership transfer;
  !> solve/apply routines merely borrow their handle.
  subroutine op_move(src, dst)
    type(beltrami_operator), intent(inout) :: src, dst

    call op_release(dst)
    dst%vvol = src%vvol
    dst%NN = src%NN
    dst%backend = src%backend
    if (allocated(src%dense)) call move_alloc(src%dense, dst%dense)
    if (allocated(src%iterative)) call move_alloc(src%iterative, dst%iterative)
    if (allocated(src%correction)) call move_alloc(src%correction, dst%correction)
    src%vvol = 0
    src%NN = 0
    src%backend = BK_DENSE
  end subroutine op_move

end module beltrami_operator_mod
