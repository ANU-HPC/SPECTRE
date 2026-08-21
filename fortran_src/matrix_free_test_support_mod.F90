! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> Numerical reference oracles and setup/teardown used only by matrix-free tests.
module matrix_free_test_support_mod

  use allglobal_mod, only: DDttcc, DDtzcc, DDzzcc, DToocc, Iquad, &
      Lcoordinatesingularity, Lplasmaregion, Lsavedguvij, Lvacuumregion, &
      Mvol, NAdof, Nt, Ntz, Nz, TDstsc, TDszsc, TTssss, dBdX, dMA, dMD, mn
  use chebyshev_mod, only: volume_integrate_chebyshev
  use input_list_mod, only: Igeometry, Lfreebound, Lrad, Mpol, Nvol, Lfreebsolver
  use vc_matrix_mod, only: vc_attach_operator_correction, vc_subtract_dvc
  use intghs_mod, only: intghs_workspace_destroy, intghs_workspace_init
  use matrices_mod, only: matrix
  use memory_mod, only: allocate_Beltrami_matrices, allocate_geometry_matrices, &
      deallocate_Beltrami_matrices, deallocate_geometry_matrices
  use fft_backend_mod, only: fft_init
  use perturbed_workspace_mod, only: alloc_matvec_multipliers, &
      alloc_perturbed_workspace, dealloc_matvec_multipliers, &
      dealloc_perturbed_workspace, matvec_multipliers, perturbed_rhs_workspace
  use precisions_mod, only: rkind
  use volume_mod, only: lregion

  implicit none(type, external)

  private
  public :: beltrami_matvec_selftest, beltrami_row_correction_selftest
  public :: beltrami_row_correction_solve_selftest, beltrami_solve_selftest
  public :: block_lu_concurrency_selftest, diag_integrals_selftest, gmres_selftest
  public :: get_test_omp_max_threads, samemn_precond_compare, set_test_omp_threads

contains

  !> Set the OpenMP team size used by subsequent calls, without exceeding the preset-time maximum.
  subroutine set_test_omp_threads(nthreads)
    use omp_lib, only: omp_set_num_threads
    integer, intent(in) :: nthreads

    call omp_set_num_threads(nthreads)
  end subroutine set_test_omp_threads

  !> Return the current OpenMP maximum team size so tests can restore it.
  subroutine get_test_omp_max_threads(nthreads)
    use omp_lib, only: omp_get_max_threads
    integer, intent(out) :: nthreads

    nthreads = omp_get_max_threads()
  end subroutine get_test_omp_max_threads

  !> Build the dense Beltrami oracle shared by the new operator/preconditioner tests.
  subroutine setup_dense_beltrami_reference(lvol, ll, NN, lquad)
    integer, intent(in) :: lvol
    integer, intent(out) :: ll, NN, lquad

    call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, &
        Lvacuumregion)
    ll = Lrad(lvol)
    NN = NAdof(lvol)
    lquad = Iquad(lvol)

    dBdX%L = .false.
    Lsavedguvij = .false.
    call fft_init(Nt, Nz)
    call allocate_geometry_matrices(lvol, .true., Ntz)
    call allocate_Beltrami_matrices(lvol, .true.)
    call intghs_workspace_init(lvol, Ntz)
    call volume_integrate_chebyshev(lquad, mn, lvol, ll, Nt, Nz)
    ! dMA/dMD are left uncorrected here: samemn_precond_compare validates the (uncorrected)
    ! preconditioner blocks against this assembly. The operator-level selftests subtract the
    ! virtual-casing rows themselves, after their preconditioner binds.
    call matrix(lvol, mn, ll, dBdX%L)
  end subroutine setup_dense_beltrami_reference

  !> Release every allocation owned by ``setup_dense_beltrami_reference``.
  subroutine teardown_dense_beltrami_reference()
    call intghs_workspace_destroy()
    call deallocate_Beltrami_matrices(.true.)
    call deallocate_geometry_matrices(.true.)
  end subroutine teardown_dense_beltrami_reference

  !> Allocate the matrix-free apply workspace shared by operator solve tests.
  subroutine setup_operator_workspace(ll, lquad, pws, lm)
    integer, intent(in) :: ll, lquad
    type(perturbed_rhs_workspace), intent(inout) :: pws
    type(matvec_multipliers), intent(inout) :: lm

    integer :: max_lldof

    if (Lcoordinatesingularity) then
      max_lldof = (ll - mod(ll, 2)) / 2
    else
      max_lldof = ll
    end if
    call alloc_perturbed_workspace(pws, Ntz, mn, lquad, ll, Mpol, max_lldof)
    call alloc_matvec_multipliers(lm, mn)
  end subroutine setup_operator_workspace

  !> Release every allocation owned by ``setup_operator_workspace``.
  subroutine teardown_operator_workspace(pws, lm)
    type(perturbed_rhs_workspace), intent(inout) :: pws
    type(matvec_multipliers), intent(inout) :: lm

    call dealloc_matvec_multipliers(lm)
    call dealloc_perturbed_workspace(pws)
  end subroutine teardown_operator_workspace

  !> \brief Self-test for the diagonal-restricted metric-integral build.
  !>
  !> Runs volume_integrate_chebyshev over all mn^2 mode pairs, snapshots the diagonal (ii == jj)
  !> entries of the seven stellarator-symmetric tensors, then re-runs it with diag_only=.true. and
  !> compares. The diagonal restriction only changes which mode pairs are visited -- both passes run
  !> the same per-pair code (the Zernike loop body, or the Chebyshev GEMM against the same operands)
  !> -- so the two agree to round-off; a larger difference means the index decode or the tensor slot
  !> convention is wrong. Returns max|delta| over all tensors, radial indices and modes.
  !> Requires geometry to be set up (call after a force evaluation).
  !> Used by tests/wrapper/test_diag_integrals.py.
  subroutine diag_integrals_selftest(lvol, max_absdiff)
    integer, intent(in) :: lvol
    real(rkind), intent(out) :: max_absdiff

    integer :: ll, ii, lldof
    real(rkind), allocatable :: sDToocc(:, :, :), sTTssss(:, :, :), sTDstsc(:, :, :), sTDszsc(:, :, :)
    real(rkind), allocatable :: sDDttcc(:, :, :), sDDtzcc(:, :, :), sDDzzcc(:, :, :)

    call lregion(lvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
    ll = Lrad(lvol)

    dBdX%L = .false.
    Lsavedguvij = .false.
    call fft_init(Nt, Nz)
    call allocate_geometry_matrices(lvol, .true., Ntz) ! full mn x mn tensors
    call intghs_workspace_init(lvol, Ntz)

    ! full pass, then keep the diagonal
    call volume_integrate_chebyshev(Iquad(lvol), mn, lvol, ll, Nt, Nz)

    lldof = ubound(DToocc, 1)
    allocate(sDToocc(0:lldof, 0:lldof, 1:mn), sTTssss(0:lldof, 0:lldof, 1:mn), &
        sTDstsc(0:lldof, 0:lldof, 1:mn), sTDszsc(0:lldof, 0:lldof, 1:mn), &
        sDDttcc(0:lldof, 0:lldof, 1:mn), sDDtzcc(0:lldof, 0:lldof, 1:mn), &
        sDDzzcc(0:lldof, 0:lldof, 1:mn))
    do ii = 1, mn
      sDToocc(:, :, ii) = DToocc(:, :, ii, ii)
      sTTssss(:, :, ii) = TTssss(:, :, ii, ii)
      sTDstsc(:, :, ii) = TDstsc(:, :, ii, ii)
      sTDszsc(:, :, ii) = TDszsc(:, :, ii, ii)
      sDDttcc(:, :, ii) = DDttcc(:, :, ii, ii)
      sDDtzcc(:, :, ii) = DDtzcc(:, :, ii, ii)
      sDDzzcc(:, :, ii) = DDzzcc(:, :, ii, ii)
    end do

    ! diagonal-only pass into the same (full) arrays: tensor_jslot returns jj, so it writes (ii, ii)
    Lsavedguvij = .false.
    call volume_integrate_chebyshev(Iquad(lvol), mn, lvol, ll, Nt, Nz, diag_only=.true.)

    max_absdiff = 0.0_rkind
    do ii = 1, mn
      max_absdiff = max(max_absdiff, maxval(abs(DToocc(:, :, ii, ii) - sDToocc(:, :, ii))))
      max_absdiff = max(max_absdiff, maxval(abs(TTssss(:, :, ii, ii) - sTTssss(:, :, ii))))
      max_absdiff = max(max_absdiff, maxval(abs(TDstsc(:, :, ii, ii) - sTDstsc(:, :, ii))))
      max_absdiff = max(max_absdiff, maxval(abs(TDszsc(:, :, ii, ii) - sTDszsc(:, :, ii))))
      max_absdiff = max(max_absdiff, maxval(abs(DDttcc(:, :, ii, ii) - sDDttcc(:, :, ii))))
      max_absdiff = max(max_absdiff, maxval(abs(DDtzcc(:, :, ii, ii) - sDDtzcc(:, :, ii))))
      max_absdiff = max(max_absdiff, maxval(abs(DDzzcc(:, :, ii, ii) - sDDzzcc(:, :, ii))))
    end do

    deallocate(sDToocc, sTTssss, sTDstsc, sTDszsc, sDDttcc, sDDtzcc, sDDzzcc)
    call intghs_workspace_destroy()
    call deallocate_geometry_matrices(.true.)
  end subroutine diag_integrals_selftest

  !> \brief Self-test for the matrix-free Beltrami operator apply (op_apply). For volume lvol of the
  !> currently loaded equilibrium it builds the dense dMA/dMD, then compares op_apply against
  !> matmul(dMA - testmu*dMD, x) over nvec deterministic input vectors. Returns the maximum relative
  !> difference (max|y_mf - y_dense| / max|y_dense|). Requires geometry to be set up (call after a
  !> force evaluation). Used by tests/wrapper/test_beltrami_matvec.py.
  subroutine beltrami_matvec_selftest(lvol, nvec, testmu, max_reldiff)
    use beltrami_operator_mod, only: beltrami_operator, op_apply, op_bind_iter_identity
    integer, intent(in) :: lvol, nvec
    real(rkind), intent(in) :: testmu
    real(rkind), intent(out) :: max_reldiff

    integer :: NN, ll, ivec, i, aslot, lquad
    real(rkind) :: maxdiff, maxnorm
    real(rkind), allocatable :: x(:), y_dense(:), y_mf(:)
    type(beltrami_operator) :: op
    type(perturbed_rhs_workspace) :: pws
    type(matvec_multipliers) :: lm

    call setup_dense_beltrami_reference(lvol, ll, NN, lquad)
    call setup_operator_workspace(ll, lquad, pws, lm)

    call op_bind_iter_identity(op, lvol, NN)
    ! direct free-boundary solve: the true vacuum operator is (dMA - dVC). Correct the dense
    ! reference and the operator under test identically, so the comparison validates the
    ! correction path of op_apply_ad at machine precision.
    if (Lfreebsolver == 1 .and. Lfreebound == 1 .and. lvol == Mvol) then
      call vc_subtract_dvc(dMA, NN)
      call vc_attach_operator_correction(op, NN)
    end if
    aslot = -1 ! serial storage slot in Ate/Aze/Ato/Azo

    allocate(x(1:NN), y_dense(1:NN), y_mf(1:NN))
    maxdiff = 0.0_rkind
    maxnorm = 0.0_rkind
    do ivec = 1, nvec
      do i = 1, NN
        x(i) = sin(0.1_rkind * i + 0.7_rkind * ivec) + 0.3_rkind * cos(0.03_rkind * i * ivec)
      end do
      y_dense(1:NN) = matmul(dMA(1:NN, 1:NN) - testmu * dMD(1:NN, 1:NN), x(1:NN))
      call op_apply(op, testmu, Lcoordinatesingularity, aslot, x, y_mf, pws, lm)
      maxdiff = max(maxdiff, maxval(abs(y_dense - y_mf)))
      maxnorm = max(maxnorm, maxval(abs(y_dense)))
    end do
    max_reldiff = maxdiff / max(maxnorm, tiny(1.0_rkind))

    deallocate(x, y_dense, y_mf)
    call teardown_operator_workspace(pws, lm)
    call teardown_dense_beltrami_reference()
  end subroutine beltrami_matvec_selftest

  !> Compare a synthetic compact row correction with an independently corrected dense action.
  !>
  !> The caller-owned values are overwritten after attachment to verify snapshot semantics.
  subroutine beltrami_row_correction_selftest(lvol, testmu, max_reldiff)
    use beltrami_operator_mod, only: beltrami_operator, op_apply, op_bind_iter_identity, &
        op_release, op_set_row_correction
    integer, intent(in) :: lvol
    real(rkind), intent(in) :: testmu
    real(rkind), intent(out) :: max_reldiff

    integer :: NN, ll, ivec, i, aslot, lquad
    integer :: rows(2)
    real(rkind) :: maxdiff, maxnorm
    real(rkind), allocatable :: correction(:, :), snapshot(:, :), x(:), y_dense(:), y_mf(:)
    type(beltrami_operator) :: op
    type(perturbed_rhs_workspace) :: pws
    type(matvec_multipliers) :: lm

    call setup_dense_beltrami_reference(lvol, ll, NN, lquad)
    call setup_operator_workspace(ll, lquad, pws, lm)

    rows = [1, NN]
    allocate(correction(2, NN), snapshot(2, NN), x(NN), y_dense(NN), y_mf(NN))
    do i = 1, NN
      correction(1, i) = 0.03_rkind * sin(0.17_rkind * i)
      correction(2, i) = -0.02_rkind * cos(0.11_rkind * i)
    end do
    snapshot = correction

    call op_bind_iter_identity(op, lvol, NN)
    call op_set_row_correction(op, 2, rows, correction)
    correction = huge(1.0_rkind)
    aslot = -1

    maxdiff = 0.0_rkind
    maxnorm = 0.0_rkind
    do ivec = 1, 4
      do i = 1, NN
        x(i) = sin(0.1_rkind * i + 0.7_rkind * ivec) + 0.3_rkind * cos(0.03_rkind * i * ivec)
      end do
      y_dense = matmul(dMA(1:NN, 1:NN) - testmu * dMD(1:NN, 1:NN), x)
      y_dense(rows(1)) = y_dense(rows(1)) - dot_product(snapshot(1, :), x)
      y_dense(rows(2)) = y_dense(rows(2)) - dot_product(snapshot(2, :), x)
      call op_apply(op, testmu, Lcoordinatesingularity, aslot, x, y_mf, pws, lm)
      maxdiff = max(maxdiff, maxval(abs(y_dense - y_mf)))
      maxnorm = max(maxnorm, maxval(abs(y_dense)))
    end do
    max_reldiff = maxdiff / max(maxnorm, tiny(1.0_rkind))

    call op_release(op)
    deallocate(correction, snapshot, x, y_dense, y_mf)
    call teardown_operator_workspace(pws, lm)
    call teardown_dense_beltrami_reference()
  end subroutine beltrami_row_correction_selftest

  !> \brief Standalone self-test of the independent GMRES(m) kernel and the block_lu preconditioner
  !> (the production pair) on a 2D 5-point Laplacian over a k x k grid (n = k*k). The
  !> preconditioner is the block-Jacobi restriction to same-grid-row couplings -- the analogue of
  !> the same-(m,n) restriction of the Beltrami operator -- so it is approximate and genuine GMRES
  !> iterations run. Returns the total inner iteration count, the true relative residual
  !> ||A x - b|| / ||b|| recomputed here, and the relative residual the solver itself reported
  !> (which must be that same true residual, not a Givens estimate of it) plus its status.
  !> Used by tests/wrapper/test_gmres_kernel.py.
  subroutine gmres_selftest(k, restart, maxiter, its, relres, relres_reported, ierr_gmres)
    use gmres_mod, only: gmres_solve, block_lu, block_lu_factor, block_lu_apply, block_lu_release
    integer, intent(in) :: k, restart, maxiter
    integer, intent(out) :: its, ierr_gmres
    real(rkind), intent(out) :: relres, relres_reported

    integer :: n, nnz, ix, iy, p, cnt, pcnt, ierr
    integer, allocatable :: ja(:), ia(:), pja(:), pia(:), grid_row(:)
    real(rkind), allocatable :: a(:), pa(:), b(:), xx(:), ax(:), preconditioner_work(:)
    real(rkind) :: bnorm
    type(block_lu) :: blu

    n = k * k
    nnz = 5 * n ! upper bound on the 5-point stencil nonzeros

    allocate(ja(nnz), ia(n + 1), pja(nnz), pia(n + 1), grid_row(n))
    allocate(a(nnz), pa(nnz), b(n), xx(n), ax(n), preconditioner_work(n))

    ! Operator CSR with columns in ascending order per row (p-k, p-1, p, p+1, p+k), and the
    ! preconditioner CSR keeping only the same-grid-row couplings (the tridiagonal blocks).
    cnt = 0
    pcnt = 0
    do iy = 1, k
      do ix = 1, k
        p = (iy - 1) * k + ix
        ia(p) = cnt + 1
        pia(p) = pcnt + 1
        grid_row(p) = iy
        if (iy > 1) then
          cnt = cnt + 1; ja(cnt) = p - k; a(cnt) = -1.0_rkind
        end if
        if (ix > 1) then
          cnt = cnt + 1; ja(cnt) = p - 1; a(cnt) = -1.0_rkind
          pcnt = pcnt + 1; pja(pcnt) = p - 1; pa(pcnt) = -1.0_rkind
        end if
        cnt = cnt + 1; ja(cnt) = p; a(cnt) = 4.0_rkind
        pcnt = pcnt + 1; pja(pcnt) = p; pa(pcnt) = 4.0_rkind
        if (ix < k) then
          cnt = cnt + 1; ja(cnt) = p + 1; a(cnt) = -1.0_rkind
          pcnt = pcnt + 1; pja(pcnt) = p + 1; pa(pcnt) = -1.0_rkind
        end if
        if (iy < k) then
          cnt = cnt + 1; ja(cnt) = p + k; a(cnt) = -1.0_rkind
        end if
      end do
    end do
    ia(n + 1) = cnt + 1
    pia(n + 1) = pcnt + 1

    call block_lu_factor(n, pa, pja, pia, grid_row, k, blu, ierr)

    b(1:n) = 1.0_rkind
    xx(1:n) = 0.0_rkind
    bnorm = norm2(b)

    call gmres_solve(n, b, xx, apply_a, apply_prec, restart, maxiter, 1.0e-12_rkind, 0.0_rkind, its, relres, ierr)

    relres_reported = relres / bnorm
    ierr_gmres = ierr

    ! true residual, recomputed independently of the solver
    call apply_a(xx, ax)
    relres = norm2(ax - b) / bnorm

    call block_lu_release(blu)
    deallocate(ja, ia, pja, pia, grid_row, a, pa, b, xx, ax, preconditioner_work)

  contains

    subroutine apply_a(v, av)
      real(rkind), intent(in) :: v(:)
      real(rkind), intent(out) :: av(:)
      integer :: ii, jj
      do ii = 1, n
        av(ii) = 0.0_rkind
        do jj = ia(ii), ia(ii + 1) - 1
          av(ii) = av(ii) + a(jj) * v(ja(jj))
        end do
      end do
    end subroutine apply_a

    subroutine apply_prec(v, z)
      real(rkind), intent(in) :: v(:)
      real(rkind), intent(out) :: z(:)
      call block_lu_apply(blu, v, z, preconditioner_work)
    end subroutine apply_prec

  end subroutine gmres_selftest

  !> Stress concurrent applications of one immutable block-LU factor with caller-private scratch.
  subroutine block_lu_concurrency_selftest(nrepeat, max_absdiff)
    use gmres_mod, only: block_lu, block_lu_apply, block_lu_factor, block_lu_release
    use omp_lib, only: omp_get_max_threads, omp_get_thread_num
    integer, intent(in) :: nrepeat
    real(rkind), intent(out) :: max_absdiff

    integer, parameter :: block_size = 4, nblocks = 3, nvec = 24
    integer, parameter :: n = block_size * nblocks, nnz = block_size * n
    integer :: col, group, i, ierr, ivec, k, repeat, row, thread_id
    integer :: ia(n + 1), ja(nnz), mode(n), nthreads
    real(rkind) :: aval(nnz), vectors(n, nvec), reference(n, nvec), results(n, nvec)
    real(rkind) :: serial_work(block_size)
    real(rkind), allocatable :: thread_work(:, :)
    type(block_lu) :: blu

    k = 0
    do row = 1, n
      ia(row) = k + 1
      group = (row - 1) / block_size + 1
      mode(row) = group
      do col = (group - 1) * block_size + 1, group * block_size
        k = k + 1
        ja(k) = col
        if (row == col) then
          aval(k) = 5.0_rkind + 0.1_rkind * group
        else
          aval(k) = 0.01_rkind * (row + col)
        end if
      end do
    end do
    ia(n + 1) = k + 1

    call block_lu_factor(n, aval, ja, ia, mode, nblocks, blu, ierr)
    if (ierr /= 0) then
      max_absdiff = huge(1.0_rkind)
      call block_lu_release(blu)
      return
    end if

    do ivec = 1, nvec
      do i = 1, n
        vectors(i, ivec) = sin(0.13_rkind * i * ivec) + 0.2_rkind * cos(0.07_rkind * i)
      end do
      call block_lu_apply(blu, vectors(:, ivec), reference(:, ivec), serial_work)
    end do

    nthreads = max(omp_get_max_threads(), 1)
    allocate(thread_work(block_size, 0:nthreads - 1))
    results = 0.0_rkind
    do repeat = 1, nrepeat
      !$omp parallel do default(none) schedule(static) &
      !$omp shared(blu, vectors, results, thread_work) private(ivec, thread_id)
      do ivec = 1, nvec
        thread_id = omp_get_thread_num()
        call block_lu_apply(blu, vectors(:, ivec), results(:, ivec), thread_work(:, thread_id))
      end do
      !$omp end parallel do
    end do

    max_absdiff = maxval(abs(results - reference))
    deallocate(thread_work)
    call block_lu_release(blu)
  end subroutine block_lu_concurrency_selftest

  !> \brief End-to-end self-test of the matrix-free Beltrami solve for volume lvol: build the
  !> dense operator (dMA - testmu*dMD), solve A x = b directly (DGESV) for a reference, then solve
  !> the same system with matrix-free GMRES (op_apply matvec) preconditioned by the exact block LU
  !> of the same-(m,n) block, and return the relative error ||x_gmres - x_dense|| / ||x_dense||
  !> and the GMRES iteration count. Validates the whole pipeline (matvec + preconditioner + Krylov)
  !> against a direct solve on a real Beltrami system. Used by tests/wrapper/test_beltrami_solve.py.
  subroutine beltrami_solve_selftest(lvol, testmu, restart, maxiter, gmres_tol, use_sparse, its, relerr)
    use beltrami_operator_mod, only: beltrami_operator, op_bind_iter, op_bind_iter_sparse, op_solve_iter, op_release, &
        op_prepare_metric
    integer, intent(in) :: lvol, restart, maxiter, use_sparse
    real(rkind), intent(in) :: testmu, gmres_tol
    integer, intent(out) :: its
    real(rkind), intent(out) :: relerr

    integer :: NN, ll, i, aslot, ierr, lquad
    real(rkind) :: resid
    real(rkind), allocatable :: Aop(:, :), b(:), xd(:), xg(:)
    integer, allocatable :: ipiv(:)
    type(beltrami_operator) :: op
    type(perturbed_rhs_workspace) :: pws
    type(matvec_multipliers) :: lm
    external :: dgesv

    call setup_dense_beltrami_reference(lvol, ll, NN, lquad)

    call setup_operator_workspace(ll, lquad, pws, lm)
    aslot = -1

    ! bind the iterative backend: use_sparse=1 builds the exact same-(m,n) block from the diagonal
    ! metric integrals (memory-optimal, no dense matrix), use_sparse=0 extracts the same block from
    ! the dense dMA/dMD. The two are equivalent by construction; this switch keeps both reachable so
    ! the dense extraction stays available as the independent reference. Bound before the
    ! virtual-casing subtraction below: production preconditioners are always uncorrected.
    if (use_sparse /= 0) then
      call op_bind_iter_sparse(op, lvol, NN, Lcoordinatesingularity, testmu)
    else
      call op_bind_iter(op, lvol, NN, dMA, dMD, testmu)
    end if

    ! direct free-boundary solve: the true vacuum operator is (dMA - dVC); correct the dense
    ! reference and the operator under test identically (the preconditioner stays uncorrected)
    if (Lfreebsolver == 1 .and. Lfreebound == 1 .and. lvol == Mvol) then
      call vc_subtract_dvc(dMA, NN)
      call vc_attach_operator_correction(op, NN)
    end if

    ! dense reference operator and direct solve
    allocate(Aop(NN, NN), b(NN), xd(NN), xg(NN), ipiv(NN))
    Aop(1:NN, 1:NN) = dMA(1:NN, 1:NN) - testmu * dMD(1:NN, 1:NN)
    do i = 1, NN
      b(i) = sin(0.11_rkind * i) + 0.4_rkind * cos(0.023_rkind * i * i)
    end do
    xd(1:NN) = b(1:NN)
    call dgesv(NN, 1, Aop, NN, ipiv, xd, NN, ierr)

    call op_prepare_metric(op, Lcoordinatesingularity, pws)
    xg(1:NN) = 0.0_rkind
    call op_solve_iter(op, testmu, Lcoordinatesingularity, aslot, b, xg, pws, lm, restart, maxiter, gmres_tol, its, resid, ierr)

    relerr = norm2(xg - xd) / max(norm2(xd), tiny(1.0_rkind))

    call op_release(op)
    deallocate(Aop, b, xd, xg, ipiv)
    call teardown_operator_workspace(pws, lm)
    call teardown_dense_beltrami_reference()
  end subroutine beltrami_solve_selftest

  !> Solve a genuinely row-corrected operator and compare GMRES with a corrected dense oracle.
  subroutine beltrami_row_correction_solve_selftest(lvol, testmu, restart, maxiter, gmres_tol, &
      its, relerr, ierr_gmres)
    use beltrami_operator_mod, only: beltrami_operator, op_bind_iter_sparse, op_prepare_metric, &
        op_release, op_set_row_correction, op_solve_iter
    integer, intent(in) :: lvol, restart, maxiter
    real(rkind), intent(in) :: testmu, gmres_tol
    integer, intent(out) :: its, ierr_gmres
    real(rkind), intent(out) :: relerr

    integer :: NN, ll, i, aslot, ierr, lquad
    integer :: rows(2)
    real(rkind) :: resid
    real(rkind), allocatable :: Aop(:, :), b(:), correction(:, :), xd(:), xg(:)
    integer, allocatable :: ipiv(:)
    type(beltrami_operator) :: op
    type(perturbed_rhs_workspace) :: pws
    type(matvec_multipliers) :: lm
    external :: dgesv

    call setup_dense_beltrami_reference(lvol, ll, NN, lquad)

    rows = [1, NN]
    allocate(Aop(NN, NN), b(NN), correction(2, NN), xd(NN), xg(NN), ipiv(NN))
    Aop = dMA(1:NN, 1:NN) - testmu * dMD(1:NN, 1:NN)
    correction(1, :) = 0.05_rkind * Aop(rows(1), :)
    correction(2, :) = -0.03_rkind * Aop(rows(2), :)
    Aop(rows(1), :) = Aop(rows(1), :) - correction(1, :)
    Aop(rows(2), :) = Aop(rows(2), :) - correction(2, :)
    do i = 1, NN
      b(i) = sin(0.11_rkind * i) + 0.4_rkind * cos(0.023_rkind * i * i)
    end do
    xd = b
    call dgesv(NN, 1, Aop, NN, ipiv, xd, NN, ierr)
    if (ierr /= 0) then
      its = 0
      ierr_gmres = ierr
      relerr = huge(1.0_rkind)
      deallocate(Aop, b, correction, xd, xg, ipiv)
      call teardown_dense_beltrami_reference()
      return
    end if

    call setup_operator_workspace(ll, lquad, pws, lm)
    call op_bind_iter_sparse(op, lvol, NN, Lcoordinatesingularity, testmu)
    call op_set_row_correction(op, 2, rows, correction)
    call op_prepare_metric(op, Lcoordinatesingularity, pws)
    aslot = -1
    xg = 0.0_rkind
    call op_solve_iter(op, testmu, Lcoordinatesingularity, aslot, b, xg, pws, lm, &
        restart, maxiter, gmres_tol, its, resid, ierr_gmres)

    relerr = norm2(xg - xd) / max(norm2(xd), tiny(1.0_rkind))

    call op_release(op)
    deallocate(Aop, b, correction, xd, xg, ipiv)
    call teardown_operator_workspace(pws, lm)
    call teardown_dense_beltrami_reference()
  end subroutine beltrami_row_correction_solve_selftest

  !> \brief Validate the production preconditioner builder (build_samemn_csr_diag, assembled from
  !> the diagonal metric integrals without a dense matrix) against the dense-extraction oracle
  !> (build_samemn_csr). Both must produce the same same-(m,n) block of (dMA - testmu*dMD); neither
  !> regularizes the diagonal, so every entry is compared. Returns the max entrywise difference
  !> relative to the largest entry. Used by tests/wrapper/test_beltrami_precond.py.
  subroutine samemn_precond_compare(lvol, testmu, maxreldiff)
    use beltrami_sparse_mod, only: build_dof_mode, samemn_csr_count, build_samemn_csr, build_samemn_csr_diag
    integer, intent(in) :: lvol
    real(rkind), intent(in) :: testmu
    real(rkind), intent(out) :: maxreldiff

    integer :: NN, ll, nnz, nnz2, i, k, lquad, ierr
    integer, allocatable :: dof_mode(:), ja(:), ia(:), ja2(:), ia2(:)
    real(rkind), allocatable :: aval(:), aval2(:), D1(:, :), D2(:, :)
    real(rkind) :: dmax, amax

    call setup_dense_beltrami_reference(lvol, ll, NN, lquad)

    ! dense-extraction reference
    allocate(dof_mode(NN))
    call build_dof_mode(lvol, NN, dof_mode)
    call samemn_csr_count(NN, dMA, dMD, testmu, dof_mode, nnz)
    allocate(aval(nnz), ja(nnz), ia(NN + 1))
    call build_samemn_csr(NN, dMA, dMD, testmu, dof_mode, aval, ja, ia)

    ! memory-optimal build from the diagonal metric integrals (no dense matrix). The full
    ! volume_integrate_chebyshev above already filled the tensors; the diagonal entries it reads are
    ! the same ones the dense assembler used, so this must agree with the extraction to round-off.
    allocate(aval2(nnz + 8 * NN), ja2(nnz + 8 * NN), ia2(NN + 1))
    call build_samemn_csr_diag(lvol, NN, Lcoordinatesingularity, testmu, aval2, ja2, ia2, nnz2, ierr)
    if (ierr /= 0) then
      maxreldiff = huge(1.0_rkind)
      call teardown_dense_beltrami_reference()
      return
    end if

    ! densify both and compare every entry (neither builder regularizes the diagonal)
    allocate(D1(NN, NN), D2(NN, NN))
    D1 = 0.0_rkind; D2 = 0.0_rkind
    do i = 1, NN
      do k = ia(i), ia(i + 1) - 1
        D1(i, ja(k)) = aval(k)
      end do
      do k = ia2(i), ia2(i + 1) - 1
        D2(i, ja2(k)) = aval2(k)
      end do
    end do

    dmax = maxval(abs(D1 - D2))
    amax = maxval(abs(D1))
    maxreldiff = dmax / max(amax, tiny(1.0_rkind))

    deallocate(dof_mode, aval, ja, ia, aval2, ja2, ia2, D1, D2)
    call teardown_dense_beltrami_reference()
  end subroutine samemn_precond_compare

end module matrix_free_test_support_mod
