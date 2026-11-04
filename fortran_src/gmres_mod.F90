! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> \brief Independent implementation of restarted, right-preconditioned GMRES(m) and an exact
!> block-diagonal LU preconditioner for block-diagonal-by-group CSR matrices. Pure linear algebra:
!> operator and preconditioner applications are passed as procedure arguments (abstract interface
!> gmres_op), so this module has no dependency on the Beltrami operator and can be unit-tested
!> standalone.
!>
!> The GMRES iteration follows Y. Saad and M. H. Schultz, "GMRES: A Generalized Minimal Residual
!> Algorithm for Solving Nonsymmetric Linear Systems", SIAM Journal on Scientific and Statistical
!> Computing 7(3), 856--869 (1986), https://doi.org/10.1137/0907058.
module gmres_mod

  use precisions_mod, only: rkind

  implicit none(type, external)

  private
  public :: gmres_op, gmres_solve
  public :: block_lu, block_lu_factor, block_lu_apply, block_lu_release

  external :: DGETRF, DGETRS, DLARTG, DTRSV

  !> \brief Exact block-diagonal LU preconditioner for a block-diagonal-by-group CSR matrix (each
  !> DOF belongs to one group; the matrix couples only DOFs in the same group). Each group's dense
  !> block is factored with partial pivoting (DGETRF), which — unlike an unpivoted ILU(0) — handles
  !> the saddle-point (zero Lagrange-row diagonals) exactly, giving a far stronger preconditioner.
  type block_lu
    integer :: n = 0, nb = 0
    integer, allocatable :: bsize(:) !< size of each block
    integer, allocatable :: doff(:) !< offset into dofs() for each block (0-based prefix)
    integer, allocatable :: dofs(:) !< concatenated global DOF indices, grouped by block
    integer, allocatable :: foff(:) !< offset into fac() for each block
    integer, allocatable :: poff(:) !< offset into piv() for each block
    real(rkind), allocatable :: fac(:) !< concatenated dense LU factors (bsize^2 per block)
    integer, allocatable :: piv(:) !< concatenated pivots (bsize per block)
  end type block_lu

  !> \brief Interface for a linear operator application av = Op . v (matvec or preconditioner).
  abstract interface
    subroutine gmres_op(v, av)
      import :: rkind
      real(rkind), intent(in) :: v(:)
      real(rkind), intent(out) :: av(:)
    end subroutine gmres_op
  end interface

contains

  !> \brief Build the exact block-diagonal LU preconditioner from a block-diagonal-by-group CSR
  !> matrix. dof_mode(id) is the group (1..nmode, or 0) of each DOF; the matrix must couple only
  !> DOFs of the same group. Each group's dense block is factored with DGETRF (partial pivoting),
  !> which handles the saddle-point exactly. A (near-)singular block is diagonally regularized so
  !> the factorization stays finite (only the preconditioner is affected, not the operator).
  subroutine block_lu_factor(n, aval, ja, ia, dof_mode, nmode, blu, ierr)
    integer, intent(in) :: n, ja(:), ia(:), dof_mode(:), nmode
    real(rkind), intent(in) :: aval(:)
    type(block_lu), intent(out) :: blu
    integer, intent(out) :: ierr

    integer :: cnt(0:nmode), bidx(0:nmode), pos(0:nmode), lidx(n)
    integer :: g, id, k, i, ib, bs, info, nb, bmax
    real(rkind) :: dscale
    real(rkind), allocatable :: ablock(:) !< untouched copy of the block being factored

    ierr = 0
    blu%n = n

    cnt(0:nmode) = 0
    do id = 1, n
      cnt(dof_mode(id)) = cnt(dof_mode(id)) + 1
    end do

    nb = 0
    do g = 0, nmode
      if (cnt(g) > 0) then
        nb = nb + 1
        bidx(g) = nb
      else
        bidx(g) = 0
      end if
    end do
    blu%nb = nb

    allocate(blu%bsize(nb), blu%doff(nb + 1), blu%foff(nb + 1), blu%poff(nb + 1))
    do g = 0, nmode
      if (bidx(g) > 0) blu%bsize(bidx(g)) = cnt(g)
    end do
    blu%doff(1) = 0; blu%foff(1) = 0; blu%poff(1) = 0
    do ib = 1, nb
      bs = blu%bsize(ib)
      blu%doff(ib + 1) = blu%doff(ib) + bs
      blu%foff(ib + 1) = blu%foff(ib) + bs * bs
      blu%poff(ib + 1) = blu%poff(ib) + bs
    end do
    allocate(blu%dofs(blu%doff(nb + 1)), blu%fac(blu%foff(nb + 1)), blu%piv(blu%poff(nb + 1)))

    ! group DOFs by block, recording each DOF's local index within its block
    pos(0:nmode) = 0
    do id = 1, n
      g = dof_mode(id)
      pos(g) = pos(g) + 1
      lidx(id) = pos(g)
      blu%dofs(blu%doff(bidx(g)) + pos(g)) = id
    end do

    ! scatter the CSR entries into the dense blocks (column-major)
    blu%fac(:) = 0.0_rkind
    do id = 1, n
      g = dof_mode(id)
      ib = bidx(g)
      bs = blu%bsize(ib)
      i = lidx(id)
      do k = ia(id), ia(id + 1) - 1
        blu%fac(blu%foff(ib) + (lidx(ja(k)) - 1) * bs + i) = aval(k)
      end do
    end do

    bmax = 0
    do ib = 1, nb
      bmax = max(bmax, blu%bsize(ib))
    end do
    allocate(ablock(max(bmax * bmax, 1)))

    ! factor each block with partial pivoting, regularizing the (near-)singular ones
    do ib = 1, nb
      bs = blu%bsize(ib)
      dscale = 0.0_rkind
      do i = 1, bs
        dscale = max(dscale, abs(blu%fac(blu%foff(ib) + (i - 1) * bs + i)))
      end do
      ! DGETRF overwrites the block with a partial LU when it fails, so keep the original entries:
      ! the regularized retry must shift the matrix, not the half-factored remains of it.
      ablock(1:bs * bs) = blu%fac(blu%foff(ib) + 1:blu%foff(ib) + bs * bs)
      call DGETRF(bs, bs, blu%fac(blu%foff(ib) + 1), bs, blu%piv(blu%poff(ib) + 1), info)
      if (info /= 0) then
        ! singular block: restore it, add a small shift and refactor so the preconditioner stays
        ! finite (only the preconditioner is affected, not the operator)
        blu%fac(blu%foff(ib) + 1:blu%foff(ib) + bs * bs) = ablock(1:bs * bs)
        do i = 1, bs
          blu%fac(blu%foff(ib) + (i - 1) * bs + i) = blu%fac(blu%foff(ib) + (i - 1) * bs + i) &
              + 1.0e-10_rkind * max(dscale, 1.0_rkind)
        end do
        call DGETRF(bs, bs, blu%fac(blu%foff(ib) + 1), bs, blu%piv(blu%poff(ib) + 1), info)
        if (info /= 0) ierr = ib
      end if
    end do

    deallocate(ablock)
  end subroutine block_lu_factor

  !> \brief Apply the block-diagonal LU preconditioner: z = M^-1 v, one dense triangular solve per
  !> block, gathering/scattering by the block's DOF list. ``work`` is caller-owned scratch of at
  !> least ``maxval(blu%bsize)`` entries, sized once per solve or active thread stripe. The LU
  !> factors are immutable, so concurrent applications may safely share ``blu`` when each caller
  !> supplies distinct scratch.
  subroutine block_lu_apply(blu, v, z, work)
    type(block_lu), intent(in) :: blu
    real(rkind), intent(in) :: v(:)
    real(rkind), intent(out) :: z(:)
    real(rkind), intent(inout) :: work(:)

    integer :: ib, bs, i, info

    do ib = 1, blu%nb
      bs = blu%bsize(ib)
      do i = 1, bs
        work(i) = v(blu%dofs(blu%doff(ib) + i))
      end do
      call DGETRS('N', bs, 1, blu%fac(blu%foff(ib) + 1), bs, blu%piv(blu%poff(ib) + 1), work, bs, info)
      do i = 1, bs
        z(blu%dofs(blu%doff(ib) + i)) = work(i)
      end do
    end do
  end subroutine block_lu_apply

  !> \brief Release the block-LU preconditioner storage.
  subroutine block_lu_release(blu)
    type(block_lu), intent(inout) :: blu
    if (allocated(blu%bsize)) deallocate(blu%bsize)
    if (allocated(blu%doff)) deallocate(blu%doff)
    if (allocated(blu%dofs)) deallocate(blu%dofs)
    if (allocated(blu%foff)) deallocate(blu%foff)
    if (allocated(blu%poff)) deallocate(blu%poff)
    if (allocated(blu%fac)) deallocate(blu%fac)
    if (allocated(blu%piv)) deallocate(blu%piv)
    blu%n = 0
    blu%nb = 0
  end subroutine block_lu_release

  !> \brief Restarted right-preconditioned GMRES(restart). Solves A x = b to relative tolerance
  !> rtol (or absolute atol) using at most maxiter total inner iterations. apply_A computes A.v,
  !> apply_prec computes M^-1.v. On entry x is the initial guess (warm start); on exit the
  !> solution. its = total inner iterations, resid = the final TRUE residual norm ||b - A x||,
  !> ierr = 0 converged, 1 = did not converge (iteration limit, or a stalled restart cycle).
  !>
  !> Convergence is declared on that true residual, never on the Givens estimate abs(g(j+1)): after
  !> a loss of Arnoldi orthogonality (ill-conditioned blocks, long restart cycles) the estimate can
  !> drift below the tolerance while ||b - A x|| is still far too large. The estimate is kept only
  !> as the cheap test for leaving the inner loop early; the residual that decides is recomputed
  !> from x once per restart cycle, which is the same single matvec the next cycle needed anyway.
  !>
  !> A requested tolerance can lie below the accuracy the residual can be evaluated with at all:
  !> b - A x is formed in floating point, so it carries an uncertainty of order
  !> eps (||A|| ||x|| + ||b||) — for these saddle-point systems that floor is orders of magnitude
  !> above eps ||b||. Reaching it IS convergence (the iterate is correct to working precision) and
  !> is reported as such; ||A|| is estimated from the matvecs the iteration performs anyway. A
  !> residual that stops improving above that floor is a genuine stall: the solve stops there with
  !> ierr = 1 instead of grinding out the remaining iterations.
  subroutine gmres_solve(n, b, x, apply_A, apply_prec, restart, maxiter, rtol, atol, its, resid, ierr)
    integer, intent(in) :: n, restart, maxiter
    real(rkind), intent(in) :: b(n), rtol, atol
    real(rkind), intent(inout) :: x(n)
    procedure(gmres_op) :: apply_A, apply_prec
    integer, intent(out) :: its, ierr
    real(rkind), intent(out) :: resid

    !> units of round-off allowed in the attainable-accuracy floor
    real(rkind), parameter :: kround = 8.0_rkind
    !> a cycle improving the residual by less than this factor has stalled
    real(rkind), parameter :: stall_factor = 0.999_rkind
    !> number of consecutive stalled cycles before giving up
    integer, parameter :: max_stall = 2

    real(rkind) :: V(n, restart + 1), Z(n, restart), H(restart + 1, restart)
    real(rkind) :: cs(restart), sn(restart), g(restart + 1), y(restart)
    real(rkind) :: w(n), r(n)
    real(rkind) :: bnorm, beta, tol, hij, temp, est
    real(rkind) :: anorm, znorm, rfloor, resid_prev
    integer :: i, j, jm, nstall

    its = 0
    resid = 0.0_rkind
    ierr = 0

    bnorm = norm2(b)
    if (bnorm == 0.0_rkind) then
      x(1:n) = 0.0_rkind
      return
    end if
    tol = max(rtol * bnorm, atol)

    anorm = 0.0_rkind
    rfloor = 0.0_rkind
    resid_prev = huge(1.0_rkind)
    nstall = 0

    call apply_A(x, w)
    r(1:n) = b(1:n) - w(1:n)
    beta = norm2(r)
    resid = beta

    outer: do
      if (resid <= max(tol, rfloor) .or. its >= maxiter) exit outer

      V(1:n, 1) = r(1:n) / beta
      g(1) = beta
      g(2:restart + 1) = 0.0_rkind
      jm = 0

      inner: do j = 1, restart
        its = its + 1
        jm = j

        call apply_prec(V(1:n, j), Z(1:n, j))
        call apply_A(Z(1:n, j), w)

        ! running lower bound on ||A||, for the attainable-accuracy floor below
        znorm = norm2(Z(1:n, j))
        if (znorm > 0.0_rkind) anorm = max(anorm, norm2(w) / znorm)

        ! modified Gram-Schmidt orthogonalization -- deliberately hand-rolled with intrinsics: a
        ! DGEMV pair would be classical GS, whose weaker orthogonality this module's convergence
        ! policy depends on avoiding, and a 1:1 DDOT/DAXPY swap changes nothing.
        do i = 1, j
          H(i, j) = dot_product(w(1:n), V(1:n, i))
          w(1:n) = w(1:n) - H(i, j) * V(1:n, i)
        end do
        H(j + 1, j) = norm2(w)
        if (H(j + 1, j) > 0.0_rkind) V(1:n, j + 1) = w(1:n) / H(j + 1, j)

        ! apply the previous Givens rotations to the new Hessenberg column
        do i = 1, j - 1
          temp = cs(i) * H(i, j) + sn(i) * H(i + 1, j)
          H(i + 1, j) = -sn(i) * H(i, j) + cs(i) * H(i + 1, j)
          H(i, j) = temp
        end do

        ! new Givens rotation to zero H(j+1,j)
        call DLARTG(H(j, j), H(j + 1, j), cs(j), sn(j), hij)
        H(j, j) = hij
        H(j + 1, j) = 0.0_rkind
        g(j + 1) = -sn(j) * g(j)
        g(j) = cs(j) * g(j)
        est = abs(g(j + 1)) ! Givens estimate: cheap, but only a lower bound once orthogonality is lost

        if (est <= tol .or. its >= maxiter) exit inner
      end do inner

      ! Solve the triangular least-squares system for y.
      y(1:jm) = g(1:jm)
      call DTRSV('U', 'N', 'N', jm, H, restart + 1, y, 1)

      ! update the solution: x <- x + Z(:,1:jm) y   (Z already holds M^-1 V). Hand-rolled on
      ! purpose: this runs inside the Jacobian's striped OMP region, where a DGEMV would hand
      ! an n x jm panel to a pthread-built BLAS pool and oversubscribe the team.
      do i = 1, jm
        x(1:n) = x(1:n) + y(i) * Z(1:n, i)
      end do

      ! true residual of the updated x; also the starting residual of the next restart cycle
      call apply_A(x, w)
      r(1:n) = b(1:n) - w(1:n)
      beta = norm2(r)
      resid = beta

      ! accuracy the residual can be evaluated with, and the stall test against it
      rfloor = kround * epsilon(1.0_rkind) * (anorm * norm2(x) + bnorm)
      if (resid > max(tol, rfloor)) then
        if (resid > stall_factor * resid_prev) then
          nstall = nstall + 1
          if (nstall >= max_stall) exit outer
        else
          nstall = 0
        end if
      end if
      resid_prev = resid
    end do outer

    if (resid > max(tol, rfloor)) ierr = 1
  end subroutine gmres_solve

end module gmres_mod
