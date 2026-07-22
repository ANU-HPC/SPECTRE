! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> \brief Clean-room restarted, right-preconditioned GMRES(m) with an ILU(0) preconditioner on a
!> CSR matrix. Pure linear algebra: the operator and preconditioner applications are passed as
!> procedure arguments (abstract interface matvec_op), so this module has no dependency on the
!> Beltrami operator and can be unit-tested standalone. The textbook algorithms (Arnoldi with
!> modified Gram-Schmidt, Givens QR of the Hessenberg, ILU(0), CSR triangular solves) are not
!> ported from any GPL/LGPL source.
module gmres_mod

  use precisions_mod, only: rkind

  implicit none(type, external)

  private
  public :: gmres_op, gmres_solve, ilu0_factor, ilu0_solve, csr_diag_ptr
  public :: block_lu, block_lu_factor, block_lu_apply, block_lu_release

  external :: DGETRF, DGETRS

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

  !> \brief Locate the diagonal position of each row of a CSR matrix. uptr(i) is the index into
  !> ja/values of the (i,i) entry. Sets ierr /= 0 if any diagonal is missing from the pattern.
  subroutine csr_diag_ptr(n, ja, ia, uptr, ierr)
    integer, intent(in) :: n, ja(:), ia(:)
    integer, intent(out) :: uptr(:), ierr

    integer :: i, k

    ierr = 0
    do i = 1, n
      uptr(i) = 0
      do k = ia(i), ia(i + 1) - 1
        if (ja(k) == i) then
          uptr(i) = k
          exit
        end if
      end do
      if (uptr(i) == 0) then
        ierr = i ! missing diagonal in row i
        return
      end if
    end do
  end subroutine csr_diag_ptr

  !> \brief ILU(0) factorization of a CSR matrix in place: luval starts as a copy of the matrix
  !> values and is overwritten with the combined L\\U factors (unit lower L, upper U) on the same
  !> sparsity pattern. uptr must hold the diagonal positions (csr_diag_ptr). iw(1:n) is integer
  !> scratch. ierr = 0 on success, > 0 = row index of a zero pivot.
  subroutine ilu0_factor(n, luval, ja, ia, uptr, iw, ierr)
    integer, intent(in) :: n, ja(:), ia(:), uptr(:)
    real(rkind), intent(inout) :: luval(:)
    integer, intent(inout) :: iw(:)
    integer, intent(out) :: ierr

    integer :: i, j, jj, jrow, jw
    real(rkind) :: tl, dscale, pfloor

    ierr = 0
    iw(1:n) = 0

    ! diagonal scale, for a relative floor on pivots (this is an indefinite saddle-point system,
    ! so unpivoted ILU(0) can produce vanishing pivots mid-factorization; floor them so the
    ! preconditioner stays finite — this only affects its quality, not the operator GMRES solves)
    dscale = 0.0_rkind
    do i = 1, n
      dscale = max(dscale, abs(luval(uptr(i))))
    end do
    pfloor = 1.0e-12_rkind * max(dscale, 1.0_rkind)

    do i = 1, n
      ! map each column present in row i to its storage position
      do j = ia(i), ia(i + 1) - 1
        iw(ja(j)) = j
      end do

      ! eliminate the strictly-lower entries of row i
      do j = ia(i), uptr(i) - 1
        jrow = ja(j)
        tl = luval(j) / luval(uptr(jrow))
        luval(j) = tl ! L(i, jrow)
        ! row_i <- row_i - tl * U(jrow, :)  over the shared sparsity pattern
        do jj = uptr(jrow) + 1, ia(jrow + 1) - 1
          jw = iw(ja(jj))
          if (jw /= 0) luval(jw) = luval(jw) - tl * luval(jj)
        end do
      end do

      ! floor a vanishing pivot to keep the factorization finite
      if (abs(luval(uptr(i))) < pfloor) then
        luval(uptr(i)) = sign(pfloor, luval(uptr(i)) + pfloor)
      end if

      ! reset the column map for the next row
      do j = ia(i), ia(i + 1) - 1
        iw(ja(j)) = 0
      end do
    end do
  end subroutine ilu0_factor

  !> \brief Apply the ILU(0) preconditioner: solve (L U) x = b by forward then back substitution,
  !> where luval/ja/ia/uptr are the factors from ilu0_factor (L has unit diagonal).
  subroutine ilu0_solve(n, luval, ja, ia, uptr, b, x)
    integer, intent(in) :: n, ja(:), ia(:), uptr(:)
    real(rkind), intent(in) :: luval(:), b(:)
    real(rkind), intent(out) :: x(:)

    integer :: i, j
    real(rkind) :: t

    ! forward solve L y = b  (unit lower); reuse x to hold y
    do i = 1, n
      t = b(i)
      do j = ia(i), uptr(i) - 1
        t = t - luval(j) * x(ja(j))
      end do
      x(i) = t
    end do

    ! back solve U x = y
    do i = n, 1, -1
      t = x(i)
      do j = uptr(i) + 1, ia(i + 1) - 1
        t = t - luval(j) * x(ja(j))
      end do
      x(i) = t / luval(uptr(i))
    end do
  end subroutine ilu0_solve

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
    integer :: g, id, k, jd, i, ib, bs, info, nb
    real(rkind) :: dscale

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

    ! regularize (near-)singular blocks, then factor each with partial pivoting
    do ib = 1, nb
      bs = blu%bsize(ib)
      dscale = 0.0_rkind
      do i = 1, bs
        dscale = max(dscale, abs(blu%fac(blu%foff(ib) + (i - 1) * bs + i)))
      end do
      call DGETRF(bs, bs, blu%fac(blu%foff(ib) + 1), bs, blu%piv(blu%poff(ib) + 1), info)
      if (info /= 0) then
        ! singular block: add a small shift and refactor so the preconditioner stays finite
        do i = 1, bs
          blu%fac(blu%foff(ib) + (i - 1) * bs + i) = blu%fac(blu%foff(ib) + (i - 1) * bs + i) &
              + 1.0e-10_rkind * max(dscale, 1.0_rkind)
        end do
        call DGETRF(bs, bs, blu%fac(blu%foff(ib) + 1), bs, blu%piv(blu%poff(ib) + 1), info)
        if (info /= 0) ierr = ib
      end if
    end do
  end subroutine block_lu_factor

  !> \brief Apply the block-diagonal LU preconditioner: z = M^-1 v, one dense triangular solve per
  !> block, gathering/scattering by the block's DOF list.
  subroutine block_lu_apply(blu, v, z)
    type(block_lu), intent(in) :: blu
    real(rkind), intent(in) :: v(:)
    real(rkind), intent(out) :: z(:)

    integer :: ib, bs, i, info, bmax
    real(rkind), allocatable :: local(:)

    bmax = 0
    do ib = 1, blu%nb
      bmax = max(bmax, blu%bsize(ib))
    end do
    allocate(local(bmax))

    do ib = 1, blu%nb
      bs = blu%bsize(ib)
      do i = 1, bs
        local(i) = v(blu%dofs(blu%doff(ib) + i))
      end do
      call DGETRS('N', bs, 1, blu%fac(blu%foff(ib) + 1), bs, blu%piv(blu%poff(ib) + 1), local, bs, info)
      do i = 1, bs
        z(blu%dofs(blu%doff(ib) + i)) = local(i)
      end do
    end do

    deallocate(local)
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
  !> solution. its = total inner iterations, resid = final preconditioner-space residual norm,
  !> ierr = 0 converged, 1 = maxiter reached without converging.
  subroutine gmres_solve(n, b, x, apply_A, apply_prec, restart, maxiter, rtol, atol, its, resid, ierr)
    integer, intent(in) :: n, restart, maxiter
    real(rkind), intent(in) :: b(n), rtol, atol
    real(rkind), intent(inout) :: x(n)
    procedure(gmres_op) :: apply_A, apply_prec
    integer, intent(out) :: its, ierr
    real(rkind), intent(out) :: resid

    real(rkind) :: V(n, restart + 1), Z(n, restart), H(restart + 1, restart)
    real(rkind) :: cs(restart), sn(restart), g(restart + 1), y(restart)
    real(rkind) :: w(n), r(n)
    real(rkind) :: bnorm, beta, tol, hij, temp
    integer :: i, j, k, jm

    its = 0
    resid = 0.0_rkind
    ierr = 0

    bnorm = norm2(b)
    if (bnorm == 0.0_rkind) then
      x(1:n) = 0.0_rkind
      return
    end if
    tol = max(rtol * bnorm, atol)

    outer: do
      call apply_A(x, w)
      r(1:n) = b(1:n) - w(1:n)
      beta = norm2(r)
      resid = beta
      if (beta <= tol .or. its >= maxiter) exit outer

      V(1:n, 1) = r(1:n) / beta
      g(1) = beta
      g(2:restart + 1) = 0.0_rkind
      jm = 0

      inner: do j = 1, restart
        its = its + 1
        jm = j

        call apply_prec(V(1:n, j), Z(1:n, j))
        call apply_A(Z(1:n, j), w)

        ! modified Gram-Schmidt orthogonalization
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
        call sym_givens(H(j, j), H(j + 1, j), cs(j), sn(j))
        hij = cs(j) * H(j, j) + sn(j) * H(j + 1, j)
        H(j, j) = hij
        H(j + 1, j) = 0.0_rkind
        g(j + 1) = -sn(j) * g(j)
        g(j) = cs(j) * g(j)
        resid = abs(g(j + 1))

        if (resid <= tol .or. its >= maxiter) exit inner
      end do inner

      ! back-substitute the triangular least-squares system for y
      do i = jm, 1, -1
        y(i) = g(i)
        do k = i + 1, jm
          y(i) = y(i) - H(i, k) * y(k)
        end do
        y(i) = y(i) / H(i, i)
      end do

      ! update the solution: x <- x + Z(:,1:jm) y   (Z already holds M^-1 V)
      do i = 1, jm
        x(1:n) = x(1:n) + y(i) * Z(1:n, i)
      end do

      if (resid <= tol .or. its >= maxiter) exit outer
    end do outer

    if (resid > tol) ierr = 1
  end subroutine gmres_solve

  !> \brief Stable Givens rotation zeroing b: returns c, s with [c s; -s c][a;b] = [r;0].
  pure subroutine sym_givens(a, b, c, s)
    real(rkind), intent(in) :: a, b
    real(rkind), intent(out) :: c, s

    real(rkind) :: t

    if (b == 0.0_rkind) then
      c = 1.0_rkind
      s = 0.0_rkind
    else if (abs(b) > abs(a)) then
      t = a / b
      s = 1.0_rkind / sqrt(1.0_rkind + t * t)
      c = t * s
    else
      t = b / a
      c = 1.0_rkind / sqrt(1.0_rkind + t * t)
      s = t * c
    end if
  end subroutine sym_givens

end module gmres_mod
