! SPDX-FileCopyrightText: 2026 Proxima Fusion
!
! SPDX-License-Identifier: MIT

!> \brief Same-(m,n) block of the Beltrami operator as a CSR matrix, used as the GMRES
!> preconditioner base. Because every Fourier mode (im(ii), in(ii)) is unique, keeping only the
!> couplings with m_i = m_j and n_i = n_j makes the block-diagonal-by-mode restriction of the
!> operator, whose per-mode pivoted LU (gmres_mod block_lu) is an exact factorization. The
!> production builder is build_samemn_csr_diag, which assembles the block from the diagonal
!> metric-integral tensors without ever forming the dense operator; build_samemn_csr extracts
!> the same block from a dense dMA/dMD and is retained as the independent test oracle.
module beltrami_sparse_mod

  use precisions_mod, only: rkind
  use allglobal_mod, only: im, in, mn, Ate, Aze, Ato, Azo, &
      Lma, Lmb, Lmc, Lmd, Lme, Lmf, Lmg, Lmh, &
      RTT, RTM, TT, &
      DToocc, TTssss, TDstsc, TDszsc, DDttcc, DDtzcc, DDzzcc
  use input_list_mod, only: Lrad, mpol, enforce_stell_sym

  implicit none(type, external)

  private
  public :: build_dof_mode, samemn_csr_count, build_samemn_csr
  public :: build_samemn_csr_diag

contains

  !> \brief Map each augmented DOF to its Fourier-mode index. dof_mode(id) = ii for every interior
  !> vector-potential DOF and every Lagrange-multiplier DOF belonging to mode ii; 0 if unused.
  subroutine build_dof_mode(lvol, NN, dof_mode)
    integer, intent(in) :: lvol, NN
    integer, intent(out) :: dof_mode(1:NN)

    integer :: ii, ll, id

    dof_mode(1:NN) = 0

    do ii = 1, mn
      do ll = 0, Lrad(lvol)
        id = Ate(lvol, 0, ii)%i(ll); if (id /= 0) dof_mode(id) = ii
        id = Aze(lvol, 0, ii)%i(ll); if (id /= 0) dof_mode(id) = ii
        if (.not.enforce_stell_sym) then
          id = Ato(lvol, 0, ii)%i(ll); if (id /= 0) dof_mode(id) = ii
          id = Azo(lvol, 0, ii)%i(ll); if (id /= 0) dof_mode(id) = ii
        end if
      end do
      if (Lma(lvol, ii) /= 0) dof_mode(Lma(lvol, ii)) = ii
      if (Lmb(lvol, ii) /= 0) dof_mode(Lmb(lvol, ii)) = ii
      if (Lme(lvol, ii) /= 0) dof_mode(Lme(lvol, ii)) = ii
      if (Lmg(lvol, ii) /= 0) dof_mode(Lmg(lvol, ii)) = ii
      if (Lmh(lvol, ii) /= 0) dof_mode(Lmh(lvol, ii)) = ii
      if (.not.enforce_stell_sym) then
        if (Lmc(lvol, ii) /= 0) dof_mode(Lmc(lvol, ii)) = ii
        if (Lmd(lvol, ii) /= 0) dof_mode(Lmd(lvol, ii)) = ii
        if (Lmf(lvol, ii) /= 0) dof_mode(Lmf(lvol, ii)) = ii
      end if
    end do
  end subroutine build_dof_mode

  !> \brief Count the nonzeros of the same-(m,n) block of (dMA - mu*dMD), so the caller can size
  !> the CSR arrays. The diagonal is always counted, even when structurally zero, so every block
  !> has its pivot slot.
  subroutine samemn_csr_count(NN, dMA, dMD, mu, dof_mode, nnz)
    integer, intent(in) :: NN
    real(rkind), intent(in) :: dMA(0:NN, 0:NN), dMD(0:NN, 0:NN), mu
    integer, intent(in) :: dof_mode(1:NN)
    integer, intent(out) :: nnz

    integer :: i, j
    real(rkind) :: v

    nnz = 0
    do i = 1, NN
      do j = 1, NN
        if (dof_mode(i) /= dof_mode(j)) cycle
        v = dMA(i, j) - mu * dMD(i, j)
        if (j == i .or. v /= 0.0_rkind) nnz = nnz + 1
      end do
    end do
  end subroutine samemn_csr_count

  !> \brief Assemble the same-(m,n) block of (dMA - mu*dMD) as a CSR matrix (aval, ja, ia) with
  !> columns in ascending order per row (the inner loop over j is naturally ordered). The diagonal
  !> is always emitted, including the genuine zeros of the saddle-point Lagrange rows: block_lu
  !> factors each mode block with partial pivoting, so no diagonal regularization is applied (or
  !> needed). ia has length NN+1. This is the dense-extraction test oracle for
  !> build_samemn_csr_diag; production never forms the dense dMA/dMD it reads.
  subroutine build_samemn_csr(NN, dMA, dMD, mu, dof_mode, aval, ja, ia)
    integer, intent(in) :: NN
    real(rkind), intent(in) :: dMA(0:NN, 0:NN), dMD(0:NN, 0:NN), mu
    integer, intent(in) :: dof_mode(1:NN)
    real(rkind), intent(out) :: aval(:)
    integer, intent(out) :: ja(:), ia(:)

    integer :: i, j, k
    real(rkind) :: v

    k = 0
    do i = 1, NN
      ia(i) = k + 1
      do j = 1, NN
        if (dof_mode(i) /= dof_mode(j)) cycle
        v = dMA(i, j) - mu * dMD(i, j)
        if (j == i .or. v /= 0.0_rkind) then
          k = k + 1
          aval(k) = v
          ja(k) = j
        end if
      end do
    end do
    ia(NN + 1) = k + 1
  end subroutine build_samemn_csr

  !> \brief Build the EXACT same-(m,n) preconditioner CSR from the diagonal metric-integral tensors.
  !>
  !> Reads the tensors produced by volume_integrate_chebyshev(..., diag_only=.true.) and evaluates
  !> the same expressions the dense assembler uses (matrices_mod.F90 Wtete/Wzete/Wteze/Wzeze,
  !> Hzete/Hteze) specialised to jj == ii. The result is therefore the same-(m,n) block of
  !> (dMA - mu*dMD) to round-off, without the dense operator ever being formed. The caller must
  !> have filled the tensors for this volume first.
  !>
  !> No ii == 1 special case is needed: the kijs/kija divisors inside volume_integrate_chebyshev
  !> already collapse to the degenerate (2m,2n) == (0,0) behaviour there. Likewise no
  !> pi2pi2nfphalf scaling -- it is applied inside the integration, and re-applying it here would
  !> silently mis-scale the preconditioner.
  subroutine build_samemn_csr_diag(lvol, NN, Lcoord, mu, aval, ja, ia, nnz_out)
    use chebyshev_mod, only: tensor_jslot
    integer, intent(in) :: lvol, NN
    logical, intent(in) :: Lcoord
    real(rkind), intent(in) :: mu
    real(rkind), intent(out) :: aval(:)
    integer, intent(out) :: ja(:), ia(:)
    integer, intent(out) :: nnz_out

    integer :: lr, mi, ni, ii, ll, pp, ll1, pp1, js, kk, nb, k, kdiag, idx
    real(rkind) :: wA, wD
    real(rkind), allocatable :: TTdata(:, :, :), TTMdata(:, :)
    integer, allocatable :: rmode(:), rll(:), rpot(:), col(:), ord(:)
    real(rkind), allocatable :: val(:)

    lr = Lrad(lvol)

    ! ---- boundary/Lagrange basis values (as in matrix/mtrxhs) ----
    allocate(TTdata(0:lr, 0:mpol, 0:1), TTMdata(0:lr, 0:mpol))
    if (Lcoord) then
      TTdata = RTT(0:lr, 0:mpol, 0:1, 0); TTMdata = RTM(0:lr, 0:mpol)
    else
      do ii = 0, mpol
        TTdata(0:lr, ii, 0:1) = TT(0:lr, 0:1, 0); TTMdata(0:lr, ii) = TT(0:lr, 0, 0)
      end do
    end if

    ! ---- reverse maps: for each DOF, which (mode, ll, potential/Lagrange kind) it is ----
    allocate(rmode(NN), rll(NN), rpot(NN))
    rmode = 0; rll = 0; rpot = 0
    do ii = 1, mn
      do ll = 0, lr
        call setrow(Ate(lvol, 0, ii)%i(ll), ii, ll, 1)
        call setrow(Aze(lvol, 0, ii)%i(ll), ii, ll, 2)
      end do
      call setrow(Lma(lvol, ii), ii, 0, 3)
      call setrow(Lmb(lvol, ii), ii, 0, 4)
      call setrow(Lme(lvol, ii), ii, 0, 5)
      call setrow(Lmg(lvol, ii), ii, 0, 6)
      call setrow(Lmh(lvol, ii), ii, 0, 7)
    end do

    ! ---- assemble CSR row by row, columns ascending per row ----
    allocate(col(4 * lr + 16), val(4 * lr + 16), ord(4 * lr + 16))
    k = 0
    do idx = 1, NN
      ia(idx) = k + 1
      ii = rmode(idx)
      if (ii == 0) then
        k = k + 1; aval(k) = 1.0_rkind; ja(k) = idx ! unmapped DOF: unit diagonal
        cycle
      end if
      mi = im(ii); ni = in(ii); ll = rll(idx)
      js = tensor_jslot(ii) ! shared writer/reader convention for the tensors' mode subscript
      if (Lcoord .and. ii == 1) then; kk = 1; else; kk = 0; end if

      if (Lcoord) then; ll1 = (ll - mod(ll, 2)) / 2; else; ll1 = ll; end if

      nb = 0
      select case (rpot(idx))
      case (1) ! Ate(ii,ll) row: interior couplings + Lagrange columns
        do pp = 0, lr
          if (Lcoord) then
            if (pp < mi .or. mod(pp + mi, 2) /= 0) cycle
            pp1 = (pp - mod(pp, 2)) / 2
          else
            pp1 = pp
          end if
          ! Wtete (Htete = 0)
          wA = +2.0_rkind * ni * ni * TTssss(ll1, pp1, ii, js) - 2.0_rkind * ni * TDszsc(ll1, pp1, ii, js) &
              - 2.0_rkind * ni * TDszsc(pp1, ll1, ii, js) + 2.0_rkind * DDzzcc(ll1, pp1, ii, js)
          call push(Ate(lvol, 0, ii)%i(pp), wA)
          ! Wzete / Hzete
          wA = +2.0_rkind * ni * mi * TTssss(ll1, pp1, ii, js) + 2.0_rkind * ni * TDstsc(ll1, pp1, ii, js) &
              - 2.0_rkind * mi * TDszsc(pp1, ll1, ii, js) - 2.0_rkind * DDtzcc(pp1, ll1, ii, js)
          wD = -DToocc(pp1, ll1, ii, js) + DToocc(ll1, pp1, ii, js)
          call push(Aze(lvol, 0, ii)%i(pp), wA - mu * wD)
        end do
        call push(Lma(lvol, ii), TTMdata(ll, mi))
        if (ii > 1) then
          call push(Lme(lvol, ii), -ni * TTdata(ll, mi, 1))
        else
          call push(Lmg(lvol, ii), TTdata(ll, mi, 1))
        end if
      case (2) ! Aze(ii,ll) row
        do pp = 0, lr
          if (Lcoord) then
            if (pp < mi .or. mod(pp + mi, 2) /= 0) cycle
            pp1 = (pp - mod(pp, 2)) / 2
          else
            pp1 = pp
          end if
          ! Wteze / Hteze
          wA = +2.0_rkind * mi * ni * TTssss(ll1, pp1, ii, js) + 2.0_rkind * ni * TDstsc(pp1, ll1, ii, js) &
              - 2.0_rkind * mi * TDszsc(ll1, pp1, ii, js) - 2.0_rkind * DDtzcc(ll1, pp1, ii, js)
          wD = +DToocc(pp1, ll1, ii, js) - DToocc(ll1, pp1, ii, js)
          call push(Ate(lvol, 0, ii)%i(pp), wA - mu * wD)
          ! Wzeze (Hzeze = 0)
          wA = +2.0_rkind * mi * mi * TTssss(ll1, pp1, ii, js) + 2.0_rkind * mi * TDstsc(ll1, pp1, ii, js) &
              + 2.0_rkind * mi * TDstsc(pp1, ll1, ii, js) + 2.0_rkind * DDttcc(ll1, pp1, ii, js)
          call push(Aze(lvol, 0, ii)%i(pp), wA)
        end do
        call push(Lmb(lvol, ii), TTdata(ll, mi, kk))
        if (ii > 1) then
          call push(Lme(lvol, ii), -mi * TTdata(ll, mi, 1))
        else
          call push(Lmh(lvol, ii), TTdata(ll, mi, 1))
        end if
      case (3) ! Lma row: couples to Ate(ii,pp)
        do pp = 0, lr
          call push(Ate(lvol, 0, ii)%i(pp), TTMdata(pp, mi))
        end do
      case (4) ! Lmb row: couples to Aze(ii,pp)
        do pp = 0, lr
          call push(Aze(lvol, 0, ii)%i(pp), TTdata(pp, mi, kk))
        end do
      case (5) ! Lme row: couples to Ate and Aze
        do pp = 0, lr
          call push(Ate(lvol, 0, ii)%i(pp), -ni * TTdata(pp, mi, 1))
          call push(Aze(lvol, 0, ii)%i(pp), -mi * TTdata(pp, mi, 1))
        end do
      case (6) ! Lmg row (ii==1): couples to Ate
        do pp = 0, lr
          call push(Ate(lvol, 0, ii)%i(pp), TTdata(pp, mi, 1))
        end do
      case (7) ! Lmh row (ii==1): couples to Aze
        do pp = 0, lr
          call push(Aze(lvol, 0, ii)%i(pp), TTdata(pp, mi, 1))
        end do
      end select

      call isort(col, nb, ord)
      kdiag = 0
      do pp = 1, nb
        k = k + 1
        aval(k) = val(ord(pp)); ja(k) = col(ord(pp))
        if (ja(k) == idx) kdiag = k
      end do
      ! NO diagonal regularization. The Lagrange rows of this saddle-point operator genuinely have
      ! zero diagonals, and block_lu factors each mode block with partial pivoting (DGETRF), which
      ! handles that exactly -- and shifts a block itself if it really is singular. Boosting the
      ! diagonal perturbs the block away from the true operator and measurably costs iterations
      ! (m13n11 vacuum: k=953 boosted vs ~36 not). Emit a structural zero if the row had no
      ! diagonal at all, so every block has its pivot slot.
      if (kdiag == 0) then
        k = k + 1; aval(k) = 0.0_rkind; ja(k) = idx
        call fix_tail_order(aval, ja, ia(idx), k)
      end if
    end do
    ia(NN + 1) = k + 1
    nnz_out = k

    deallocate(TTdata, TTMdata, rmode, rll, rpot, col, val, ord)

  contains

    subroutine setrow(id, iim, illv, ipot)
      integer, intent(in) :: id, iim, illv, ipot
      if (id /= 0) then
        rmode(id) = iim; rll(id) = illv; rpot(id) = ipot
      end if
    end subroutine setrow

    ! append one (column, value) to the current row's buffer; skip absent DOFs and structural zeros
    ! (but always keep the diagonal so the factorization has a pivot slot)
    subroutine push(jd, v)
      integer, intent(in) :: jd
      real(rkind), intent(in) :: v
      if (jd == 0) return
      if (v == 0.0_rkind .and. jd /= idx) return
      nb = nb + 1
      col(nb) = jd; val(nb) = v
    end subroutine push

  end subroutine build_samemn_csr_diag

  !> \brief Integer sort (insertion) returning the ascending order permutation ord(1:n) of a(1:n).
  pure subroutine isort(a, n, ord)
    integer, intent(in) :: a(:), n
    integer, intent(out) :: ord(:)
    integer :: i, j, key
    do i = 1, n
      ord(i) = i
    end do
    do i = 2, n
      key = ord(i)
      j = i - 1
      do while (j >= 1)
        if (a(ord(j)) <= a(key)) exit
        ord(j + 1) = ord(j)
        j = j - 1
      end do
      ord(j + 1) = key
    end do
  end subroutine isort

  !> \brief If a just-appended diagonal is out of column order in the current row [rowstart..k],
  !> bubble it left into place so the row stays ascending.
  pure subroutine fix_tail_order(aval, ja, rowstart, k)
    real(rkind), intent(inout) :: aval(:)
    integer, intent(inout) :: ja(:)
    integer, intent(in) :: rowstart, k
    integer :: p, jt
    real(rkind) :: vt
    p = k
    do while (p > rowstart)
      if (ja(p - 1) <= ja(p)) exit
      jt = ja(p); ja(p) = ja(p - 1); ja(p - 1) = jt
      vt = aval(p); aval(p) = aval(p - 1); aval(p - 1) = vt
      p = p - 1
    end do
  end subroutine fix_tail_order

end module beltrami_sparse_mod
