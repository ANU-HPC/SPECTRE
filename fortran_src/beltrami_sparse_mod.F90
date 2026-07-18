! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

!> \brief Sparse same-(m,n) approximation of the Beltrami operator, used as the GMRES
!> preconditioner base. Because every Fourier mode (im(ii), in(ii)) is unique, keeping only the
!> couplings with m_i = m_j and n_i = n_j makes the approximation block-diagonal in the mode
!> index; its ILU(0) is then an exact block factorization. This is the same idea as SPEC's
!> spsint/spsmat (the "tokamak with the same major/minor radius" preconditioner). This module
!> currently builds the CSR by extracting the same-(m,n) block from the dense dMA/dMD; a later
!> step replaces that with the reduced flux-averaged-metric integrals so the dense matrix need
!> never be formed for the large vacuum volume.
module beltrami_sparse_mod

  use precisions_mod, only: rkind
  use allglobal_mod, only: im, in, mn, NAdof, Ate, Aze, Ato, Azo, &
      Lma, Lmb, Lmc, Lmd, Lme, Lmf, Lmg, Lmh, &
      gaussianabscissae, gaussianweight, RTT, RTM, TT, pi2pi2nfphalf, Iquad
  use input_list_mod, only: Lrad, mpol, enforce_stell_sym
  use base_functions_mod, only: get_cheby, get_zernike

  implicit none(type, external)

  private
  public :: build_dof_mode, samemn_csr_count, build_samemn_csr, build_samemn_csr_sparse

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
  !> the CSR arrays. The diagonal is always counted (needed for ILU) even when structurally zero.
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
  !> is always emitted. ia has length NN+1.
  !>
  !> The augmented operator has zero diagonal entries on the Lagrange-multiplier rows (a saddle
  !> point), and the coordinate-singularity (Zernike) blocks can produce zero pivots too. An
  !> unpivoted ILU(0) needs nonzero pivots, so a near-zero diagonal is boosted to the scale of its
  !> row. This only changes the preconditioner (not the operator GMRES solves), so it affects the
  !> iteration count, never the solution.
  subroutine build_samemn_csr(NN, dMA, dMD, mu, dof_mode, aval, ja, ia)
    integer, intent(in) :: NN
    real(rkind), intent(in) :: dMA(0:NN, 0:NN), dMD(0:NN, 0:NN), mu
    integer, intent(in) :: dof_mode(1:NN)
    real(rkind), intent(out) :: aval(:)
    integer, intent(out) :: ja(:), ia(:)

    integer :: i, j, k, kdiag
    real(rkind) :: v, rowmax

    k = 0
    do i = 1, NN
      ia(i) = k + 1
      kdiag = 0
      rowmax = 0.0_rkind
      do j = 1, NN
        if (dof_mode(i) /= dof_mode(j)) cycle
        v = dMA(i, j) - mu * dMD(i, j)
        if (j == i .or. v /= 0.0_rkind) then
          k = k + 1
          aval(k) = v
          ja(k) = j
          if (j == i) kdiag = k
          if (j /= i) rowmax = max(rowmax, abs(v))
        end if
      end do
      ! regularize a (near-)zero pivot so the unpivoted ILU(0) stays well defined
      if (abs(aval(kdiag)) <= 1.0e-12_rkind * (1.0_rkind + rowmax)) then
        aval(kdiag) = max(rowmax, 1.0_rkind)
      end if
    end do
    ia(NN + 1) = k + 1
  end subroutine build_samemn_csr

  !> \brief Build the same-(m,n) preconditioner CSR directly from the flux-surface-averaged metric,
  !> WITHOUT forming the dense dMA/dMD (this is the memory win for the large vacuum volume). Ports
  !> SPEC's spsint (reduced integrals from the (0,0) metric harmonic) and spsmat (per-mode block
  !> assembly). Stellarator-symmetric only; the caller falls back to the dense extraction otherwise.
  !> guvijsave is the metric at the quadrature nodes (Ntz,3,3,lquad). The CSR comes out in DOF order
  !> with columns ascending per row, ready for ilu0_factor. Because the same-(m,n) block couples only
  !> through the (0,0) harmonic, the result equals the same-(m,n) block of the dense (dMA - mu*dMD).
  subroutine build_samemn_csr_sparse(lvol, NN, Lcoord, mu, guvijsave, ntz, lquad, aval, ja, ia, nnz_out)
    integer, intent(in) :: lvol, NN, ntz, lquad
    logical, intent(in) :: Lcoord
    real(rkind), intent(in) :: mu, guvijsave(ntz, 3, 3, lquad)
    real(rkind), intent(out) :: aval(:)
    integer, intent(out) :: ja(:), ia(:)
    integer, intent(out) :: nnz_out

    integer :: lr, lldof, mi, ni, ii, ll, pp, ll1, pp1, jq, idx, idxr, kk, nb, k, kdiag
    real(rkind) :: lss, jw, sbar, Tl, Dl, Tp, Dp, factorcc, factorss, rowmax, vdiag, wA, wD
    real(rkind) :: gss, gtt, gtz, gzz
    real(rkind), allocatable :: basis(:, :, :, :)
    real(rkind), allocatable :: DToocc(:, :, :), TTssss(:, :, :), DDttcc(:, :, :), DDtzcc(:, :, :), DDzzcc(:, :, :)
    real(rkind), allocatable :: TTdata(:, :, :), TTMdata(:, :)
    ! per-row scratch: candidate (column, value) pairs before sorting
    integer, allocatable :: rmode(:), rll(:), rpot(:), col(:), ord(:)
    real(rkind), allocatable :: val(:)

    lr = Lrad(lvol)
    if (Lcoord) then
      lldof = (lr - mod(lr, 2)) / 2
    else
      lldof = lr
    end if

    ! ---- reduced integrals from the flux-averaged metric (spsint) ----
    allocate(basis(0:lr, 0:mpol, 0:1, lquad))
    allocate(DToocc(0:lldof, 0:lldof, 0:mpol), TTssss(0:lldof, 0:lldof, 0:mpol))
    allocate(DDttcc(0:lldof, 0:lldof, 0:mpol), DDtzcc(0:lldof, 0:lldof, 0:mpol), DDzzcc(0:lldof, 0:lldof, 0:mpol))
    DToocc = 0.0_rkind; TTssss = 0.0_rkind; DDttcc = 0.0_rkind; DDtzcc = 0.0_rkind; DDzzcc = 0.0_rkind

    do jq = 1, lquad
      lss = gaussianabscissae(jq, lvol)
      sbar = (lss + 1.0_rkind) * 0.5_rkind
      if (Lcoord) then
        call get_zernike(sbar, lr, mpol, basis(0:lr, 0:mpol, 0:1, jq))
      else
        call get_cheby(lss, lr, basis(0:lr, 0, 0:1, jq))
      end if
    end do

    do jq = 1, lquad
      jw = gaussianweight(jq, lvol)
      gss = sum(guvijsave(1:ntz, 1, 1, jq)) / real(ntz, rkind)
      gtt = sum(guvijsave(1:ntz, 2, 2, jq)) / real(ntz, rkind)
      gtz = sum(guvijsave(1:ntz, 2, 3, jq)) / real(ntz, rkind)
      gzz = sum(guvijsave(1:ntz, 3, 3, jq)) / real(ntz, rkind)
      do mi = 0, mpol
        do ll = 0, lr
          do pp = 0, lr
            if (Lcoord) then
              if (ll < mi .or. pp < mi) cycle
              if (mod(ll + mi, 2) /= 0 .or. mod(pp + mi, 2) /= 0) cycle
              ll1 = (ll - mod(ll, 2)) / 2
              pp1 = (pp - mod(pp, 2)) / 2
              Tl = basis(ll, mi, 0, jq); Dl = basis(ll, mi, 1, jq) * 0.5_rkind
              Tp = basis(pp, mi, 0, jq); Dp = basis(pp, mi, 1, jq) * 0.5_rkind
            else
              if (mi > 0) cycle
              ll1 = ll; pp1 = pp
              Tl = basis(ll, 0, 0, jq); Dl = basis(ll, 0, 1, jq)
              Tp = basis(pp, 0, 0, jq); Dp = basis(pp, 0, 1, jq)
            end if
            DToocc(ll1, pp1, mi) = DToocc(ll1, pp1, mi) + Dl * Tp * jw          ! goomne = 1
            TTssss(ll1, pp1, mi) = TTssss(ll1, pp1, mi) + Tl * Tp * gss * jw
            DDttcc(ll1, pp1, mi) = DDttcc(ll1, pp1, mi) + Dl * Dp * gtt * jw
            DDtzcc(ll1, pp1, mi) = DDtzcc(ll1, pp1, mi) + Dl * Dp * gtz * jw
            DDzzcc(ll1, pp1, mi) = DDzzcc(ll1, pp1, mi) + Dl * Dp * gzz * jw
          end do
        end do
      end do
    end do
    DToocc = DToocc * pi2pi2nfphalf; TTssss = TTssss * pi2pi2nfphalf
    DDttcc = DDttcc * pi2pi2nfphalf; DDtzcc = DDtzcc * pi2pi2nfphalf; DDzzcc = DDzzcc * pi2pi2nfphalf

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

    ! ---- assemble CSR row by row (spsmat formulas), columns ascending per row ----
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
      if (Lcoord) then; idxr = mi; else; idxr = 0; end if
      if (ii == 1) then; factorcc = 2.0_rkind; factorss = 0.0_rkind
      else; factorcc = 1.0_rkind; factorss = 1.0_rkind; end if
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
          wA = 2.0_rkind * ni * ni * TTssss(ll1, pp1, idxr) * factorss + 2.0_rkind * DDzzcc(ll1, pp1, idxr) * factorcc
          call push(Ate(lvol, 0, ii)%i(pp), wA) ! Htete = 0
          wA = 2.0_rkind * ni * mi * TTssss(ll1, pp1, idxr) * factorss - 2.0_rkind * DDtzcc(pp1, ll1, idxr) * factorcc
          wD = (-DToocc(pp1, ll1, idxr) + DToocc(ll1, pp1, idxr)) * factorcc
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
          wA = 2.0_rkind * mi * ni * TTssss(ll1, pp1, idxr) * factorss - 2.0_rkind * DDtzcc(ll1, pp1, idxr) * factorcc
          wD = (DToocc(pp1, ll1, idxr) - DToocc(ll1, pp1, idxr)) * factorcc
          call push(Ate(lvol, 0, ii)%i(pp), wA - mu * wD)
          wA = 2.0_rkind * mi * mi * TTssss(ll1, pp1, idxr) * factorss + 2.0_rkind * DDttcc(ll1, pp1, idxr) * factorcc
          call push(Aze(lvol, 0, ii)%i(pp), wA) ! Hzeze = 0
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
      kdiag = 0; rowmax = 0.0_rkind; vdiag = 0.0_rkind
      do pp = 1, nb
        k = k + 1
        aval(k) = val(ord(pp)); ja(k) = col(ord(pp))
        if (ja(k) == idx) then; kdiag = k; vdiag = aval(k); else; rowmax = max(rowmax, abs(aval(k))); end if
      end do
      if (kdiag == 0) then
        k = k + 1; aval(k) = max(rowmax, 1.0_rkind); ja(k) = idx
        call fix_tail_order(aval, ja, ia(idx), k)
      else if (abs(vdiag) <= 1.0e-12_rkind * (1.0_rkind + rowmax)) then
        aval(kdiag) = max(rowmax, 1.0_rkind)
      end if
    end do
    ia(NN + 1) = k + 1
    nnz_out = k

    deallocate(basis, DToocc, TTssss, DDttcc, DDtzcc, DDzzcc, TTdata, TTMdata, rmode, rll, rpot, col, val, ord)

  contains

    subroutine setrow(id, iim, illv, ipot)
      integer, intent(in) :: id, iim, illv, ipot
      if (id /= 0) then
        rmode(id) = iim; rll(id) = illv; rpot(id) = ipot
      end if
    end subroutine setrow

    ! append one (column, value) to the current row's buffer; skip absent DOFs and structural zeros
    ! (but always keep the diagonal so ILU has a pivot slot)
    subroutine push(jd, v)
      integer, intent(in) :: jd
      real(rkind), intent(in) :: v
      if (jd == 0) return
      if (v == 0.0_rkind .and. jd /= idx) return
      nb = nb + 1
      col(nb) = jd; val(nb) = v
    end subroutine push

  end subroutine build_samemn_csr_sparse

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
