! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module chebyshev_mod
  use precisions_mod, only: rkind
  use input_list_mod, only: mpol, Wvolume_integrate_chebyshev, Wmacros, enforce_stell_sym
  use allglobal_mod, only: im, gaussianweight, gaussianabscissae, DToocc, DToocs, DToosc, DTooss, TTsscc, TTsscs, TTsssc, TTssss, &
      TDstcc, TDstcs, TDstsc, TDstss, TDszcc, TDszcs, TDszsc, TDszss, DDttcc, DDttcs, DDttsc, DDttss, &
      DDtzcc, DDtzcs, DDtzsc, DDtzss, DDzzcc, DDzzcs, DDzzsc, DDzzss, kija, kijs, goomne, goomno, gssmne, &
      gssmno, gstmne, gstmno, gszmne, gszmno, gttmne, gttmno, gtzmne, gtzmno, gzzmne, gzzmno, &
      Lcoordinatesingularity, pi2pi2nfphalf, Lsavedguvij, dBdX, Nt, Nz, guvijsave, Rij, Zij, sg, inv_jac, guvij
  use memory_allocation_mod, only: sallocate
  use base_functions_mod, only: get_cheby, get_zernike
  use metric_mod, only: compute_metric_quantities, compute_guvijsave

  implicit none(type, external)

  private
  public :: volume_integrate_chebyshev

contains

  subroutine volume_integrate_chebyshev(lquad, mn, lvol, lrad, nt_loc, nz_loc)
    integer, intent(in) :: lquad, mn, lvol, lrad, nt_loc, nz_loc

    integer :: jquad, ll, pp, ll1, pp1, ii, jj, mn2, lp2, mn2_max, lp2_max

    integer :: kka, kks, Lcurvature, ideriv

    real(rkind) :: lss, jthweight, Tl, Dl, Tp, Dp, TlTp, TlDp, DlTp, DlDp, ikda, ikds, imn2, ilrad

    real(rkind) :: foocc, foocs, foosc, fooss
    real(rkind) :: fsscc, fsscs, fsssc, fssss
    real(rkind) :: fstcc, fstcs, fstsc, fstss
    real(rkind) :: fszcc, fszcs, fszsc, fszss
    real(rkind) :: fttcc, fttcs, fttsc, fttss
    real(rkind) :: ftzcc, ftzcs, ftzsc, ftzss
    real(rkind) :: fzzcc, fzzcs, fzzsc, fzzss

    real(rkind) :: sbar
    real(rkind), allocatable :: basis(:, :, :, :)

    integer :: ntz_loc

    ntz_loc = nt_loc * nz_loc

    mn2_max = mn * mn
    lp2_max = (lrad + 1) * (lrad + 1)
    imn2 = 1.0_rkind / real(mn, rkind)
    ilrad = 1.0_rkind / real(lrad + 1, rkind)

    DToocc = 0.0_rkind
    TTssss = 0.0_rkind
    TDstsc = 0.0_rkind
    TDszsc = 0.0_rkind
    DDttcc = 0.0_rkind
    DDtzcc = 0.0_rkind
    DDzzcc = 0.0_rkind

    if (.not.enforce_stell_sym) then
      DToocs = 0.0_rkind
      DToosc = 0.0_rkind
      DTooss = 0.0_rkind

      TTsscc = 0.0_rkind
      TTsscs = 0.0_rkind
      TTsssc = 0.0_rkind

      TDstcc = 0.0_rkind
      TDstcs = 0.0_rkind
      TDstss = 0.0_rkind

      TDszcc = 0.0_rkind
      TDszcs = 0.0_rkind
      TDszss = 0.0_rkind

      DDttcs = 0.0_rkind
      DDttsc = 0.0_rkind
      DDttss = 0.0_rkind

      DDtzcs = 0.0_rkind
      DDtzsc = 0.0_rkind
      DDtzss = 0.0_rkind

      DDzzcs = 0.0_rkind
      DDzzsc = 0.0_rkind
      DDzzss = 0.0_rkind
    end if

    call sallocate(basis, 0, lrad, 0, mpol, 0, 1, 1, lquad, 0.0_rkind)

    if (dBdX%L) then
      Lcurvature = 3
      ideriv = 1
    else
      Lcurvature = 1
      ideriv = 0
    end if

    if (.not.Lsavedguvij) then
      call compute_guvijsave(lquad, lvol, ideriv, Lcurvature, nt_loc, nz_loc, &
          guvijsave, Rij, Zij, sg, inv_jac, guvij)
    end if
    call compute_metric_quantities(lquad, lvol, ntz_loc)! compute metric elements

    do jquad = 1, lquad
      lss = gaussianabscissae(jquad, lvol)
      jthweight = gaussianweight(jquad, lvol)
      sbar = (lss + 1.0_rkind) * 1.0_rkind / 2.0_rkind
      if (Lcoordinatesingularity) then
        call get_zernike(sbar, lrad, mpol, basis(:, :, 0:1, jquad))! use Zernike polynomials
      else
        call get_cheby(lss, lrad, basis(:, 0, 0:1, jquad))
      end if
    end do

    !$OMP PARALLEL DO SHARED(lquad,lrad,lvol,mn,basis,mn2_max,lp2_max) &
    !$OMP& PRIVATE(jquad,lss,jthweight,sbar,mn2,ii,jj,kka,kks,ikds,ikda,lp2,ll,pp,ll1,pp1,Tl,Tp,Dl,Dp,TlTP,Tldp,DlTp,DlDp, &
    !$OMP& foocc,fssss,fstsc,fszsc,fttcc,ftzcc,fzzcc,foocs,foosc,fooss,fsscc,fsscs,fsssc, &
    !$OMP& fstcc,fstcs,fstss,fszcc,fszcs,fszss,fttcs,fttsc,fttss,ftzcs,ftzsc,ftzss,fzzcs,fzzsc,fzzss)
    do mn2 = 1, mn2_max
      ii = mod(mn2 - 1, mn) + 1
      jj = (mn2 - ii) / mn + 1

      do jquad = 1, lquad ! Gaussian quadrature loop
        lss = gaussianabscissae(jquad, lvol)
        jthweight = gaussianweight(jquad, lvol)
        sbar = (lss + 1.0_rkind) * 1.0_rkind / 2.0_rkind

        kks = kijs(ii, jj, 0)
        kka = kija(ii, jj, 0)
        ikds = jthweight / kijs(ii, jj, 1)
        ikda = jthweight / kija(ii, jj, 1)

        foocc = (+goomne(kks, jquad) * abs(ikds) + goomne(kka, jquad) * abs(ikda))
        fssss = (+gssmne(kks, jquad) * abs(ikds) - gssmne(kka, jquad) * abs(ikda))
        fstsc = (+gstmno(kks, jquad) * ikds + gstmno(kka, jquad) * ikda)
        fszsc = (+gszmno(kks, jquad) * ikds + gszmno(kka, jquad) * ikda)
        fttcc = (+gttmne(kks, jquad) * abs(ikds) + gttmne(kka, jquad) * abs(ikda))
        ftzcc = (+gtzmne(kks, jquad) * abs(ikds) + gtzmne(kka, jquad) * abs(ikda))
        fzzcc = (+gzzmne(kks, jquad) * abs(ikds) + gzzmne(kka, jquad) * abs(ikda))

        if (.not.enforce_stell_sym) then
          foocs = (-goomno(kks, jquad) * ikds + goomno(kka, jquad) * ikda)
          foosc = (+goomno(kks, jquad) * ikds + goomno(kka, jquad) * ikda)
          fooss = (+goomne(kks, jquad) * abs(ikds) - goomne(kka, jquad) * abs(ikda))

          fsscc = (+gssmne(kks, jquad) * abs(ikds) + gssmne(kka, jquad) * abs(ikda))
          fsscs = (-gssmno(kks, jquad) * ikds + gssmno(kka, jquad) * ikda)
          fsssc = (+gssmno(kks, jquad) * ikds + gssmno(kka, jquad) * ikda)

          fstcc = (+gstmne(kks, jquad) * abs(ikds) + gstmne(kka, jquad) * abs(ikda))
          fstcs = (-gstmno(kks, jquad) * ikds + gstmno(kka, jquad) * ikda)
          fstss = (+gstmne(kks, jquad) * abs(ikds) - gstmne(kka, jquad) * abs(ikda))

          fszcc = (+gszmne(kks, jquad) * abs(ikds) + gszmne(kka, jquad) * abs(ikda))
          fszcs = (-gszmno(kks, jquad) * ikds + gszmno(kka, jquad) * ikda)
          fszss = (+gszmne(kks, jquad) * abs(ikds) - gszmne(kka, jquad) * abs(ikda))

          fttcs = (-gttmno(kks, jquad) * ikds + gttmno(kka, jquad) * ikda)
          fttsc = (+gttmno(kks, jquad) * ikds + gttmno(kka, jquad) * ikda)
          fttss = (+gttmne(kks, jquad) * abs(ikds) - gttmne(kka, jquad) * abs(ikda))

          ftzcs = (-gtzmno(kks, jquad) * ikds + gtzmno(kka, jquad) * ikda)
          ftzsc = (+gtzmno(kks, jquad) * ikds + gtzmno(kka, jquad) * ikda)
          ftzss = (+gtzmne(kks, jquad) * abs(ikds) - gtzmne(kka, jquad) * abs(ikda))

          fzzcs = (-gzzmno(kks, jquad) * ikds + gzzmno(kka, jquad) * ikda)
          fzzsc = (+gzzmno(kks, jquad) * ikds + gzzmno(kka, jquad) * ikda)
          fzzss = (+gzzmne(kks, jquad) * abs(ikds) - gzzmne(kka, jquad) * abs(ikda))
        end if

        do lp2 = 1, lp2_max
          ll = mod(lp2 - 1, lrad + 1)
          pp = (lp2 - ll - 1) / (lrad + 1)

          if (Lcoordinatesingularity) then
            ll1 = (ll - mod(ll, 2)) / 2 ! shrinked dof for Zernike
            pp1 = (pp - mod(pp, 2)) / 2 ! shrinked dof for Zernike

            if (ll < im(ii)) cycle ! zernike only non-zero for ll>=ii
            if (pp < im(jj)) cycle ! zernike only non-zero for pp>=jj
            if (mod(ll + im(ii), 2)/= 0) cycle ! zernike only non-zero if ll and ii have the same parity
            if (mod(pp + im(jj), 2)/= 0) cycle ! zernike only non-zero if pp and jj have the same parity

            Tl = basis(ll, im(ii), 0, jquad)! use Zernike polynomials
            Dl = basis(ll, im(ii), 1, jquad) * 1.0_rkind / 2.0_rkind ! use Zernike polynomials

            Tp = basis(pp, im(jj), 0, jquad)! use Zernike polynomials
            Dp = basis(pp, im(jj), 1, jquad) * 1.0_rkind / 2.0_rkind ! use Zernike polynomials

          else
            ll1 = ll
            pp1 = pp

            Tl = basis(ll, 0, 0, jquad)
            Dl = basis(ll, 0, 1, jquad)

            Tp = basis(pp, 0, 0, jquad)
            Dp = basis(pp, 0, 1, jquad)
          end if

          TlTp = Tl * Tp * pi2pi2nfphalf
          TlDp = Tl * Dp * pi2pi2nfphalf
          DlTp = Dl * Tp * pi2pi2nfphalf
          DlDp = Dl * Dp * pi2pi2nfphalf

          DToocc(ll1, pp1, ii, jj) = DToocc(ll1, pp1, ii, jj) + DlTp * foocc
          TTssss(ll1, pp1, ii, jj) = TTssss(ll1, pp1, ii, jj) + TlTp * fssss
          TDstsc(ll1, pp1, ii, jj) = TDstsc(ll1, pp1, ii, jj) + TlDp * fstsc
          TDszsc(ll1, pp1, ii, jj) = TDszsc(ll1, pp1, ii, jj) + TlDp * fszsc
          DDttcc(ll1, pp1, ii, jj) = DDttcc(ll1, pp1, ii, jj) + DlDp * fttcc
          DDtzcc(ll1, pp1, ii, jj) = DDtzcc(ll1, pp1, ii, jj) + DlDp * ftzcc
          DDzzcc(ll1, pp1, ii, jj) = DDzzcc(ll1, pp1, ii, jj) + DlDp * fzzcc

          if (.not.enforce_stell_sym) then
            DToocs(ll1, pp1, ii, jj) = DToocs(ll1, pp1, ii, jj) + DlTp * foocs
            DToosc(ll1, pp1, ii, jj) = DToosc(ll1, pp1, ii, jj) + DlTp * foosc
            DTooss(ll1, pp1, ii, jj) = DTooss(ll1, pp1, ii, jj) + DlTp * fooss

            TTsscc(ll1, pp1, ii, jj) = TTsscc(ll1, pp1, ii, jj) + TlTp * fsscc
            TTsscs(ll1, pp1, ii, jj) = TTsscs(ll1, pp1, ii, jj) + TlTp * fsscs
            TTsssc(ll1, pp1, ii, jj) = TTsssc(ll1, pp1, ii, jj) + TlTp * fsssc

            TDstcc(ll1, pp1, ii, jj) = TDstcc(ll1, pp1, ii, jj) + TlDp * fstcc
            TDstcs(ll1, pp1, ii, jj) = TDstcs(ll1, pp1, ii, jj) + TlDp * fstcs
            TDstss(ll1, pp1, ii, jj) = TDstss(ll1, pp1, ii, jj) + TlDp * fstss

            TDszcc(ll1, pp1, ii, jj) = TDszcc(ll1, pp1, ii, jj) + TlDp * fszcc
            TDszcs(ll1, pp1, ii, jj) = TDszcs(ll1, pp1, ii, jj) + TlDp * fszcs
            TDszss(ll1, pp1, ii, jj) = TDszss(ll1, pp1, ii, jj) + TlDp * fszss

            DDttcs(ll1, pp1, ii, jj) = DDttcs(ll1, pp1, ii, jj) + DlDp * fttcs
            DDttsc(ll1, pp1, ii, jj) = DDttsc(ll1, pp1, ii, jj) + DlDp * fttsc
            DDttss(ll1, pp1, ii, jj) = DDttss(ll1, pp1, ii, jj) + DlDp * fttss

            DDtzcs(ll1, pp1, ii, jj) = DDtzcs(ll1, pp1, ii, jj) + DlDp * ftzcs
            DDtzsc(ll1, pp1, ii, jj) = DDtzsc(ll1, pp1, ii, jj) + DlDp * ftzsc
            DDtzss(ll1, pp1, ii, jj) = DDtzss(ll1, pp1, ii, jj) + DlDp * ftzss

            DDzzcs(ll1, pp1, ii, jj) = DDzzcs(ll1, pp1, ii, jj) + DlDp * fzzcs
            DDzzsc(ll1, pp1, ii, jj) = DDzzsc(ll1, pp1, ii, jj) + DlDp * fzzsc
            DDzzss(ll1, pp1, ii, jj) = DDzzss(ll1, pp1, ii, jj) + DlDp * fzzss
          end if
        end do
      end do
    end do
    !$OMP END PARALLEL DO

    deallocate(basis)
  end subroutine volume_integrate_chebyshev
end module chebyshev_mod
