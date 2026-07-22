! SPDX-FileCopyrightText: 2026 Proxima Fusion
! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module matrices_mod
  use precisions_mod, only: rkind
  use input_list_mod, only: Wmacros, Wmatrix, Wmtrxhs, Mpol, enforce_stell_sym, Lfreebound, lvcvacuum
  use vc_matrix_mod, only: vc_subtract_dvc
  ! dBdX is NOT imported: matrix() takes the one field it needs (dBdX_L) as an explicit argument
  ! instead (see matrix's signature), since it is read inside this module's own live !$OMP
  ! region and dBdX as a whole is declared THREADPRIVATE elsewhere.
  use allglobal_mod, only: im, in, NAdof, dMA, dMD, Mvol, Ate, Ato, Aze, Azo, Lma, Lmb, Lmc, Lmd, Lme, Lmf, Lmg, Lmh, &
      Lcoordinatesingularity, TT, RTT, RTM, DToocc, DToocs, DToosc, DTooss, TTsscc, TTsscs, TTsssc, TTssss, &
      TDstcc, TDstcs, TDstsc, TDstss, TDszcc, TDszcs, TDszsc, TDszss, DDttcc, DDttcs, DDttsc, DDttss, &
      DDtzcc, DDtzcs, DDtzsc, DDtzss, DDzzcc, DDzzcs, DDzzsc, DDzzss, &
      ! Tss/Tsc/Dts/Dtc/Dzs/Dzc are read only in mtrxhs, from its per-thread workspace (pws), so
      ! they are not imported here; mtrxhs declares local pointers of the same name.
      Ttc, Tts, Tzc, Tzs, Lmavalue, Lmbvalue, Lmcvalue, Lmdvalue, Lmevalue, Lmfvalue, Lmgvalue, Lmhvalue
  use memory_allocation_mod, only: sallocate
  use precisions_mod, only: rkind

  implicit none(type, external)

  private
  public :: matrix, mtrxhs

contains

  subroutine matrix(lvol, mn, lrad, dBdX_L)
    integer, intent(in) :: lvol, mn, lrad
    logical, intent(in) :: dBdX_L ! value of dBdX%L for this call

    integer :: NN, ii, jj, ll, kk, pp, ll1, pp1, mi, ni, mj, nj, mimj, minj, nimj, ninj, mjmi, mjni, njmi, njni, id, jd

    real(rkind) :: Wtete, Wteto, Wtote, Wtoto
    real(rkind) :: Wteze, Wtezo, Wtoze, Wtozo
    real(rkind) :: Wzete, Wzeto, Wzote, Wzoto
    real(rkind) :: Wzeze, Wzezo, Wzoze, Wzozo

    real(rkind) :: Htete, Hteto, Htote, Htoto
    real(rkind) :: Hteze, Htezo, Htoze, Htozo
    real(rkind) :: Hzete, Hzeto, Hzote, Hzoto
    real(rkind) :: Hzeze, Hzezo, Hzoze, Hzozo

    real(rkind), allocatable :: TTdata(:, :, :), TTMdata(:, :)! queues to construct sparse matrices

    NN = NAdof(lvol)! shorthand

    dMA(0:NN, 0:NN) = 0.0_rkind
    dMD(0:NN, 0:NN) = 0.0_rkind

    call sallocate(TTdata, 0, lrad, 0, mpol, 0, 1, 0.0_rkind)
    call sallocate(TTMdata, 0, lrad, 0, mpol, 0.0_rkind)

    ! fill in Zernike/Chebyshev polynomials depending on Lcooridnatesingularity
    if (Lcoordinatesingularity) then
      TTdata(0:lrad, 0:mpol, 0:1) = RTT(0:lrad, 0:mpol, 0:1, 0)
      TTMdata(0:lrad, 0:mpol) = RTM(0:lrad, 0:mpol)
    else
      do ii = 0, mpol
        TTdata(0:lrad, ii, 0:1) = TT(0:lrad, 0:1, 0)
        TTMdata(0:lrad, ii) = TT(0:lrad, 0, 0)
      end do
    end if

    if (enforce_stell_sym) then
      !$OMP PARALLEL DO &
      !$OMP& PRIVATE(ii,jj,ll,pp,mi,ni,mj,nj,mimj,minj,nimj,ninj,ll1,pp1,Wtete,Wzete,Wteze,Wzeze,Htete,Hzete,Hteze,Hzeze,id,jd,kk)  &
      !$OMP& SHARED(lvol,lrad)
      do ii = 1, mn
        mi = im(ii)
        ni = in(ii)

        do jj = 1, mn
          mj = im(jj)
          nj = in(jj)
          mimj = mi * mj
          minj = mi * nj
          nimj = ni * mj
          ninj = ni * nj

          do ll = 0, lrad

            do pp = 0, lrad

              if (Lcoordinatesingularity) then
                if (ll < mi .or. pp < mj) cycle ! rule out zero components of Zernike
                if (mod(ll + mi, 2) + mod(pp + mj, 2) > 0) cycle ! rule out zero components of Zernike
                ll1 = (ll - mod(ll, 2)) / 2 ! shrinked dof for Zernike
                pp1 = (pp - mod(pp, 2)) / 2 ! shrinked dof for Zernike
              else
                ll1 = ll
                pp1 = pp
              end if

              Wtete = +2 * ninj * TTssss(ll1, pp1, ii, jj) - 2 * ni * TDszsc(ll1, pp1, ii, jj) - 2 * nj * TDszsc(pp1, ll1&
                  &, jj, ii) + 2 * DDzzcc(ll1, pp1, ii, jj)
              Wzete = +2 * nimj * TTssss(ll1, pp1, ii, jj) + 2 * ni * TDstsc(ll1, pp1, ii, jj) - 2 * mj * TDszsc(pp1, ll1&
                  &, jj, ii) - 2 * DDtzcc(pp1, ll1, jj, ii)
              Wteze = +2 * minj * TTssss(ll1, pp1, ii, jj) + 2 * nj * TDstsc(pp1, ll1, jj, ii) - 2 * mi * TDszsc(ll1, pp1&
                  &, ii, jj) - 2 * DDtzcc(ll1, pp1, ii, jj)
              Wzeze = +2 * mimj * TTssss(ll1, pp1, ii, jj) + 2 * mi * TDstsc(ll1, pp1, ii, jj) + 2 * mj * TDstsc(pp1, ll1&
                  &, jj, ii) + 2 * DDttcc(ll1, pp1, ii, jj)

              Htete = 0.0_rkind
              Hzete = -DToocc(pp1, ll1, jj, ii) + DToocc(ll1, pp1, ii, jj)
              Hteze = +DToocc(pp1, ll1, jj, ii) - DToocc(ll1, pp1, ii, jj)
              Hzeze = 0.0_rkind

              id = Ate(lvol, 0, ii)%i(ll)
              jd = Ate(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wtete
              dMD(id, jd) = Htete
              jd = Aze(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzete
              dMD(id, jd) = Hzete
              id = Aze(lvol, 0, ii)%i(ll)
              jd = Ate(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wteze
              dMD(id, jd) = Hteze
              jd = Aze(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzeze
              dMD(id, jd) = Hzeze

            end do
          end do
        end do

        if (dBdX_L) cycle

        if (Lcoordinatesingularity .and. ii == 1) then
          kk = 1
        else
          kk = 0
        end if

        do ll = 0, lrad
          id = Ate(lvol, 0, ii)%i(ll)
          jd = Lma(lvol, ii)
          dMA(id, jd) = +TTMdata(ll, mi)
          id = Aze(lvol, 0, ii)%i(ll)
          jd = Lmb(lvol, ii)
          dMA(id, jd) = +TTdata(ll, mi, kk)
          if (ii > 1) then
            id = Ate(lvol, 0, ii)%i(ll)
            jd = Lme(lvol, ii)
            dMA(id, jd) = -ni * TTdata(ll, mi, 1)
            id = Aze(lvol, 0, ii)%i(ll)
            jd = Lme(lvol, ii)
            dMA(id, jd) = -mi * TTdata(ll, mi, 1)
          else
            id = Ate(lvol, 0, ii)%i(ll)
            jd = Lmg(lvol, ii)
            dMA(id, jd) = +TTdata(ll, mi, 1)
            id = Aze(lvol, 0, ii)%i(ll)
            jd = Lmh(lvol, ii)
            dMA(id, jd) = +TTdata(ll, mi, 1)
          end if

        end do

        do pp = 0, lrad
          id = Lma(lvol, ii)
          jd = Ate(lvol, 0, ii)%i(pp)
          dMA(id, jd) = +TTMdata(pp, mi)
          id = Lmb(lvol, ii)
          jd = Aze(lvol, 0, ii)%i(pp)
          dMA(id, jd) = +TTdata(pp, mi, kk)
          if (ii > 1) then
            id = Lme(lvol, ii)
            jd = Ate(lvol, 0, ii)%i(pp)
            dMA(id, jd) = -ni * TTdata(pp, mi, 1)
            id = Lme(lvol, ii)
            jd = Aze(lvol, 0, ii)%i(pp)
            dMA(id, jd) = -mi * TTdata(pp, mi, 1)
          else
            id = Lmg(lvol, ii)
            jd = Ate(lvol, 0, ii)%i(pp)
            dMA(id, jd) = +TTdata(pp, mi, 1)
            id = Lmh(lvol, ii)
            jd = Aze(lvol, 0, ii)%i(pp)
            dMA(id, jd) = +TTdata(pp, mi, 1)
          end if
        end do

      end do
      !$OMP END PARALLEL DO

    else ! .not.enforce_stell_sym

      !$OMP PARALLEL DO &
      !$OMP& PRIVATE(ii,jj,ll,pp,mi,ni,mj,nj,mjmi,mjni,njmi,njni,ll1,pp1,Wtete,Wzete,Wteze,Wzeze,Htete,Hzete,Hteze,Hzeze,Wteto,Wzeto,Wtezo,Wzezo,Hteto,Hzeto,Htezo,Hzezo,Wtote,Wzote,Wtoze,Wzoze,Htote,Hzote,Htoze,Hzoze,Wtoto,Wzoto,Wtozo,Wzozo,Htoto,Hzoto,Htozo,Hzozo,id,jd,kk) &
      !$OMP& SHARED(lvol,lrad)
      do ii = 1, mn
        mi = im(ii)
        ni = in(ii)

        do jj = 1, mn
          mj = im(jj)
          nj = in(jj)
          mjmi = mi * mj
          mjni = ni * mj
          njmi = mi * nj
          njni = ni * nj

          do ll = 0, lrad

            do pp = 0, lrad

              if (Lcoordinatesingularity) then
                if (ll < mi .or. pp < mj) cycle ! rule out zero components of Zernike
                if (mod(ll + mi, 2) + mod(pp + mj, 2) > 0) cycle ! rule out zero components of Zernike
                ll1 = (ll - mod(ll, 2)) / 2 ! shrinked dof for Zernike
                pp1 = (pp - mod(pp, 2)) / 2 ! shrinked dof for Zernike
              else
                ll1 = ll
                pp1 = pp
              end if

              Wtete = 2 * (+njni * TTssss(pp1, ll1, jj, ii) - nj * TDszsc(pp1, ll1, jj, ii) - ni * TDszsc(ll1, pp1, ii,&
                  & jj) + DDzzcc(pp1, ll1, jj, ii))
              Wtote = 2 * (-njni * TTsscs(pp1, ll1, jj, ii) + nj * TDszcc(pp1, ll1, jj, ii) - ni * TDszss(ll1, pp1, ii,&
                  & jj) + DDzzsc(pp1, ll1, jj, ii))
              Wzete = 2 * (+mjni * TTssss(pp1, ll1, jj, ii) - mj * TDszsc(pp1, ll1, jj, ii) + ni * TDstsc(ll1, pp1, ii,&
                  & jj) - DDtzcc(pp1, ll1, jj, ii))
              Wzote = 2 * (-mjni * TTsscs(pp1, ll1, jj, ii) + mj * TDszcc(pp1, ll1, jj, ii) + ni * TDstss(ll1, pp1, ii,&
                  & jj) - DDtzsc(pp1, ll1, jj, ii))

              Wteto = 2 * (-njni * TTsssc(pp1, ll1, jj, ii) - nj * TDszss(pp1, ll1, jj, ii) + ni * TDszcc(ll1, pp1, ii,&
                  & jj) + DDzzcs(pp1, ll1, jj, ii))
              Wtoto = 2 * (+njni * TTsscc(pp1, ll1, jj, ii) + nj * TDszcs(pp1, ll1, jj, ii) + ni * TDszcs(ll1, pp1, ii,&
                  & jj) + DDzzss(pp1, ll1, jj, ii))
              Wzeto = 2 * (-mjni * TTsssc(pp1, ll1, jj, ii) - mj * TDszss(pp1, ll1, jj, ii) - ni * TDstcc(ll1, pp1, ii,&
                  & jj) - DDtzcs(pp1, ll1, jj, ii))
              Wzoto = 2 * (+mjni * TTsscc(pp1, ll1, jj, ii) + mj * TDszcs(pp1, ll1, jj, ii) - ni * TDstcs(ll1, pp1, ii,&
                  & jj) - DDtzss(pp1, ll1, jj, ii))

              Wteze = 2 * (+njmi * TTssss(pp1, ll1, jj, ii) + nj * TDstsc(pp1, ll1, jj, ii) - mi * TDszsc(ll1, pp1, ii,&
                  & jj) - DDtzcc(pp1, ll1, jj, ii))
              Wtoze = 2 * (-njmi * TTsscs(pp1, ll1, jj, ii) - nj * TDstcc(pp1, ll1, jj, ii) - mi * TDszss(ll1, pp1, ii,&
                  & jj) - DDtzsc(pp1, ll1, jj, ii))
              Wzeze = 2 * (+mjmi * TTssss(pp1, ll1, jj, ii) + mj * TDstsc(pp1, ll1, jj, ii) + mi * TDstsc(ll1, pp1, ii,&
                  & jj) + DDttcc(pp1, ll1, jj, ii))
              Wzoze = 2 * (-mjmi * TTsscs(pp1, ll1, jj, ii) - mj * TDstcc(pp1, ll1, jj, ii) + mi * TDstss(ll1, pp1, ii,&
                  & jj) + DDttsc(pp1, ll1, jj, ii))

              Wtezo = 2 * (-njmi * TTsssc(pp1, ll1, jj, ii) + nj * TDstss(pp1, ll1, jj, ii) + mi * TDszcc(ll1, pp1, ii,&
                  & jj) - DDtzcs(pp1, ll1, jj, ii))
              Wtozo = 2 * (+njmi * TTsscc(pp1, ll1, jj, ii) - nj * TDstcs(pp1, ll1, jj, ii) + mi * TDszcs(ll1, pp1, ii,&
                  & jj) - DDtzss(pp1, ll1, jj, ii))
              Wzezo = 2 * (-mjmi * TTsssc(pp1, ll1, jj, ii) + mj * TDstss(pp1, ll1, jj, ii) - mi * TDstcc(ll1, pp1, ii,&
                  & jj) + DDttcs(pp1, ll1, jj, ii))
              Wzozo = 2 * (+mjmi * TTsscc(pp1, ll1, jj, ii) - mj * TDstcs(pp1, ll1, jj, ii) - mi * TDstcs(ll1, pp1, ii,&
                  & jj) + DDttss(pp1, ll1, jj, ii))

              Htete = 0.0_rkind
              Htote = 0.0_rkind
              Hzete = -DToocc(pp1, ll1, jj, ii) + DToocc(ll1, pp1, ii, jj)
              Hzote = -DToosc(pp1, ll1, jj, ii) + DToocs(ll1, pp1, ii, jj)

              Hteto = 0.0_rkind
              Htoto = 0.0_rkind
              Hzeto = -DToocs(pp1, ll1, jj, ii) + DToosc(ll1, pp1, ii, jj)
              Hzoto = -DTooss(pp1, ll1, jj, ii) + DTooss(ll1, pp1, ii, jj)

              Hteze = +DToocc(pp1, ll1, jj, ii) - DToocc(ll1, pp1, ii, jj)
              Htoze = +DToosc(pp1, ll1, jj, ii) - DToocs(ll1, pp1, ii, jj)
              Hzeze = 0.0_rkind
              Hzoze = 0.0_rkind

              Htezo = +DToocs(pp1, ll1, jj, ii) - DToosc(ll1, pp1, ii, jj)
              Htozo = +DTooss(pp1, ll1, jj, ii) - DTooss(ll1, pp1, ii, jj)
              Hzezo = 0.0_rkind
              Hzozo = 0.0_rkind

              id = Ate(lvol, 0, ii)%i(ll)
              jd = Ate(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wtete
              dMD(id, jd) = Htete
              jd = Ato(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wtote
              dMD(id, jd) = Htote
              jd = Aze(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzete
              dMD(id, jd) = Hzete
              jd = Azo(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzote
              dMD(id, jd) = Hzote
              id = Ato(lvol, 0, ii)%i(ll)
              jd = Ate(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wteto
              dMD(id, jd) = Hteto
              jd = Ato(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wtoto
              dMD(id, jd) = Htoto
              jd = Aze(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzeto
              dMD(id, jd) = Hzeto
              jd = Azo(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzoto
              dMD(id, jd) = Hzoto
              id = Aze(lvol, 0, ii)%i(ll)
              jd = Ate(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wteze
              dMD(id, jd) = Hteze
              jd = Ato(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wtoze
              dMD(id, jd) = Htoze
              jd = Aze(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzeze
              dMD(id, jd) = Hzeze
              jd = Azo(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzoze
              dMD(id, jd) = Hzoze
              id = Azo(lvol, 0, ii)%i(ll)
              jd = Ate(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wtezo
              dMD(id, jd) = Htezo
              jd = Ato(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wtozo
              dMD(id, jd) = Htozo
              jd = Aze(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzezo
              dMD(id, jd) = Hzezo
              jd = Azo(lvol, 0, jj)%i(pp)
              dMA(id, jd) = Wzozo
              dMD(id, jd) = Hzozo

            end do

          end do

        end do

        if (dBdX_L) cycle

        if (Lcoordinatesingularity .and. ii == 1) then
          kk = 1
        else
          kk = 0
        end if

        do ll = 0, lrad ! Chebyshev polynomial

          id = Ate(lvol, 0, ii)%i(ll)
          jd = Lma(lvol, ii)
          dMA(id, jd) = TTMdata(ll, mi)
          id = Aze(lvol, 0, ii)%i(ll)
          jd = Lmb(lvol, ii)
          dMA(id, jd) = TTdata(ll, mi, kk)
          if (ii > 1) then
            id = Ato(lvol, 0, ii)%i(ll)
            jd = Lmc(lvol, ii)
            dMA(id, jd) = TTMdata(ll, mi)
            id = Azo(lvol, 0, ii)%i(ll)
            jd = Lmd(lvol, ii)
            dMA(id, jd) = TTdata(ll, mi, 0)
            id = Ate(lvol, 0, ii)%i(ll)
            jd = Lme(lvol, ii)
            dMA(id, jd) = -ni * TTdata(ll, mi, 1)
            id = Aze(lvol, 0, ii)%i(ll)
            jd = Lme(lvol, ii)
            dMA(id, jd) = -mi * TTdata(ll, mi, 1)
            id = Ato(lvol, 0, ii)%i(ll)
            jd = Lmf(lvol, ii)
            dMA(id, jd) = +ni * TTdata(ll, mi, 1)
            id = Azo(lvol, 0, ii)%i(ll)
            jd = Lmf(lvol, ii)
            dMA(id, jd) = +mi * TTdata(ll, mi, 1)
          else
            id = Ate(lvol, 0, ii)%i(ll)
            jd = Lmg(lvol, ii)
            dMA(id, jd) = +TTdata(ll, mi, 1)
            id = Aze(lvol, 0, ii)%i(ll)
            jd = Lmh(lvol, ii)
            dMA(id, jd) = +TTdata(ll, mi, 1)
          end if

        end do

        do pp = 0, lrad
          id = Lma(lvol, ii)
          jd = Ate(lvol, 0, ii)%i(pp)
          dMA(id, jd) = TTMdata(pp, mi)
          id = Lmb(lvol, ii)
          jd = Aze(lvol, 0, ii)%i(pp)
          dMA(id, jd) = TTdata(pp, mi, kk)
          if (ii > 1) then
            id = Lmc(lvol, ii)
            jd = Ato(lvol, 0, ii)%i(pp)
            dMA(id, jd) = TTMdata(pp, mi)
            id = Lmd(lvol, ii)
            jd = Azo(lvol, 0, ii)%i(pp)
            dMA(id, jd) = TTdata(pp, mi, 0)
            id = Lme(lvol, ii)
            jd = Ate(lvol, 0, ii)%i(pp)
            dMA(id, jd) = -ni * TTdata(pp, mi, 1)
            id = Lme(lvol, ii)
            jd = Aze(lvol, 0, ii)%i(pp)
            dMA(id, jd) = -mi * TTdata(pp, mi, 1)
            id = Lmf(lvol, ii)
            jd = Ato(lvol, 0, ii)%i(pp)
            dMA(id, jd) = +ni * TTdata(pp, mi, 1)
            id = Lmf(lvol, ii)
            jd = Azo(lvol, 0, ii)%i(pp)
            dMA(id, jd) = +mi * TTdata(pp, mi, 1)
          else
            id = Lmg(lvol, ii)
            jd = Ate(lvol, 0, ii)%i(pp)
            dMA(id, jd) = +TTdata(pp, mi, 1)
            id = Lmh(lvol, ii)
            jd = Aze(lvol, 0, ii)%i(pp)
            dMA(id, jd) = +TTdata(pp, mi, 1)
          end if
        end do

      end do
      !$OMP END PARALLEL DO

    end if

    ! direct free-boundary solve: embed the virtual-casing operator in the vacuum-volume matrix so
    ! that the Lme/Lmf constraint rows enforce (B.n on the computational boundary) minus (plasma
    ! contribution) = coil field; matrixBG then drops iBns/iBnc from dMG. Only the constraint rows
    ! are touched (compact row storage, cached on the boundary geometry). The perturbed-matrix
    ! case (dBdX_L) is handled matrix-free in get_perturbed_rhs via vc_deriv_rhs_add.
    if (lvcvacuum == 1 .and. Lfreebound == 1 .and. lvol == Mvol .and. .not. dBdX_L) then
      call vc_subtract_dvc(dMA, NN)
    end if

    ! call subroutine matrixBG to construct dMB and dMG
    call matrixBG(lvol, mn, lrad)

    deallocate(TTdata)
    deallocate(TTMdata)

  end subroutine matrix

  subroutine matrixBG(lvol, mn, lrad)
    ! only compute the dMB and dMG matrix for matrix-free mode
    use input_list_mod, only: enforce_stell_sym, lvcvacuum
    use precisions_mod, only: rkind
    use allglobal_mod, only: NAdof, im, in, dMG, dMB, iVnc, iVns, iBnc, iBns, Lme, Lmf, Lmg, Lmh
    implicit none
    integer, intent(in) :: lvol, mn, lrad

    integer :: NN, ii, id, mi, ni

    NN = NAdof(lvol)! shorthand

    dMB(0:NN, 1:2) = 0.0_rkind
    dMG(0:NN) = 0.0_rkind

    if (enforce_stell_sym) then

      do ii = 1, mn
        mi = im(ii)
        ni = in(ii)

        if (ii > 1) then
          id = Lme(lvol, ii)

          ! with the Laplace-virtual-casing matrix the plasma contribution iBns is part of the
          ! operator (dMA - dVC), so only the external (coil) field remains on the right-hand side
          if (lvcvacuum == 1) then
            dMG(id) = -iVns(ii)
          else
            dMG(id) = - (iVns(ii) + iBns(ii))
          end if
        else
          id = Lmg(lvol, ii)

          dMB(id, 1) = -1.0_rkind
          id = Lmh(lvol, ii)

          dMB(id, 2) = +1.0_rkind ! changed sign
        end if

      end do

    else ! .not.enforce_stell_sym

      do ii = 1, mn
        mi = im(ii)
        ni = in(ii)

        if (ii > 1) then
          id = Lme(lvol, ii)

          if (lvcvacuum == 1) then
            dMG(id) = -iVns(ii)
          else
            dMG(id) = - (iVns(ii) + iBns(ii))
          end if
          id = Lmf(lvol, ii)

          if (lvcvacuum == 1) then
            dMG(id) = -iVnc(ii)
          else
            dMG(id) = - (iVnc(ii) + iBnc(ii))
          end if
        else
          id = Lmg(lvol, ii)

          dMB(id, 1) = -1.0_rkind

          id = Lmh(lvol, ii)

          dMB(id, 2) = +1.0_rkind ! changed sign
        end if

      end do

    end if
  end subroutine matrixBG

  subroutine mtrxhs(lvol, mn, lrad, resultA, idx, Lcoordinatesingularity, pws)
    use perturbed_workspace_mod, only: perturbed_rhs_workspace
    integer, intent(in) :: lvol, mn, lrad, idx
    real(rkind), intent(out) :: resultA(0:NAdof(lvol))
    logical, intent(in) :: Lcoordinatesingularity
    ! Perturbed-path only: the per-thread workspace holds the projection arrays that intghs
    ! produced (target: pointer aliasing).
    type(perturbed_rhs_workspace), intent(inout), target :: pws
    ! real(rkind), intent(out) :: resultD(0:NAdof(lvol)) ! Unused

    integer :: NN, ii, ll, ll1, mi, ni, id, kk
    real(rkind) :: Wte, Wto, Wze, Wzo, Hte, Hto, Hze, Hzo
    ! real(rkind), allocatable :: TTdata(:, :, :), TTMdata(:, :)
    real(rkind) :: TTdata(0:lrad, 0:mpol, 0:1), TTMdata(0:lrad, 0:mpol)

    ! Local pointers aliasing the per-thread workspace projection arrays, shadowing the
    ! module-global names used in the body.
    real(rkind), pointer :: Tss(:, :), Dtc(:, :), Dzc(:, :), Tsc(:, :), Dts(:, :), Dzs(:, :)

    Tss => pws%Tss; Dtc => pws%Dtc; Dzc => pws%Dzc
    Tsc => pws%Tsc; Dts => pws%Dts; Dzs => pws%Dzs

    NN = NAdof(lvol)! shorthand

    ! making use of only the zeroth component of dMA
    resultA(0:NN) = 0.0_rkind
    ! resultD(0:NN) = 0.0_rkind

    ! call sallocate(TTdata, 0, lrad, 0, mpol, 0, 1, 0.0_rkind)
    ! call sallocate(TTMdata, 0, lrad, 0, mpol, 0.0_rkind)

    ! fill in Zernike/Chebyshev polynomials depending on Lcooridnatesingularity
    if (Lcoordinatesingularity) then
      TTdata(0:lrad, 0:mpol, 0:1) = RTT(0:lrad, 0:mpol, 0:1, 0)
      TTMdata(0:lrad, 0:mpol) = RTM(0:lrad, 0:mpol)
    else
      do ii = 0, mpol
        TTdata(0:lrad, ii, 0:1) = TT(0:lrad, 0:1, 0)
        TTMdata(0:lrad, ii) = TT(0:lrad, 0, 0)
      end do
    end if

    if (enforce_stell_sym) then
      ! Serial: this loop used to carry its own !$OMP PARALLEL DO, but that nests inside the
      ! outer per-DOF parallel region added in field_mod (dead weight once outer is threaded).
      do ii = 1, mn
        do ll = 0, lrad
          mi = im(ii)
          ni = in(ii)

          if (Lcoordinatesingularity) then
            if (ll < mi) cycle ! rule out zero components of Zernike
            if (mod(ll + mi, 2) > 0) cycle ! rule out zero components of Zernike
            ll1 = (ll - mod(ll, 2)) / 2 ! shrinked dof for Zernike
          else
            ll1 = ll
          end if

          Wte = -2.0_rkind * ni * Tss(ll1, ii) + 2.0_rkind * Dzc(ll1, ii)
          Wze = -2.0_rkind * mi * Tss(ll1, ii) - 2.0_rkind * Dtc(ll1, ii)

          id = Ate(lvol, 0, ii)%i(ll)
          resultA(id) = Wte
          id = Aze(lvol, 0, ii)%i(ll)
          resultA(id) = Wze
        end do
      end do

      !! Seems to be not needed. If so, remove
      ! if (.not.dBdX%L) then
      !   !$OMP PARALLEL DO PRIVATE(ii,mi,ni,ll,ll1,kk,Wte,Wze,Hte,Hze,id) SHARED(mn,lrad,resultA,TTMdata,TTdata) COLLAPSE(2) 
      !   do ii = 1, mn
      !     do ll = 0, lrad
      !       mi = im(ii)
      !       ni = in(ii)

      !       if (Lcoordinatesingularity) then
      !         if (ll < mi) cycle ! rule out zero components of Zernike
      !         if (mod(ll + mi, 2) > 0) cycle ! rule out zero components of Zernike
      !         ll1 = (ll - mod(ll, 2)) / 2 ! shrinked dof for Zernike
      !       else
      !         ll1 = ll
      !       end if

      !       Hte = Ttc(ll1, ii)
      !       Hze = Tzc(ll1, ii)

      !       id = Ate(lvol, 0, ii)%i(ll)
      !       resultD(id) = Hte
      !       id = Aze(lvol, 0, ii)%i(ll)
      !       resultD(id) = Hze

      !       if (Lcoordinatesingularity .and. ii == 1) then
      !         kk = 1
      !       else
      !         kk = 0
      !       end if

      !       ! add Langrange multipliers
      !       id = Ate(lvol, 0, ii)%i(ll)
      !       resultA(id) = resultA(id) + Lmavalue(lvol, ii) * TTMdata(ll, mi)
      !       id = Aze(lvol, 0, ii)%i(ll)
      !       resultA(id) = resultA(id) + Lmbvalue(lvol, ii) * TTdata(ll, mi, kk)
      !       if (ii > 1) then
      !         id = Ate(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmevalue(lvol, ii) * (-ni * TTdata(ll, mi, 1))
      !         id = Aze(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmevalue(lvol, ii) * (-mi * TTdata(ll, mi, 1))
      !       else
      !         id = Ate(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmgvalue(lvol, ii) * TTdata(ll, mi, 1)
      !         id = Aze(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmhvalue(lvol, ii) * TTdata(ll, mi, 1)
      !       end if

      !       id = Lma(lvol, ii)
      !       resultA(id) = resultA(id) + Ate(lvol, idx, ii)%s(ll) * TTMdata(ll, mi)
      !       id = Lmb(lvol, ii)
      !       resultA(id) = resultA(id) + Aze(lvol, idx, ii)%s(ll) * TTdata(ll, mi, kk)
      !       if (ii > 1) then
      !         id = Lme(lvol, ii)
      !         resultA(id) = resultA(id) + Ate(lvol, idx, ii)%s(ll) * (-ni * TTdata(ll, mi, 1))
      !         id = Lme(lvol, ii)
      !         resultA(id) = resultA(id) + Aze(lvol, idx, ii)%s(ll) * (-mi * TTdata(ll, mi, 1))
      !       else
      !         id = Lmg(lvol, ii)
      !         resultA(id) = resultA(id) + Ate(lvol, idx, ii)%s(ll) * TTdata(ll, mi, 1)
      !         id = Lmh(lvol, ii)
      !         resultA(id) = resultA(id) + Aze(lvol, idx, ii)%s(ll) * TTdata(ll, mi, 1)
      !       end if

      !     end do
      !   end do
      !   !$OMP END PARALLEL DO

      ! end if

    else ! .not.enforce_stell_sym
      ! Serial: see note above the enforce_stell_sym loop in this routine.
      do ii = 1, mn
        mi = im(ii)
        ni = in(ii)

        do ll = 0, lrad

          if (Lcoordinatesingularity) then
            if (ll < mi) cycle ! rule out zero components of Zernike
            if (mod(ll + mi, 2) > 0) cycle ! rule out zero components of Zernike
            ll1 = (ll - mod(ll, 2)) / 2 ! shrinked dof for Zernike
          else
            ll1 = ll
          end if

          Wte = -2.0_rkind * ni * Tss(ll1, ii) + 2.0_rkind * Dzc(ll1, ii)
          Wze = -2.0_rkind * mi * Tss(ll1, ii) - 2.0_rkind * Dtc(ll1, ii)
          Wto = +2.0_rkind * ni * Tsc(ll1, ii) + 2.0_rkind * Dzs(ll1, ii)
          Wzo = +2.0_rkind * mi * Tsc(ll1, ii) - 2.0_rkind * Dts(ll1, ii)

          id = Ate(lvol, 0, ii)%i(ll)
          resultA(id) = Wte
          id = Aze(lvol, 0, ii)%i(ll)
          resultA(id) = Wze
          id = Ato(lvol, 0, ii)%i(ll)
          resultA(id) = Wto
          id = Azo(lvol, 0, ii)%i(ll)
          resultA(id) = Wzo
        end do
      end do

      ! if (.not.dBdX%L) then
      !   !$OMP PARALLEL DO PRIVATE(ii,mi,ni,ll,ll1,kk,Wte,Wze,Wzo,Wto,Hte,Hze,Hto,Hzo,id) &
      !   !$OMP& SHARED(mn,lrad,resultA,resultD,TTMdata,TTdata)
      !   do ii = 1, mn
      !     mi = im(ii)
      !     ni = in(ii)

      !     do ll = 0, lrad

      !       if (Lcoordinatesingularity) then
      !         if (ll < mi) cycle ! rule out zero components of Zernike
      !         if (mod(ll + mi, 2) > 0) cycle ! rule out zero components of Zernike
      !         ll1 = (ll - mod(ll, 2)) / 2 ! shrinked dof for Zernike
      !       else
      !         ll1 = ll
      !       end if

      !       Hte = Ttc(ll1, ii)
      !       Hze = Tzc(ll1, ii)
      !       Hto = Tts(ll1, ii)
      !       Hzo = Tzs(ll1, ii)

      !       id = Ate(lvol, 0, ii)%i(ll)
      !       resultD(id) = Hte
      !       id = Aze(lvol, 0, ii)%i(ll)
      !       resultD(id) = Hze
      !       id = Ato(lvol, 0, ii)%i(ll)
      !       resultD(id) = Hto
      !       id = Azo(lvol, 0, ii)%i(ll)
      !       resultD(id) = Hzo

      !       if (Lcoordinatesingularity .and. ii == 1) then
      !         kk = 1
      !       else
      !         kk = 0
      !       end if

      !       ! add Langrange multipliers
      !       id = Ate(lvol, 0, ii)%i(ll)
      !       resultA(id) = resultA(id) + Lmavalue(lvol, ii) * TTMdata(ll, mi)
      !       id = Aze(lvol, 0, ii)%i(ll)
      !       resultA(id) = resultA(id) + Lmbvalue(lvol, ii) * TTdata(ll, mi, kk)
      !       id = Ato(lvol, 0, ii)%i(ll)
      !       resultA(id) = resultA(id) + Lmcvalue(lvol, ii) * TTMdata(ll, mi)
      !       id = Azo(lvol, 0, ii)%i(ll)
      !       resultA(id) = resultA(id) + Lmdvalue(lvol, ii) * TTdata(ll, mi, kk)
      !       if (ii > 1) then
      !         id = Ate(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmevalue(lvol, ii) * (-ni * TTdata(ll, mi, 1))
      !         id = Aze(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmevalue(lvol, ii) * (-mi * TTdata(ll, mi, 1))
      !         id = Ato(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmfvalue(lvol, ii) * (+ni * TTdata(ll, mi, 1))
      !         id = Azo(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmfvalue(lvol, ii) * (+mi * TTdata(ll, mi, 1))
      !       else
      !         id = Ate(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmgvalue(lvol, ii) * TTdata(ll, mi, 1)
      !         id = Aze(lvol, 0, ii)%i(ll)
      !         resultA(id) = resultA(id) + Lmhvalue(lvol, ii) * TTdata(ll, mi, 1)
      !       end if

      !       id = Lma(lvol, ii)
      !       resultA(id) = resultA(id) + Ate(lvol, idx, ii)%s(ll) * TTMdata(ll, mi)
      !       id = Lmb(lvol, ii)
      !       resultA(id) = resultA(id) + Aze(lvol, idx, ii)%s(ll) * TTdata(ll, mi, kk)
      !       id = Lmc(lvol, ii)
      !       resultA(id) = resultA(id) + Ato(lvol, idx, ii)%s(ll) * TTMdata(ll, mi)
      !       id = Lmd(lvol, ii)
      !       resultA(id) = resultA(id) + Azo(lvol, idx, ii)%s(ll) * TTdata(ll, mi, kk)
      !       if (ii > 1) then
      !         id = Lme(lvol, ii)
      !         resultA(id) = resultA(id) + Ate(lvol, idx, ii)%s(ll) * (-ni * TTdata(ll, mi, 1))
      !         id = Lme(lvol, ii)
      !         resultA(id) = resultA(id) + Aze(lvol, idx, ii)%s(ll) * (-mi * TTdata(ll, mi, 1))
      !         id = Lmf(lvol, ii)
      !         resultA(id) = resultA(id) + Ato(lvol, idx, ii)%s(ll) * (+ni * TTdata(ll, mi, 1))
      !         id = Lmf(lvol, ii)
      !         resultA(id) = resultA(id) + Azo(lvol, idx, ii)%s(ll) * (+mi * TTdata(ll, mi, 1))
      !       else
      !         id = Lmg(lvol, ii)
      !         resultA(id) = resultA(id) + Ate(lvol, idx, ii)%s(ll) * TTdata(ll, mi, 1)
      !         id = Lmh(lvol, ii)
      !         resultA(id) = resultA(id) + Aze(lvol, idx, ii)%s(ll) * TTdata(ll, mi, 1)
      !       end if
      !     end do
      !   end do
      !   !$OMP END PARALLEL DO
      ! end if
    
    end if

    ! deallocate(TTdata)
    ! deallocate(TTMdata)
  end subroutine mtrxhs
end module matrices_mod
