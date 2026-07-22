! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module preset_mod
#include "mpi_precomp.h"
  USE_MPI

  use constants_mod, only: goldenmean, pi2
  use precisions_mod, only: rkind, small
  use io_mod, only: ounit
  use input_list_mod
  use allglobal_mod
  use memory_allocation_mod, only: sallocate, nallocate
  use errors_mod, only: fatal
  use volume_mod, only: lregion
  use mpi_collectives_mod, only: broadcast
  use base_functions_mod, only: get_cheby, get_zernike, get_zernike_rm
  use geometry_mod, only: compute_coordinates
  use numerical_mod, only: gi00ab, getimn, gauleg, set_fftw_plan
  use vector_potential_writer_mod, only: write_vector_potentials
  use coordinate_mod, only: rzaxis
  use field_mod, only: alloc_ntz_arrays
  use omp_lib, only: omp_get_max_threads

  implicit none(type, external)

  private
  public :: initialize_internal_arrays

contains

  subroutine initialize_internal_arrays
    real(rkind) :: cput

    integer :: idof, jk, ll, ii, ideriv, vvol, mi, ni, mj, nj, mk, nk, mimj, ninj, mkmj, nknj, jj, kk, lvol, mm, nn
    integer :: lquad, igauleg, maxIquad, Mrad, Lcurvature, zerdof
    integer :: n_aslot_lo ! lowest Ate/Aze/Ato/Azo derivative slot (see allocation below)
    real(rkind) :: teta, zeta, arg, lss, cszeta(0:1)

    integer :: idx_mode

    integer :: Linitgues = 1 !< controls how initial guess for Beltrami field is constructed
    !< <ul>
    !< <li> only relevant for routines that require an initial guess for the Beltrami fields, such as the SQP and Newton
    !methods,
    !<      or the sparse linear solver
    !< <li> if \c Linitgues = 0, the initial guess for the Beltrami field is trivial
    !< <li> if \c Linitgues = 1, the initial guess for the Beltrami field is an integrable approximation
    !< <li> if \c Linitgues = 2, the initial guess for the Beltrami field is read from file
    !< <li> if \c Linitgues = 3, the initial guess for the Beltrami field will be randomized with the maximum maxrndgues
    !< </ul>
    real(rkind) :: maxrndgues = 1.0_rkind !< the maximum random number of the Beltrami field if \c Linitgues = 3

    ! the following was in global:readin previously
    ! set internal parameters that depend on physicslist

    call fatal(Lfreebound < 0 .or. Lfreebound > 1, __FILE__, "illegal Lfreebound ")

    Mvol = Nvol + Lfreebound

    call sallocate(beltramierror, 1, Mvol, 1, 9, 0.0_rkind)

    mn = 1 + Ntor + Mpol * (2 * Ntor + 1) ! Fourier resolution of interface geometry & vector potential

    call sallocate(im, 1, mn, 0)
    call sallocate(in, 1, mn, 0)

    call gi00ab(Mpol, Ntor, Nfp, mn, im(1:mn), in(1:mn))! this sets the im and in mode identification arrays

    call nallocate(halfmm, 1, mn)
    halfmm = im(1:mn) * 1.0_rkind / 2.0_rkind

    lMpol = 4 * Mpol
    lNtor = 4 * Ntor ! extra-enhanced resolution for metrics

    mne = 1 + lNtor + lMpol * (2 * lNtor + 1) ! resolution of metrics: enhanced resolution

    call sallocate(ime, 1, mne, 0)
    call sallocate(ine, 1, mne, 0)

    call gi00ab(lMpol, lNtor, Nfp, mne, ime(1:mne), ine(1:mne))

    sMpol = iMpol
    sNtor = iNtor

    if (iMpol <= 0) sMpol = Mpol - iMpol
    if (iNtor <= 0) sNtor = Ntor - iNtor
    if (Ntor == 0) sNtor = 0

    mns = 1 + sNtor + sMpol * (2 * sNtor + 1) ! resolution of straight-field line transformation on interfaces

    call sallocate(ims, 1, mns, 0)
    call sallocate(ins, 1, mns, 0)

    call gi00ab(sMpol, sNtor, Nfp, mns, ims(1:mns), ins(1:mns))! note that the field periodicity factor is included in
    ! ins

    ! set internal parameters that depend on numericlist

    ! set internal parameters that depend on globallist

    ! set internal parameters that depend on diagnosticslist

    if (Lcheck == 5) then
      nPpts = 0 ! will check Hessian using finite-differences
    end if

    call sallocate(iRbc, 1, mn, 0, Mvol, 0.0_rkind)! interface Fourier harmonics
    call sallocate(iZbs, 1, mn, 0, Mvol, 0.0_rkind)
    call sallocate(iRbs, 1, mn, 0, Mvol, 0.0_rkind)
    call sallocate(iZbc, 1, mn, 0, Mvol, 0.0_rkind)

    call sallocate(iVns, 1, mn, 0.0_rkind)
    call sallocate(iBns, 1, mn, 0.0_rkind)
    call sallocate(iVnc, 1, mn, 0.0_rkind)
    call sallocate(iBnc, 1, mn, 0.0_rkind)

    call sallocate(ajk, 1, mn, 0.0_rkind)! this must be allocated & assigned now, as it is used in readin: primarily
    ! used in packxi

    do kk = 1, mn
      mk = im(kk)
      nk = in(kk)

      if (mk == 0) ajk(kk) = pi2

    end do

    if (myid == 0) then ! read plasma boundary & computational boundary: initialize interface geometry

      if (Lchangeangle) write(ounit, '("readin : ", 10x ," : CHANGING ANGLE ")')

      do ii = 1, mn
        mm = im(ii)
        nn = in(ii) / Nfp ! set plasma boundary, computational boundary

        if (Lchangeangle) then
          jj = -1
          kk = -nn ! change sign of poloidal angle
        else
          jj = +1
          kk = +nn
        end if

        if (mm == 0 .and. nn == 0) then

          iRbc(ii, Nvol) = Rbc(nn, mm)! plasma        boundary is ALWAYS given by namelist Rbc & Zbs
          iZbs(ii, Nvol) = 0.0_rkind
          if (.not.enforce_stell_sym) then
            iRbs(ii, Nvol) = 0.0_rkind
            iZbc(ii, Nvol) = Zbc(nn, mm)
          else
            iRbs(ii, Nvol) = 0.0_rkind
            iZbc(ii, Nvol) = 0.0_rkind
          end if

          if (Lfreebound == 1) then

            iRbc(ii, Mvol) = Rwc(nn, mm)! computational boundary is ALWAYS given by namelist Rwc & Zws
            iZbs(ii, Mvol) = 0.0_rkind
            if (.not.enforce_stell_sym) then
              iRbs(ii, Mvol) = 0.0_rkind
              iZbc(ii, Mvol) = Zwc(nn, mm)
            else
              iRbs(ii, Mvol) = 0.0_rkind
              iZbc(ii, Mvol) = 0.0_rkind
            end if

            iVns(ii) = 0.0_rkind
            iBns(ii) = 0.0_rkind
            if (.not.enforce_stell_sym) then
              iVnc(ii) = Vnc(nn, mm)! I guess that this must be 0.0_rkind, because \div B = 0
              iBnc(ii) = Bnc(nn, mm)! I guess that this must be 0.0_rkind, because \div B = 0
            else
              iVnc(ii) = 0.0_rkind
              iBnc(ii) = 0.0_rkind
            end if

          end if

        else

          iRbc(ii, Nvol) = Rbc(kk, mm) + Rbc(-kk, -mm)! plasma        boundary is ALWAYS given by namelist Rbc & Zbs
          iZbs(ii, Nvol) = (Zbs(kk, mm) - Zbs(-kk, -mm)) * jj
          if (.not.enforce_stell_sym) then
            iRbs(ii, Nvol) = (Rbs(kk, mm) - Rbs(-kk, -mm)) * jj
            iZbc(ii, Nvol) = Zbc(kk, mm) + Zbc(-kk, -mm)
          else
            iRbs(ii, Nvol) = 0.0_rkind
            iZbc(ii, Nvol) = 0.0_rkind
          end if

          if (Lfreebound == 1) then

            iRbc(ii, Mvol) = Rwc(kk, mm) + Rwc(-kk, -mm)! computational boundary is ALWAYS given by namelist Rwc & Zws
            iZbs(ii, Mvol) = (Zws(kk, mm) - Zws(-kk, -mm)) * jj
            if (.not.enforce_stell_sym) then
              iRbs(ii, Mvol) = (Rws(kk, mm) - Rws(-kk, -mm)) * jj
              iZbc(ii, Mvol) = Zwc(kk, mm) + Zwc(-kk, -mm)
            else
              iRbs(ii, Mvol) = 0.0_rkind
              iZbc(ii, Mvol) = 0.0_rkind
            end if

          end if

          if ( Lfreebound == 1 .or. (.not.Lbdybnzero) ) then

            iVns(ii) = (Vns(kk, mm) - Vns(-kk, -mm)) * jj
            iBns(ii) = (Bns(kk, mm) - Bns(-kk, -mm)) * jj
            if (.not.enforce_stell_sym) then
              iVnc(ii) = Vnc(kk, mm) + Vnc(-kk, -mm)
              iBnc(ii) = Bnc(kk, mm) + Bnc(-kk, -mm)
            else
              iVnc(ii) = 0.0_rkind
              iBnc(ii) = 0.0_rkind
            end if

          end if

        end if

      end do

      if (Lbnszero) then
        iBns(1:mn) = 0.0_rkind
        if (.not.enforce_stell_sym) iBnc(1:mn) = 0.0_rkind
      end if

      select case (Linitialize)

      case (:0) ! Linitialize=0 : initial guess for geometry of the interior surfaces is given in the input file

        if (Lchangeangle) then
          jj = -1 ! change sign of poloidal angle
        else
          jj = +1
        end if

        do idx_mode = 1, num_modes ! will read in Fourier harmonics until the end of file is reached
          mm = mmRZRZ(idx_mode)
          nn = nnRZRZ(idx_mode)

          do ii = 1, mn
            mi = im(ii)
            ni = in(ii)! loop over harmonics within range
            if (mm == 0 .and. mi == 0 .and. nn * Nfp == ni) then
              iRbc(ii, 1:Nvol - 1) = allRZRZ(1, 1:Nvol - 1, idx_mode)! select relevant harmonics
              iZbs(ii, 1:Nvol - 1) = allRZRZ(2, 1:Nvol - 1, idx_mode)! select relevant harmonics
              if (.not.enforce_stell_sym) then
                iRbs(ii, 1:Nvol - 1) = allRZRZ(3, 1:Nvol - 1, idx_mode)! select relevant harmonics
                iZbc(ii, 1:Nvol - 1) = allRZRZ(4, 1:Nvol - 1, idx_mode)! select relevant harmonics
              else
                iRbs(ii, 1:Nvol - 1) = 0.0_rkind ! select relevant harmonics
                iZbc(ii, 1:Nvol - 1) = 0.0_rkind ! select relevant harmonics
              end if
            elseif (mm == mi .and. nn * Nfp == jj * ni) then
              iRbc(ii, 1:Nvol - 1) = allRZRZ(1, 1:Nvol - 1, idx_mode)! select relevant harmonics
              iZbs(ii, 1:Nvol - 1) = jj * allRZRZ(2, 1:Nvol - 1, idx_mode)! select relevant harmonics
              if (.not.enforce_stell_sym) then
                iRbs(ii, 1:Nvol - 1) = jj * allRZRZ(3, 1:Nvol - 1, idx_mode)! select relevant harmonics
                iZbc(ii, 1:Nvol - 1) = allRZRZ(4, 1:Nvol - 1, idx_mode)! select relevant harmonics
              else
                iRbs(ii, 1:Nvol - 1) = 0.0_rkind ! select relevant harmonics
                iZbc(ii, 1:Nvol - 1) = 0.0_rkind ! select relevant harmonics
              end if
            end if
          end do

        end do

      end select ! end select case( Linitialize )

      if (Igeometry == 3) then
        if (Rac(0) > 0.0_rkind) then ! user has supplied logically possible coordinate axis
          iRbc(1:Ntor + 1, 0) = Rac(0:Ntor)
          iZbs(1:Ntor + 1, 0) = Zas(0:Ntor)
          iRbs(1:Ntor + 1, 0) = Ras(0:Ntor)
          iZbc(1:Ntor + 1, 0) = Zac(0:Ntor)
        else ! see initialize_internal_arrays for poloidal-average specification of coordinate axis and geometrical initialization
        end if
      end if

    end if

    call broadcast(iRbc(1:mn, 0:Mvol), (Mvol + 1) * mn, 0)
    if (Igeometry == 3) then
      call broadcast(iZbs(1:mn, 0:Mvol), (Mvol + 1) * mn, 0)! only required for ii > 1
    end if
    if (.not.enforce_stell_sym) then
      call broadcast(iRbs(1:mn, 0:Mvol), (Mvol + 1) * mn, 0)! only required for ii > 1
      if (Igeometry == 3) then
        call broadcast(iZbc(1:mn, 0:Mvol), (Mvol + 1) * mn, 0)
      end if
    end if

    if (Lfreebound == 1) then
      call broadcast(iVns(1:mn), mn, 0)! only required for ii > 1
      call broadcast(iBns(1:mn), mn, 0)! only required for ii > 1
      if (.not.enforce_stell_sym) then
        call broadcast(iVnc(1:mn), mn, 0)
        call broadcast(iBnc(1:mn), mn, 0)
      end if
    end if

    if (Igeometry == 1 .or. Igeometry == 2) then
      iRbc(1:mn, 0) = 0.0_rkind ! innermost volume must be trivial: this is used in volume: innermost interface is
      ! coordinate axis
      if (.not.enforce_stell_sym) then
        iRbs(1:mn, 0) = 0.0_rkind ! innermost volume must be trivial: this is used in volume
      end if
    end if

    if (Igeometry == 3) then
      iZbs(1, 0:Mvol) = 0.0_rkind ! Zbs_{m=0,n=0} is irrelevant
    end if
    if (.not.enforce_stell_sym) then
      iRbs(1, 0:Mvol) = 0.0_rkind ! Rbs_{m=0,n=0} is irrelevant
    end if

    if (Igeometry == 1 .and. Lreflect == 1) then ! reflect upper and lower bound in slab, each take half the amplitude
      iRbc(2:mn, Mvol) = iRbc(2:mn, Mvol) * 1.0_rkind / 2.0_rkind
      iRbc(2:mn, 0) = -iRbc(2:mn, Mvol)
      if (.not.enforce_stell_sym) then
        iRbs(2:mn, Mvol) = iRbs(2:mn, Mvol) * 1.0_rkind / 2.0_rkind
        iRbs(2:mn, 0) = -iRbs(2:mn, Mvol)
      end if
    end if

    Rscale = iRbc(1, Mvol)! this will be used to normalize the geometrical degrees-of-freedom

    if (myid == 0) write(ounit, '("readin : ", 10x ," : myid=",i3," ; Rscale=",es22.15," ")')myid, Rscale

    call random_seed()

    pi2nfp = pi2 / Nfp

    pi2pi2nfp = pi2 * pi2nfp
    pi2pi2nfphalf = pi2 * pi2nfp * 1.0_rkind / 2.0_rkind
    pi2pi2nfpquart = pi2 * pi2nfp * 1.0_rkind / 4.0_rkind

    Mrad = maxval(Lrad(1:Mvol))

    if (myid == 0) write(ounit, '("initialize_internal_arrays : ",10x," : myid=",i3,"  Mrad=",i3," : Lrad=",257(i3,",",:))')myid, &
        Mrad,&
        & Lrad(1:Mvol)

    !                            Rbc  Zbs    Rbs    Zbc
    select case (Igeometry)
    case (1:2)
      if (enforce_stell_sym) LGdof = mn
      if (.not.enforce_stell_sym) LGdof = mn + mn - 1
    case (3)
      if (enforce_stell_sym) LGdof = mn + mn - 1
      if (.not.enforce_stell_sym) LGdof = mn + mn - 1 + mn - 1 + mn
    end select

    NGdof = (Mvol - 1) * LGdof

    if (Winitialize_internal_arrays) then
      cput = MPI_Wtime()
      write(ounit, '("initialize_internal_arrays : ",f10.2," : myid=",i3," ; NGdof=",i9,"")')cput - cpus, myid, NGdof
    end if

    do vvol = 0, Nvol

      if (ql(vvol) == 0 .and. qr(vvol) == 0) then
        iota(vvol) = iota(vvol)
      else
        iota(vvol) = (pl(vvol) + goldenmean * pr(vvol)) / (ql(vvol) + goldenmean * qr(vvol))
      end if

      if (lq(vvol) == 0 .and. rq(vvol) == 0) then
        oita(vvol) = oita(vvol)
      else
        oita(vvol) = (lp(vvol) + goldenmean * rp(vvol)) / (lq(vvol) + goldenmean * rq(vvol))
      end if

      if (Winitialize_internal_arrays .and. myid == 0) write(ounit, 1002)vvol, pl(vvol), ql(vvol), pr(vvol), qr(vvol), iota(vvol),&
          & lp(vvol)&
          & , lq(vvol), rp(vvol), rq(vvol), oita(vvol)

      1002 format("initialize_internal_arrays : ", 10x, " :      ", 3x, " ; transform : ", i3, " : (", i3, " /", i3, " ) * (", i3, &
          " /", i3, " ) = ", f18.15, "  ", "(", i3, " /", i3, " ) * (", i3, " /", i3, " ) = ", f18.15, "  ")
    end do

    call sallocate(dtflux, 1, Mvol, 0.0_rkind)
    call sallocate(dpflux, 1, Mvol, 0.0_rkind)

    ! with Igeom=1, Lcons=3, need to provide total pflux
    if (Lconstraint == 3 .and. Igeometry == 1) then
      total_pflux = pflux(Mvol) * phiedge / pi2
      pflux(Mvol) = 0
    end if

    select case (Igeometry)
    case (1)
      dtflux(1) = tflux(1)
      dpflux(1) = pflux(1)! Cartesian             : this is the "inverse" operation defined in xspech
    case (2:3)
      dtflux(1) = tflux(1)
      dpflux(1) = 0.0_rkind ! cylindrical or toroidal
    end select

    dtflux(2:Mvol) = tflux(2:Mvol) - tflux(1:Mvol - 1)
    dpflux(2:Mvol) = pflux(2:Mvol) - pflux(1:Mvol - 1)

    dtflux(1:Mvol) = dtflux(1:Mvol) * phiedge / pi2
    dpflux(1:Mvol) = dpflux(1:Mvol) * phiedge / pi2

    if (Lconstraint == 3) then

      mu(1) = Ivolume(1) / (tflux(1) * phiedge)

      do vvol = 2, Mvol
        mu(vvol) = (Ivolume(vvol) - Ivolume(vvol - 1)) / ( (tflux(vvol) - tflux(vvol - 1)) * phiedge)
      end do
    end if

    call sallocate(sweight, 1, Mvol, 0.0_rkind)
    !sweight(1:Mvol) = upsilon * tflux(1:Mvol)**wpoloidal ! toroidal flux in vacuum region is not constant
    do vvol = 1, Mvol
      sweight(vvol) = upsilon * (vvol * 1.0_rkind / Nvol) ** wpoloidal
    end do

    call sallocate(TT, 0, Mrad, 0, 1, 0, 1, 0.0_rkind)
    call sallocate(RTT, 0, Lrad(1), 0, Mpol, 0, 1, 0, 1, 0.0_rkind)
    call sallocate(RTM, 0, Lrad(1), 0, Mpol, 0.0_rkind)

    call get_cheby(-1.0_rkind, Mrad, TT(:, 0, :))
    call get_cheby(1.0_rkind, Mrad, TT(:, 1, :))

    call get_zernike(0.0_rkind, Lrad(1), Mpol, RTT(:, :, 0, :))
    call get_zernike(1.0_rkind, Lrad(1), Mpol, RTT(:, :, 1, :))
    call get_zernike_rm(0.0_rkind, Lrad(1), Mpol, RTM(:, :))

    call sallocate(ImagneticOK, 1, Mvol, .false.)
    call sallocate(beltrami_gmres_status, 1, Mvol, 0)
    call sallocate(beltrami_gmres_iterations, 1, Mvol, 0)
    call sallocate(beltrami_gmres_residual, 1, Mvol, 0.0_rkind)
    call sallocate(force_jvp_status, 1, Mvol, 0)

    force_ft_trunc_grad_alloc = .false.

    call sallocate(ki, 1, mn, 0, 1, 0)
    call sallocate(kija, 1, mn, 1, mn, 0, 1, 0)
    call sallocate(kijs, 1, mn, 1, mn, 0, 1, 0)

    do ii = 1, mn
      mi = im(ii)
      ni = in(ii)

      call getimn(lMpol, lNtor, Nfp, mi, ni, kk)
      if (kk > 0) then
        if (mi == 0 .and. ni == 0) then
          ki(ii, 0:1) = [kk, 1]
        else
          ki(ii, 0:1) = [kk, 2]
        end if
      end if

      do jj = 1, mn
        mj = im(jj)
        nj = in(jj)
        mimj = mi + mj
        ninj = ni + nj !   adding

        call getimn(lMpol, lNtor, Nfp, mimj, ninj, kk)
        if (kk > 0) then
          if (mimj == 0 .and. ninj == 0) then
            kija(ii, jj, 0:1) = [kk, 1]
          else
            kija(ii, jj, 0:1) = [kk, 2]
          end if
        end if
        mimj = mi - mj
        ninj = ni - nj ! subtracting

        if (mimj > 0 .or. (mimj == 0 .and. ninj >= 0) ) then
          call getimn(lMpol, lNtor, Nfp, mimj, ninj, kk)
          if (kk > 0) then
            if (mimj == 0 .and. ninj == 0) then
              kijs(ii, jj, 0:1) = [kk, 1]
            else
              kijs(ii, jj, 0:1) = [kk, 2]
            end if
          end if
        else
          call getimn(lMpol, lNtor, Nfp, -mimj, -ninj, kk)
          if (kk > 0) then
            kijs(ii, jj, 0:1) = [kk, -2] ! only the sine modes need the sign factor
          end if
        end if

      end do

    end do

    if (Igeometry == 2) then ! standard cylindrical

      call sallocate(djkp, 1, mn, 1, mn, 0)! only used in volume: trigonometric identities
      call sallocate(djkm, 1, mn, 1, mn, 0)! only used in volume: trigonometric identities

      do ii = 1, mn
        mi = im(ii)
        ni = in(ii)
        do jj = 1, mn
          mj = im(jj)
          nj = in(jj)
          if (mi - mj == 0 .and. ni - nj == 0) djkp(ii, jj) = 1
          if (mi + mj == 0 .and. ni + nj == 0) djkm(ii, jj) = 1
        end do
      end do

    end if

    call sallocate(iotakkii, 1, mn, 0)! used to identify matrix elements in straight-field-line angle transformation

    call sallocate(iotaksub, 1, mn, 1, mns, 0)
    call sallocate(iotaksgn, 1, mn, 1, mns, 0)
    call sallocate(iotakadd, 1, mn, 1, mns, 0)

    do kk = 1, mn
      mk = im(kk)
      nk = in(kk)

      call getimn(sMpol, sNtor, Nfp, mk, nk, ii)
      if (ii > 0) iotakkii(kk) = ii

      do jj = 1, mns
        mj = ims(jj)
        nj = ins(jj)

        mkmj = mk - mj
        nknj = nk - nj

        if (mkmj > 0 .or. (mkmj == 0 .and. nknj >= 0) ) then

          call getimn(sMpol, sNtor, Nfp, mkmj, nknj, ii)
          if (ii > 0) then
            iotaksub(kk, jj) = ii
            iotaksgn(kk, jj) = 1
          end if

        else

          call getimn(sMpol, sNtor, Nfp, -mkmj, -nknj, ii)
          if (ii > 0) then
            iotaksub(kk, jj) = ii
            iotaksgn(kk, jj) = -1
          end if

        end if

        mkmj = mk + mj
        nknj = nk + nj

        call getimn(sMpol, sNtor, Nfp, mkmj, nknj, ii)
        if (ii > 0) then
          iotakadd(kk, jj) = ii
        end if

      end do

    end do

    ! Allocate space for the toroidal current array in each interface

    call sallocate(IPDt, 1, Mvol, 0.0_rkind)
    if (Lfreebound == 1) then
      call sallocate(IPDtDpf, 1, Mvol, 1, Mvol, 0.0_rkind)
    else
      if (Igeometry == 1) then
        ! add an additional constraint to make the total pflux = 0
        call sallocate(IPDtDpf, 1, Mvol, 1, Mvol, 0.0_rkind)
      else
        call sallocate(IPDtDpf, 1, Mvol - 1, 1, Mvol - 1, 0.0_rkind)
      end if
    end if

    call sallocate(cheby, 0, Mrad, 0, 2, 0.0_rkind)
    call sallocate(zernike, 0, Lrad(1), 0, Mpol, 0, 2, 0.0_rkind)

    call sallocate(Iquad, 1, Mvol, 0)

    do vvol = 1, Mvol

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

      if (Nquad > 0) then
        Iquad(vvol) = Nquad
      else
        if (Lcoordinatesingularity) Iquad(vvol) = Mpol + 2 * Lrad(vvol) - Nquad ! NEED TO REVISE REGULARIZATION FACTORS
        if (.not.Lcoordinatesingularity) Iquad(vvol) = 2 * Lrad(vvol) - Nquad
      end if

    end do

    maxIquad = maxval(Iquad(1:Mvol))

    call sallocate(gaussianweight, 1, maxIquad, 1, Mvol, 0.0_rkind)! perhaps it would be neater to make this a structure
    call sallocate(gaussianabscissae, 1, maxIquad, 1, Mvol, 0.0_rkind)

    do vvol = 1, Mvol

      lquad = Iquad(vvol)

      call gauleg(lquad, gaussianweight(1:lquad, vvol), gaussianabscissae(1:lquad, vvol), igauleg)! JAB

      if (myid == 0) then
        cput = MPI_Wtime()
        select case (igauleg) !                                                  123456789012345
        case (0)
          if (Winitialize_internal_arrays) write(ounit, 1000)cput - cpus, vvol, igauleg, "success", gaussianabscissae(1:lquad, &
              vvol)
        case (1)
          write(ounit, 1000)cput - cpus, vvol, igauleg, "failed", gaussianabscissae(1:lquad, vvol)
        case (2)
          write(ounit, 1000)cput - cpus, vvol, igauleg, "input error", gaussianabscissae(1:lquad, vvol)
        case default
          write(ounit, 1000)cput - cpus, vvol, igauleg, "weird", gaussianabscissae(1:lquad, vvol)
          call fatal(.true., __FILE__, "weird ifail returned by gauleg ")

        end select
        if (Winitialize_internal_arrays) write(ounit, 1001)gaussianweight(1:lquad, vvol)
      end if

      1000 format("initialize_internal_arrays : ", f10.2, " : lvol=", i3, " ; igauleg=", i5, " ; ", a15, "  abscissae =", 99f09.05)
      1001 format("initialize_internal_arrays : ", 10x, " :      ", 3x, "           ", 5x, "   ", 15x, "  weights   =", 99f09.05)

    end do

    if (Lconstraint == 2) then
      call fatal(Lfreebound == 1, __FILE__, "The combination of helicity constraint and free boundary is under        &
          & construction ")

      if (Igeometry == 3 .and. myid == 0) then
        write(ounit, *)'WARNING: The Hessian matrix needs further review for Igeometry = 3'
        write(ounit, *)'         However, it can still serve the purpose of L_findzero = 2'
      end if
    end if

    call nallocate(BBweight, 1, mn)
    BBweight = opsilon * exp(-escale * (im(1:mn) ** 2 + (in(1:mn) / Nfp) ** 2) )

    if (myid == 0 .and. escale > small) then
      do ii = 1, mn
        write(ounit, '("initialize_internal_arrays : ", 10x, " : myid=", i3, " ; (", i3, ",", i3, ") : BBweight=", es13.5, " ")') myid, im(ii), in(ii&
            ) / Nfp,&
            & BBweight(ii)
      end do
    end if

    call sallocate(mmpp, 1, mn, 0.0_rkind)

    do ii = 1, mn
      mi = im(ii)

      if (mi == 0) then
        mmpp(ii) = 0.0_rkind
      else
        mmpp(ii) = mi ** pcondense
      end if

    end do

    call sallocate(NAdof, 1, Mvol, 0)! Beltrami degrees-of-freedom in each annulus

    ! Ate/Aze/Ato/Azo derivative slots: 0 holds the solution, 1:2 the flux derivatives, and the
    ! negative slots hold perturbed solutions (thread tid writes slot -(tid+1) on the parallel
    ! Jacobian DOF loop; -1 doubles as the serial perturbed slot). Lower bound at least -2.
    n_aslot_lo = min(-2, -omp_get_max_threads())
    call nallocate(Ate, 1, Mvol, n_aslot_lo, 2, 1, mn)! recall that this is type:sub-grid
    call nallocate(Aze, 1, Mvol, n_aslot_lo, 2, 1, mn)
    call nallocate(Ato, 1, Mvol, n_aslot_lo, 2, 1, mn)
    call nallocate(Azo, 1, Mvol, n_aslot_lo, 2, 1, mn)

    call sallocate(Lma, 1, Mvol, 1, mn, 0)! degree of freedom index: for Lagrange multiplier
    call sallocate(Lmb, 1, Mvol, 1, mn, 0)
    call sallocate(Lmc, 1, Mvol, 1, mn, 0)! only need Lmc(2:mn) : only for .not.enforce_stell_sym
    call sallocate(Lmd, 1, Mvol, 1, mn, 0)! only need Lmd(2:mn) : only for .not.enforce_stell_sym
    call sallocate(Lme, 1, Mvol, 1, mn, 0)! only need Lme(2:mn)
    call sallocate(Lmf, 1, Mvol, 1, mn, 0)! only need Lmf(2:mn) : only for .not.enforce_stell_sym
    call sallocate(Lmg, 1, Mvol, 1, mn, 0)! only need Lmg(1   )
    call sallocate(Lmh, 1, Mvol, 1, mn, 0)! only need Lmh(1   )

    call sallocate(Lmavalue, 1, Mvol, 1, mn, 0.0_rkind)
    call sallocate(Lmbvalue, 1, Mvol, 1, mn, 0.0_rkind)
    call sallocate(Lmcvalue, 1, Mvol, 1, mn, 0.0_rkind)
    call sallocate(Lmdvalue, 1, Mvol, 1, mn, 0.0_rkind)
    call sallocate(Lmevalue, 1, Mvol, 1, mn, 0.0_rkind)
    call sallocate(Lmfvalue, 1, Mvol, 1, mn, 0.0_rkind)
    call sallocate(Lmgvalue, 1, Mvol, 1, mn, 0.0_rkind)
    call sallocate(Lmhvalue, 1, Mvol, 1, mn, 0.0_rkind)

    do vvol = 1, Mvol

      call lregion(vvol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)

      if (Lcoordinatesingularity) then
        zerdof = 0 ! count Zernike degree of freedom
        do ii = 2, Mpol ! for m>1
          do jj = ii, Lrad(vvol), 2
            zerdof = zerdof + 2 * ntor + 1 ! plus and minus sign for n>1, unique for n==0
            if (.not.enforce_stell_sym) zerdof = zerdof + 2 * ntor + 1 ! plus and minus sign for n
          end do
        end do
        zerdof = zerdof * 2 ! we have one for At and 1.0_rkind for Az

        do jj = 0, Lrad(vvol), 2 ! for m==0
          zerdof = zerdof + ntor + 1 ! minus sign for n, Aze
          if (jj >= 2) zerdof = zerdof + ntor + 1 ! minus sign for n, Ate, without l=0 due to recombination

          if (.not.enforce_stell_sym) then
            zerdof = zerdof + ntor ! sin component minus sign for n, Azo
            if (jj >= 2) zerdof = zerdof + ntor ! minus sign for n, Ato, without l=0 due to recombination
          end if
        end do

        if (Mpol >= 1) then ! for m==1
          do jj = 1, Lrad(vvol), 2
            zerdof = zerdof + 2 * ntor + 1 ! minus and plus sign for n, Aze
            if (jj >= 2) zerdof = zerdof + 2 * ntor + 1 ! minus sign for n, Ate, without l=0 due to recombination

            if (.not.enforce_stell_sym) then
              zerdof = zerdof + 2 * ntor + 1 ! sin component minus and plus sign for n, Azo
              if (jj >= 2) zerdof = zerdof + 2 * ntor + 1 ! minus and plus sign for n, Ato, without l=0 due to
              ! recombination
            end if
          end do
        end if

        ! the degree of freedom in the Beltrami field without Lagrange multipliers
        !                                     a    c      b        d      e      f      g   h
        if (enforce_stell_sym) NAdof(vvol) = zerdof + mn + Ntor + 1 + mn - 1 + 1 + 0
        if (.not.enforce_stell_sym) NAdof(vvol) = zerdof + mn + mn - 1 + Ntor + 1 + Ntor + mn - 1 + mn - 1 + 1 + 0 !
        !this is broken at the moment

        ! due to basis recombination, Lma will not have the m=0 and m=1 harmonics. We substract them now
        ! m = 0
        NAdof(vvol) = NAdof(vvol) - (ntor + 1)
        if (.not.enforce_stell_sym) NAdof(vvol) = NAdof(vvol) - ntor

        ! m = 1
        if (Mpol >= 1) then
          NAdof(vvol) = NAdof(vvol) - (2 * ntor + 1)
          if (.not.enforce_stell_sym) NAdof(vvol) = NAdof(vvol) - (2 * ntor + 1)
        end if
      else ! .not.Lcoordinatesingularity
        if (enforce_stell_sym) NAdof(vvol) = 2 * (mn) * (Lrad(vvol)) + mn - 1 + 1 + 1
        if (.not.enforce_stell_sym) NAdof(vvol) = 2 * (mn + mn - 1) * (Lrad(vvol)) + mn - 1 + mn - 1 + 1 + 1
      end if

      do ii = 1, mn ! loop over Fourier harmonics

        do ideriv = n_aslot_lo, 2 ! loop over derivatives (per-thread perturbed slots + 0:2)

          call sallocate(Ate(vvol, ideriv, ii)%s, 0, Lrad(vvol), 0.0_rkind)
          call sallocate(Aze(vvol, ideriv, ii)%s, 0, Lrad(vvol), 0.0_rkind)
          call sallocate(Ato(vvol, ideriv, ii)%s, 0, Lrad(vvol), 0.0_rkind)
          call sallocate(Azo(vvol, ideriv, ii)%s, 0, Lrad(vvol), 0.0_rkind)

        end do

        ideriv = 0

        call sallocate(Ate(vvol, ideriv, ii)%i, 0, Lrad(vvol), 0)! degree of freedom index
        call sallocate(Aze(vvol, ideriv, ii)%i, 0, Lrad(vvol), 0)
        call sallocate(Ato(vvol, ideriv, ii)%i, 0, Lrad(vvol), 0)
        call sallocate(Azo(vvol, ideriv, ii)%i, 0, Lrad(vvol), 0)

      end do

      select case (Linitgues) ! for iterative solver of the Beltrami fields, an initial guess is required
      case (0)
      case (1)
        Ate(vvol, 0, 1)%s(0:1) = dtflux(vvol) * 1.0_rkind / 2.0_rkind ! this is an integrable approximation
        Aze(vvol, 0, 1)%s(0:1) = dpflux(vvol) * 1.0_rkind / 2.0_rkind ! this is an integrable approximation
        if (Lcoordinatesingularity) then
          Ate(vvol, 0, 1)%s(2) = dtflux(vvol) * 1.0_rkind / 2.0_rkind * 1.0_rkind / 2.0_rkind
        end if
      case (2)
        ! will call write_vector_potentials below to read initial vector potential from file
      case (3)
        ! the initial guess will be randomized, maximum is maxrndgues
        do ii = 1, mn ! loop over Fourier harmonics

          do ideriv = n_aslot_lo, 2 ! loop over derivatives (per-thread perturbed slots + 0:2)

            call random_number(Ate(vvol, ideriv, ii)%s)
            call random_number(Aze(vvol, ideriv, ii)%s)
            Ate(vvol, ideriv, ii)%s = Ate(vvol, ideriv, ii)%s * maxrndgues
            Aze(vvol, ideriv, ii)%s = Aze(vvol, ideriv, ii)%s * maxrndgues
            if (.not.enforce_stell_sym) then
              call random_number(Ato(vvol, ideriv, ii)%s)
              call random_number(Azo(vvol, ideriv, ii)%s)
              Ato(vvol, ideriv, ii)%s = Ato(vvol, ideriv, ii)%s * maxrndgues
              Azo(vvol, ideriv, ii)%s = Azo(vvol, ideriv, ii)%s * maxrndgues
            end if

          end do

        end do

      end select

      idof = 0 ! degree of freedom index: reset to 0 in each volume

      if (Lcoordinatesingularity) then

        do ii = 1, mn
          mi = im(ii)
          ni = in(ii)

          do ll = 0, Lrad(vvol)
            ! Zernike is non 0.0_rkind only if ll>=mi and when they have the same parity
            if (ll >= mi .and. mod(mi + ll, 2) == 0) then
              ! We use the basis combination for m=0 and 1. They don't have ll=0 component.
              if (.not. ( (ll == 0 .and. mi == 0) .or. (ll == 1 .and. mi == 1) ) ) then
                idof = idof + 1
                Ate(vvol, 0, ii)%i(ll) = idof ! Zernike
              end if
              idof = idof + 1
              Aze(vvol, 0, ii)%i(ll) = idof
              if (.not.enforce_stell_sym .and. ii > 1) then
                if (.not. ( (ll == 0 .and. mi == 0) .or. (ll == 1 .and. mi == 1) ) ) then
                  idof = idof + 1
                  Ato(vvol, 0, ii)%i(ll) = idof ! Zernike
                end if
                idof = idof + 1
                Azo(vvol, 0, ii)%i(ll) = idof
              end if ! .not.enforce_stell_sym
            end if ! Zernike
          end do

        end do

        do ii = 1, mn
          mi = im(ii)
          ni = in(ii)
          ! Lma is for Ate boundary condition on axis. For m=0 and 1, the boundary condition has been satisfied by basis
          ! recombination, so they are excluded.
          if (mi /= 0 .and. mi /= 1) then
            idof = idof + 1
            Lma(vvol, ii) = idof
          end if
          ! Lmb is for Aze boundary condition on axis. We only have that for m=0.
          if (mi == 0) then
            idof = idof + 1
            Lmb(vvol, ii) = idof
          end if
          ! Lme is for B.n at the outer boundary cos component. We don't have it for m=n=0.
          if (ii > 1) then
            idof = idof + 1
            Lme(vvol, ii) = idof
          end if
          ! Lmg is for dtflux. We only have it for m=n=0.
          if (ii == 1) then
            idof = idof + 1
            Lmg(vvol, ii) = idof
          end if
          if (.not.enforce_stell_sym) then
            ! Lmc is for Ato boundary condition on axis. Same as Lma.
            if (mi /= 0 .and. mi /= 1) then
              idof = idof + 1
              Lmc(vvol, ii) = idof
            end if
            ! Lmf is for B.n at the outer boundary sin component. Same as Lme.
            if (ii > 1) then
              idof = idof + 1
              Lmf(vvol, ii) = idof
            end if
            ! Lmd is for Azo on axis. We only have it for m=0, but not m=n=0.
            if (ii > 1 .and. mi == 0) then
              idof = idof + 1
              Lmd(vvol, ii) = idof
            end if
          end if

        end do

        call fatal(idof /= NAdof(vvol), __FILE__, "need to count Beltrami degrees-of-freedom more carefully  for         &
            &  coordinate singularity")

        call fatal( (idof + 1) ** 2 >= huge(idof), __FILE__, "NAdof too big, should be smaller than maximum of int32 type")

      else ! .not.Lcoordinatesingularity

        do ii = 1, mn
          ! We use basis recombination method to ensure the inner boundary has At=Az=0. Therefore they don't have ll=0
          ! component.
          do ll = 1, Lrad(vvol)
            idof = idof + 1
            Ate(vvol, 0, ii)%i(ll) = idof
            idof = idof + 1
            Aze(vvol, 0, ii)%i(ll) = idof
            if (ii > 1 .and. .not.enforce_stell_sym) then
              idof = idof + 1
              Ato(vvol, 0, ii)%i(ll) = idof
              idof = idof + 1
              Azo(vvol, 0, ii)%i(ll) = idof
            end if
          end do
        end do

        do ii = 1, mn
          ! Lme is for B.n at the outer boundary cos component. We don't have it for m=n=0.
          if (ii > 1) then
            idof = idof + 1
            Lme(vvol, ii) = idof
          end if
          ! Lmf is for B.n at the outer boundary sin component. Same as Lme
          if (ii > 1 .and. .not.enforce_stell_sym) then
            idof = idof + 1
            Lmf(vvol, ii) = idof
          end if
          ! Lmg and Lmh are the dtflux and dpflux constraint. Only present for m=n=0
          if (ii == 1) then
            idof = idof + 1
            Lmg(vvol, ii) = idof
            idof = idof + 1
            Lmh(vvol, ii) = idof
          end if
        end do

        call fatal(idof /= NAdof(vvol), __FILE__, "need to count degrees-of-freedom more carefully for new matrix")

        call fatal( (idof + 1) ** 2 >= huge(idof), __FILE__, "NAdof too big, should be smaller than maximum of int32 type")

      end if

      call fatal(idof /= NAdof(vvol), __FILE__, "impossible logic")

      do ii = 1, mn
        do jj = 0, Lrad(vvol)
          if (Ate(vvol, 0, ii)%i(jj) == 0) Ate(vvol, 0, ii)%s(jj) = 0.0_rkind
          if (Aze(vvol, 0, ii)%i(jj) == 0) Aze(vvol, 0, ii)%s(jj) = 0.0_rkind
          if (.not.enforce_stell_sym) then
            if (Ato(vvol, 0, ii)%i(jj) == 0) Azo(vvol, 0, ii)%s(jj) = 0.0_rkind
            if (Azo(vvol, 0, ii)%i(jj) == 0) Azo(vvol, 0, ii)%s(jj) = 0.0_rkind
          end if
        end do !jj
      end do !ii

    end do

    if (Linitgues == 2) then
      call write_vector_potentials('R')! read initial guess for Beltrami field from file
    end if

    if (myid == 0) then
      cput = MPI_Wtime()
      write(ounit, '("initialize_internal_arrays : ", 10x ," : ")')
      write(ounit, '("initialize_internal_arrays : ",f10.2," : Nquad=", i4, " ; mn=", i5, " ; NGdof=", i6, "  NAdof=", 16(i6), ")")')cput - &
          cpus,&
          & Nquad, mn, NGdof, NAdof(1:min(Mvol, 16))
    end if

    ! Fourier transforms

    Nt = max(Ndiscrete * 4 * Mpol, 1)
    Nz = max(Ndiscrete * 4 * Ntor, 1)
    Ntz = Nt * Nz
    soNtz = 1.0_rkind / sqrt(1.0_rkind * Ntz)! exaggerated discrete resolution

    hNt = Nt / 2
    if (Nz > 1) then
      hNz = Nz / 2
    else
      hNz = 0
    end if

    if (myid == 0) then
      cput = MPI_Wtime()
      write(ounit, '("initialize_internal_arrays : ", 10x ," : ")')
      write(ounit, '("initialize_internal_arrays : ",f10.2," : Nt=", i6, " ; Nz=", i6, " ; Ntz=", i9, ")")')cput - cpus, Nt, Nz, Ntz
    end if

    call sallocate(iRij, 1, Ntz, 0, Mvol, 0.0_rkind)! interface geometry in real space
    call sallocate(iZij, 1, Ntz, 0, Mvol, 0.0_rkind)!
    call sallocate(dRij, 1, Ntz, 1, Mvol, 0.0_rkind)! interface geometry in real space: poloidal derivative
    call sallocate(dZij, 1, Ntz, 1, Mvol, 0.0_rkind)
    call sallocate(tRij, 1, Ntz, 0, Mvol, 0.0_rkind)! interface geometry in real space: poloidal derivative
    call sallocate(tZij, 1, Ntz, 0, Mvol, 0.0_rkind)

    call sallocate(Rij, 1, Ntz, 0, 3, 0, 3, 0.0_rkind)! these are used for inverse fft to reconstruct real space
    ! geometry from interpolated Fourier harmonics
    call sallocate(Zij, 1, Ntz, 0, 3, 0, 3, 0.0_rkind)
    call sallocate(sg, 1, Ntz, 0, 3, 0.0_rkind)
    call sallocate(inv_jac, 1, Ntz, 0.0_rkind)
    call sallocate(guvij, 1, Ntz, 0, 3, 0, 3, -1, 3, 0.0_rkind)! need this on higher resolution grid for accurate
    ! Fourier decomposition
    call sallocate(gvuij, 1, Ntz, 0, 3, 0, 3, 0.0_rkind)! need this on higher resolution grid for accurate Fourier
    ! decomposition

    call sallocate(goomne, 0, mne, 1, maxIquad, 0.0_rkind)! workspace for Fourier decomposition of metric terms
    call sallocate(goomno, 0, mne, 1, maxIquad, 0.0_rkind)
    call sallocate(gssmne, 0, mne, 1, maxIquad, 0.0_rkind)! workspace for Fourier decomposition of metric terms
    call sallocate(gssmno, 0, mne, 1, maxIquad, 0.0_rkind)
    call sallocate(gstmne, 0, mne, 1, maxIquad, 0.0_rkind)! workspace for Fourier decomposition of metric terms
    call sallocate(gstmno, 0, mne, 1, maxIquad, 0.0_rkind)
    call sallocate(gszmne, 0, mne, 1, maxIquad, 0.0_rkind)! workspace for Fourier decomposition of metric terms
    call sallocate(gszmno, 0, mne, 1, maxIquad, 0.0_rkind)
    call sallocate(gttmne, 0, mne, 1, maxIquad, 0.0_rkind)! workspace for Fourier decomposition of metric terms
    call sallocate(gttmno, 0, mne, 1, maxIquad, 0.0_rkind)
    call sallocate(gtzmne, 0, mne, 1, maxIquad, 0.0_rkind)! workspace for Fourier decomposition of metric terms
    call sallocate(gtzmno, 0, mne, 1, maxIquad, 0.0_rkind)
    call sallocate(gzzmne, 0, mne, 1, maxIquad, 0.0_rkind)! workspace for Fourier decomposition of metric terms
    call sallocate(gzzmno, 0, mne, 1, maxIquad, 0.0_rkind)

    if (Igeometry == 3) then
      call sallocate(dRadR, 1, mn, 0, 1, 0, 1, 1, mn, 0.0_rkind)
      call sallocate(dRadZ, 1, mn, 0, 1, 0, 1, 1, mn, 0.0_rkind)
      call sallocate(dZadR, 1, mn, 0, 1, 0, 1, 1, mn, 0.0_rkind)
      call sallocate(dZadZ, 1, mn, 0, 1, 0, 1, 1, mn, 0.0_rkind)
    end if

    call set_fftw_plan(Nt, Nz)
    call alloc_ntz_arrays(Nt, Nz)

    call sallocate(efmn, 1, mne, 0.0_rkind)! Fourier harmonics workspace
    call sallocate(ofmn, 1, mne, 0.0_rkind)
    call sallocate(cfmn, 1, mne, 0.0_rkind)
    call sallocate(sfmn, 1, mne, 0.0_rkind)
    call sallocate(evmn, 1, mne, 0.0_rkind)
    call sallocate(odmn, 1, mne, 0.0_rkind)
    call sallocate(comn, 1, mne, 0.0_rkind)
    call sallocate(simn, 1, mne, 0.0_rkind)

    call sallocate(gteta, 1, Ntz, 0.0_rkind)
    call sallocate(gzeta, 1, Ntz, 0.0_rkind)

    call sallocate(cosi, 1, Ntz, 1, mn, 0.0_rkind)
    call sallocate(sini, 1, Ntz, 1, mn, 0.0_rkind)

    call fatal(Nz == 0, __FILE__, "illegal division ")

    call fatal(Nt == 0, __FILE__, "illegal division ")

    do ii = 1, mn
      mi = im(ii)
      ni = in(ii)! loop over Fourier harmonics

      do kk = 0, Nz - 1
        zeta = kk * pi2nfp / Nz
        do jj = 0, Nt - 1
          teta = jj * pi2 / Nt
          jk = 1 + jj + kk * Nt
          arg = mi * teta - ni * zeta
          gteta(jk) = teta
          gzeta(jk) = zeta
          cosi(jk, ii) = cos(arg)
          sini(jk, ii) = sin(arg)
        end do
      end do

    end do

    if (Igeometry == 3 .and. iRbc(1, 0) < small) then ! have not yet assigned coordinate axis: see global:readin for
      ! user-supplied Rac, Zas, etc.

      select case (Linitialize)
      case (:-1)
        vvol = Nvol + Linitialize
      case (0)
        vvol = 1 ! this is really a dummy: no interpolation of interface geometry is required: packxi calls rzaxis with
        ! lvol=1
      case (1)
        vvol = Nvol
      case (2)
        vvol = Mvol
      end select

      call rzaxis(Mvol, mn, Nt, Nz, iRbc(1:mn, 0:Mvol), iZbs(1:mn, 0:Mvol), iRbs(1:mn, 0:Mvol), iZbc(1:mn, 0:Mvol), vvol,&
          & .false.)! set coordinate axis

    end if

    call sallocate(psifactor, 1, mn, 1, Mvol, 0.0_rkind)
    call sallocate(inifactor, 1, mn, 1, Mvol, 0.0_rkind)

    psifactor(1:mn, 1:Mvol) = 1.0_rkind
    inifactor(1:mn, 1:Mvol) = 1.0_rkind

    select case (Igeometry)

    case (1)

      psifactor(1:mn, 1:Nvol) = 1.0_rkind

    case (2)

      do vvol = 1, Nvol
        do ii = 1, mn
          if (im(ii) == 0) then
            psifactor(ii, vvol) = tflux(vvol) ** (+1.0_rkind / 2.0_rkind)
          else
            psifactor(ii, vvol) = tflux(vvol) ** (halfmm(ii) - 1.0_rkind / 2.0_rkind)
          end if
        end do
      end do

    case (3)

      do vvol = 1, Nvol
        do ii = 1, mn
          if (im(ii) == 0) then
            psifactor(ii, vvol) = Rscale * tflux(vvol) ** 0.0_rkind
            inifactor(ii, vvol) = Rscale * tflux(vvol) ** 1.0_rkind / 2.0_rkind
          else
            psifactor(ii, vvol) = Rscale * tflux(vvol) ** halfmm(ii)
            inifactor(ii, vvol) = Rscale * tflux(vvol) ** halfmm(ii)
          end if
        end do
      end do

    case default

      call fatal(.true., __FILE__, "invalid Igeometry for construction of psifactor ")

    end select

    if (Linitialize /= 0) then ! interpolate / extrapolate interior interface geometry

      select case (Igeometry)

      case (1) ! Cartesian

        !FATAL( initialize_internal_arrays, Linitialize.ne.1, geometrical initialization under construction for Cartesian )

        do vvol = 1, Nvol
          iRbc(1:mn, vvol) = iRbc(1:mn, Mvol) * tflux(vvol) / tflux(Mvol)
          if (.not.enforce_stell_sym) then
            iRbs(2:mn, vvol) = iRbs(2:mn, Mvol) * tflux(vvol) / tflux(Mvol)
          end if
        end do

      case (2) ! cylindrical - standard

        call fatal(Linitialize /= 1, __FILE__, "geometrical initialization under construction for cylindrical ")

        do vvol = 1, Nvol - 1
          iRbc(1:mn, vvol) = iRbc(1:mn, Nvol) * psifactor(1:mn, vvol)
          if (.not.enforce_stell_sym) then
            iRbs(2:mn, vvol) = iRbs(2:mn, Nvol) * psifactor(2:mn, vvol)
          end if
        end do

      case (3) ! toroidal

        call fatal(Linitialize < 0, __FILE__, "geometrical initialization under construction for toroidal ")
        ! see commented-out source below

        lvol = Nvol - 1 + Linitialize

        call fatal(lvol > Mvol, __FILE__, "perhaps illegal combination of Linitialize and Lfreebound ")

        do vvol = 1, lvol - 1
          iRbc(1:mn, vvol) = iRbc(1:mn, 0) + (iRbc(1:mn, lvol) - iRbc(1:mn, 0)) * (inifactor(1:mn, vvol) / Rscale) /&
              & tflux(lvol) ** halfmm(1:mn)
          iZbs(2:mn, vvol) = iZbs(2:mn, 0) + (iZbs(2:mn, lvol) - iZbs(2:mn, 0)) * (inifactor(2:mn, vvol) / Rscale) /&
              & tflux(lvol) ** halfmm(2:mn)
          if (.not.enforce_stell_sym) then
            iRbs(2:mn, vvol) = iRbs(2:mn, 0) + (iRbs(2:mn, lvol) - iRbs(2:mn, 0)) * (inifactor(2:mn, vvol) / Rscale) /&
                & tflux(lvol) ** halfmm(2:mn)
            iZbc(1:mn, vvol) = iZbc(1:mn, 0) + (iZbc(1:mn, lvol) - iZbc(1:mn, 0)) * (inifactor(1:mn, vvol) / Rscale) /&
                & tflux(lvol) ** halfmm(1:mn)
          end if
        end do

      end select

    end if

    call sallocate(diotadxup, 0, 1, -1, 2, 1, Mvol, 0.0_rkind)! measured rotational transform on inner/outer interfaces
    ! in each annulus
    call sallocate(dItGpdxtp, 0, 1, -1, 2, 1, Mvol, 0.0_rkind)! measured plasma and linking currents


    ! Construction of ``force''

    call sallocate(Bemn, 1, mn, 1, Mvol, 0, 1, 0.0_rkind)
    call sallocate(Bomn, 1, mn, 1, Mvol, 0, 1, 0.0_rkind)
    call sallocate(Iomn, 1, mn, 1, Mvol, 0.0_rkind)
    call sallocate(Iemn, 1, mn, 1, Mvol, 0.0_rkind)
    call sallocate(Somn, 1, mn, 1, Mvol, 0, 1, 0.0_rkind)
    call sallocate(Semn, 1, mn, 1, Mvol, 0, 1, 0.0_rkind)
    call sallocate(Pomn, 1, mn, 1, Mvol, 0, 2, 0.0_rkind)
    call sallocate(Pemn, 1, mn, 1, Mvol, 0, 2, 0.0_rkind)

    call sallocate(BBe, 1, Mvol - 1, 0.0_rkind)
    call sallocate(IIo, 1, Mvol - 1, 0.0_rkind)
    call sallocate(BBo, 1, Mvol - 1, 0.0_rkind)
    call sallocate(IIe, 1, Mvol - 1, 0.0_rkind)

    call sallocate(Btemn, 1, mn, 0, 1, 1, Mvol, 0.0_rkind)! these are declared in global, calculated in sc00aa,
    ! broadcast in xspech, and written to file in hdfint
    call sallocate(Bzemn, 1, mn, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(Btomn, 1, mn, 0, 1, 1, Mvol, 0.0_rkind)
    call sallocate(Bzomn, 1, mn, 0, 1, 1, Mvol, 0.0_rkind)

    call sallocate(Bloweremn, 1, mn, 1, 3, 0.0_rkind)! these are declared in global, calculated in getbco, used in mtrxhs
    call sallocate(Bloweromn, 1, mn, 1, 3, 0.0_rkind)

    call sallocate(vvolume, 1, Mvol, 0.0_rkind)! volume integral of \sqrt g
    call sallocate(lBBintegral, 1, Mvol, 0.0_rkind)! volume integral of B.B
    call sallocate(lABintegral, 1, Mvol, 0.0_rkind)! volume integral of A.B

    if (enforce_stell_sym) lmns = 1 + (mns - 1) ! number of independent degrees-of-freedom in angle transformation
    if (.not.enforce_stell_sym) lmns = 1 + (mns - 1) + (mns - 1) ! number of independent degrees-of-freedom in angle
    ! transformation

    call sallocate(dlambdaout, 1, lmns, 1, Mvol, 0, 1, 0.0_rkind)

    if (Lfreebound > 0) then ! Only do for free-boundary

      if (Lvcgrid == 1) then
        if ( vcNt < Nt ) then
          call fatal(.true., __FILE__, "The plasma boundary resolution for virtual casing vcNt must be >= Nt")
        end if
        if ( vcNz < Nz ) then
          call fatal(.true., __FILE__, "The plasma boundary resolution for virtual casing vcNz must be >= Nz")
        end if

        ! Factor of nfp because plasma boundary for virtual casing has to cover all field periods, but resolution is given for one
        ! field period.
        call sallocate(Jxyz, 1, vcNt * vcNz * Nfp, 1, 3, 0.0_rkind)! Cartesian components of virtual casing surface current
        call sallocate(Pbxyz, 1, vcNt * vcNz * Nfp, 1, 3, 0.0_rkind)! Cartesian points on the plasma boundary

        prevcstride = huge(prevcstride)
      end if

      call sallocate(Dxyz, 1, 3, 1, Ntz, 0.0_rkind)! Cartesian components of computational boundary position
      call sallocate(Nxyz, 1, 3, 1, Ntz, 0.0_rkind)! Cartesian components of computational boundary normal

      lvol = Mvol
      lss = 1.0_rkind
      Lcurvature = 1
      Lcoordinatesingularity = .false. ! will only require normal field on outer interface = computational boundary

      call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)! will need Rij, Zij: THE COMPUTATIONAL BOUNDARY DOES NOT CHANGE

      do kk = 0, Nz - 1
        zeta = kk * pi2nfp / Nz

        if (Igeometry == 3) then
          cszeta(0:1) = [cos(zeta), sin(zeta)]
        end if

        do jj = 0, Nt - 1
          teta = jj * pi2 / Nt
          jk = 1 + jj + kk * Nt

          select case (Igeometry)
          case (1) ! Igeometry = 1
            Dxyz(1:3, jk) = [teta, zeta, Rij(jk, 0, 0)]
            Nxyz(1:3, jk) = [-Rij(jk, 2, 0), -Rij(jk, 3, 0), 1.0_rkind]
          case (2) ! Igeometry = 2
            call fatal(.true., __FILE__, "free-boundary calculations not yet implemented in cylindrical geometry ")

          case (3) ! Igeometry = 3
            Dxyz(1:3, jk) = [Rij(jk, 0, 0) * cszeta(0), Rij(jk, 0, 0) * cszeta(1), Zij(jk, 0, 0)]
            Nxyz(1:3, jk) = [Rij(jk, 2, 0) * cszeta(1) * Zij(jk, 3, 0) - Zij(jk, 2, 0) * (Rij(jk, 3, 0) * cszeta(1) +&
                & Rij(jk, 0, 0) * cszeta(0)) , -Rij(jk, 2, 0) * cszeta(0) * Zij(jk, 3, 0) + Zij(jk, 2, 0) * (Rij(jk, 3, 0)&
                & * cszeta(0) - Rij(jk, 0, 0) * cszeta(1)) , Rij(jk, 0, 0) * Rij(jk, 2, 0)]
          end select

        end do

      end do

    end if ! Lfreebound > 1

    if (Lconstraint == 3) then
      Localconstraint = .false.
    else
      Localconstraint = .true.
    end if
  end subroutine initialize_internal_arrays
end module preset_mod
