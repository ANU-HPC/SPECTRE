! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module diagnostics_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind
  use constants_mod, only: pi2
  use io_mod, only: ounit
  use input_list_mod, only: Wmacros, Wcompute_beltrami_error, Nvol, Lrad, mu, mpol, Igeometry, Nfp, Lerrortype, enforce_stell_sym
  use allglobal_mod, only: myid, cpus, ivol, im, in, cheby, zernike, Ate, Aze, Ato, Azo, sg, inv_jac, guvij, Rij, Zij, Nt, Nz,&
      & efmn, ofmn, cfmn, sfmn, Lcoordinatesingularity, beltramierror, Rij, Zij, Node, pi2nfp, ivol, RTT, TT,&
      & dtflux, dpflux
  use errors_mod, only: fatal
  use base_functions_mod, only: get_cheby_d2, get_zernike_d2
  use magnetic_field_mod, only: compute_magnetic_field
  use geometry_mod, only: compute_coordinates
  use numerical_mod, only: invfft

  implicit none(type, external)

  private
  public :: compute_beltrami_error

  external :: CDGQF

contains

  subroutine compute_beltrami_error(lvol, Ntz, lquad, mn)
    integer, intent(in) :: lvol
    integer, intent(in) :: Ntz
    integer, intent(in) :: lquad
    integer, intent(in) :: mn

    real(rkind) :: cput, cpui
    integer :: jquad, Lcurvature, ll, ii, jj, kk, uu, ideriv, twolquad, mm, jk

    real(rkind) :: lss, sbar, gBu(1:Ntz, 1:3, 0:3), gJu(1:Ntz, 1:3), jerror(1:3), jerrormax(1:3), intvol
    real(rkind) :: B_cartesian(1:Ntz, 1:3), J_cartesian(1:Ntz, 1:3)

    real(rkind) :: Atemn(1:mn, 0:2), Azemn(1:mn, 0:2), Atomn(1:mn, 0:2), Azomn(1:mn, 0:2)

    integer :: itype, icdgqf
    real(rkind) :: aa, bb, cc, dd, weight(1:lquad + 1), abscis(1:lquad), workfield(1:2 * lquad)

    real(rkind) :: zeta, teta, st(2), Bst(2), gBzeta

    itype = 1
    aa = -1.0_rkind
    bb = +1.0_rkind
    cc = 0.0_rkind
    dd = 0.0_rkind
    twolquad = 2 * lquad

    ! Prepare Gaussian quadrature
    call CDGQF(lquad, abscis(1:lquad), weight(1:lquad), itype, aa, bb, twolquad, workfield(1:twolquad), icdgqf)
    weight(lquad + 1) = 0.0_rkind

    cput = MPI_Wtime()
    select case (icdgqf)
    case (0)
      if (Wcompute_beltrami_error) write(ounit, 1000)cput - cpus, myid, lvol, icdgqf, "success", abscis(1:lquad)
    case (1)
      write(ounit, 1000)cput - cpus, myid, lvol, icdgqf, "failed", abscis(1:lquad)
    case (2)
      write(ounit, 1000)cput - cpus, myid, lvol, icdgqf, "input error", abscis(1:lquad)
    case (3)
      write(ounit, 1000)cput - cpus, myid, lvol, icdgqf, "input error", abscis(1:lquad)
    case (4)
      write(ounit, 1000)cput - cpus, myid, lvol, icdgqf, "weight overflow", abscis(1:lquad)
    case (5)
      write(ounit, 1000)cput - cpus, myid, lvol, icdgqf, "weight zero", abscis(1:lquad)
    case (6)
      write(ounit, 1000)cput - cpus, myid, lvol, icdgqf, "failed", abscis(1:lquad)
    case default
      write(ounit, 1000)cput - cpus, myid, lvol, icdgqf, "weird", abscis(1:lquad)
    end select

    if (Wcompute_beltrami_error) write(ounit, 1001)weight(1:lquad)

    1000 format("compute_beltrami_error : ", f10.2, " : myid=", i3, " ; lvol=", i3, " ; icdgqf=", i3, " ; ", a15, " ", :, &
        " abscissae =", 99f10.6)
    1001 format("compute_beltrami_error : ", 10x, " :       ", 3x, "          ", 3x, "            ", 3x, "    ", 15x, " ", :, &
        " weights   =", 99f10.6)

    call fatal(icdgqf /= 0, __FILE__, "failed to construct Gaussian integration abscisae and weights ")

    jerror(1:3) = 0.0_rkind
    ideriv = 0 ! 3.0_rkind components of the error in \curl B - mu B: initialize summation
    jerrormax(1:3) = 0.0_rkind
    intvol = 0.0_rkind

    do jquad = 1, lquad + 1 ! loop over radial sub-sub-grid (numerical quadrature)

      if (jquad == lquad + 1) then
        lss = 1.0_rkind
        sbar = 1.0_rkind
      else
        lss = abscis(jquad)
        sbar = (lss + 1.0_rkind) * 1.0_rkind / 2.0_rkind
      end if

      Lcurvature = 2

      call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn)! returns coordinates, metrics, . . .

      if (Lcoordinatesingularity) then ! Zernike 1 Jul 2019
        call get_zernike_d2(sbar, Lrad(lvol), mpol, zernike)
      else
        call get_cheby_d2(lss, Lrad(lvol), cheby(0:Lrad(lvol), 0:2))
      end if

      Atemn(1:mn, 0:2) = 0.0_rkind ! initialize summation over Chebyshev/Zernike polynomials
      Azemn(1:mn, 0:2) = 0.0_rkind
      if (.not.enforce_stell_sym) then
        Atomn(1:mn, 0:2) = 0.0_rkind
        Azomn(1:mn, 0:2) = 0.0_rkind
      else
        Atomn(1:mn, 0:2) = 0.0_rkind ! these are used below
        Azomn(1:mn, 0:2) = 0.0_rkind
      end if

      if (Lcoordinatesingularity) then
        do ll = 0, Lrad(lvol)! radial (Chebyshev) resolution of magnetic vector potential

          do ii = 1, mn ! Fourier resolution of magnetic vector potential
            mm = im(ii)
            if (ll < mm) cycle
            if (mod(ll + mm, 2) > 0) cycle

            Atemn(ii, 0) = Atemn(ii, 0) + Ate(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 0)
            Atemn(ii, 1) = Atemn(ii, 1) + Ate(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 1) * 1.0_rkind / 2.0_rkind
            Atemn(ii, 2) = Atemn(ii, 2) + Ate(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 2) * 1.0_rkind / 4.0_rkind

            Azemn(ii, 0) = Azemn(ii, 0) + Aze(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 0)
            Azemn(ii, 1) = Azemn(ii, 1) + Aze(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 1) * 1.0_rkind / 2.0_rkind
            Azemn(ii, 2) = Azemn(ii, 2) + Aze(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 2) * 1.0_rkind / 4.0_rkind

            if (.not.enforce_stell_sym) then

              Atomn(ii, 0) = Atomn(ii, 0) + Ato(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 0)
              Atomn(ii, 1) = Atomn(ii, 1) + Ato(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 1) * 1.0_rkind / 2.0_rkind
              Atomn(ii, 2) = Atomn(ii, 2) + Ato(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 2) * 1.0_rkind / 4.0_rkind

              Azomn(ii, 0) = Azomn(ii, 0) + Azo(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 0)
              Azomn(ii, 1) = Azomn(ii, 1) + Azo(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 1) * 1.0_rkind / 2.0_rkind
              Azomn(ii, 2) = Azomn(ii, 2) + Azo(lvol, ideriv, ii)%s(ll) * zernike(ll, mm, 2) * 1.0_rkind / 4.0_rkind
            end if
          end do
        end do

      else
        do ll = 0, Lrad(lvol)! radial (Chebyshev) resolution of magnetic vector potential
          do ii = 1, mn ! Fourier resolution of magnetic vector potential
            Atemn(ii, 0) = Atemn(ii, 0) + Ate(lvol, ideriv, ii)%s(ll) * cheby(ll, 0)
            Atemn(ii, 1) = Atemn(ii, 1) + Ate(lvol, ideriv, ii)%s(ll) * cheby(ll, 1)
            Atemn(ii, 2) = Atemn(ii, 2) + Ate(lvol, ideriv, ii)%s(ll) * cheby(ll, 2)

            Azemn(ii, 0) = Azemn(ii, 0) + Aze(lvol, ideriv, ii)%s(ll) * cheby(ll, 0)
            Azemn(ii, 1) = Azemn(ii, 1) + Aze(lvol, ideriv, ii)%s(ll) * cheby(ll, 1)
            Azemn(ii, 2) = Azemn(ii, 2) + Aze(lvol, ideriv, ii)%s(ll) * cheby(ll, 2)

            if (.not.enforce_stell_sym) then
              Atomn(ii, 0) = Atomn(ii, 0) + Ato(lvol, ideriv, ii)%s(ll) * cheby(ll, 0)
              Atomn(ii, 1) = Atomn(ii, 1) + Ato(lvol, ideriv, ii)%s(ll) * cheby(ll, 1)
              Atomn(ii, 2) = Atomn(ii, 2) + Ato(lvol, ideriv, ii)%s(ll) * cheby(ll, 2)

              Azomn(ii, 0) = Azomn(ii, 0) + Azo(lvol, ideriv, ii)%s(ll) * cheby(ll, 0)
              Azomn(ii, 1) = Azomn(ii, 1) + Azo(lvol, ideriv, ii)%s(ll) * cheby(ll, 1)
              Azomn(ii, 2) = Azomn(ii, 2) + Azo(lvol, ideriv, ii)%s(ll) * cheby(ll, 2)
            end if
          end do
        end do
      end if

      ofmn(1:mn) = -im(1:mn) * Azemn(1:mn, 0) - in(1:mn) * Atemn(1:mn, 0)
      efmn(1:mn) = +im(1:mn) * Azomn(1:mn, 0) + in(1:mn) * Atomn(1:mn, 0)
      sfmn(1:mn) = -im(1:mn) * Azemn(1:mn, 1) - in(1:mn) * Atemn(1:mn, 1)
      cfmn(1:mn) = +im(1:mn) * Azomn(1:mn, 1) + in(1:mn) * Atomn(1:mn, 1)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 1, 0), &
          gBu(1:Ntz, 1, 1))!  (gB^s)   , d(gB^s)/ds

      efmn(1:mn) = -im(1:mn) * im(1:mn) * Azemn(1:mn, 0) - im(1:mn) * in(1:mn) * Atemn(1:mn, 0)
      ofmn(1:mn) = -im(1:mn) * im(1:mn) * Azomn(1:mn, 0) - im(1:mn) * in(1:mn) * Atomn(1:mn, 0)
      cfmn(1:mn) = +in(1:mn) * im(1:mn) * Azemn(1:mn, 0) + in(1:mn) * in(1:mn) * Atemn(1:mn, 0)
      sfmn(1:mn) = +in(1:mn) * im(1:mn) * Azomn(1:mn, 0) + in(1:mn) * in(1:mn) * Atomn(1:mn, 0)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 1, 2), &
          gBu(1:Ntz, 1, 3))! d(gB^s)/dt, d(gB^s)/dz

      efmn(1:mn) = -Azemn(1:mn, 1)
      ofmn(1:mn) = -Azomn(1:mn, 1)
      cfmn(1:mn) = -Azemn(1:mn, 2)
      sfmn(1:mn) = -Azomn(1:mn, 2)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 2, 0), &
          gBu(1:Ntz, 2, 1))!  (gB^t)   , d(gB^t)/ds

      ofmn(1:mn) = +im(1:mn) * Azemn(1:mn, 1)
      efmn(1:mn) = -im(1:mn) * Azomn(1:mn, 1)
      sfmn(1:mn) = -in(1:mn) * Azemn(1:mn, 1)
      cfmn(1:mn) = +in(1:mn) * Azomn(1:mn, 1)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 2, 2), &
          gBu(1:Ntz, 2, 3))! d(gB^t)/dt, d(gB^t)/dz

      efmn(1:mn) = +Atemn(1:mn, 1)
      ofmn(1:mn) = +Atomn(1:mn, 1)
      cfmn(1:mn) = +Atemn(1:mn, 2)
      sfmn(1:mn) = +Atomn(1:mn, 2)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 3, 0), &
          gBu(1:Ntz, 3, 1))!  (gB^z)   , d(gB^z)/ds

      ofmn(1:mn) = -im(1:mn) * Atemn(1:mn, 1)
      efmn(1:mn) = +im(1:mn) * Atomn(1:mn, 1)
      sfmn(1:mn) = +in(1:mn) * Atemn(1:mn, 1)
      cfmn(1:mn) = -in(1:mn) * Atomn(1:mn, 1)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), gBu(1:Ntz, 3, 2), &
          gBu(1:Ntz, 3, 3))! d(gB^z)/dt, d(gB^z)/dz

      do ii = 1, 3
        select case (ii)
        case (1)
          jj = 2
          kk = 3
        case (2)
          jj = 3
          kk = 1
        case (3)
          jj = 1
          kk = 2
        end select

        gJu(1:Ntz, ii) = 0.0_rkind

        do uu = 1, 3 ! summation over uu

          gJu(1:Ntz, ii) = gJu(1:Ntz, ii) + (gBu(1:Ntz, uu, jj) * guvij(1:Ntz, uu, kk, 0) &
              + gBu(1:Ntz, uu, 0) * guvij(1:Ntz, uu, kk, jj) &
              - gBu(1:Ntz, uu, 0) * guvij(1:Ntz, uu, kk, 0) * sg(1:Ntz, jj) * inv_jac(1:Ntz)) &
              - (gBu(1:Ntz, uu, kk) * guvij(1:Ntz, uu, jj, 0) &
              + gBu(1:Ntz, uu, 0) * guvij(1:Ntz, uu, jj, kk) &
              - gBu(1:Ntz, uu, 0) * guvij(1:Ntz, uu, jj, 0) * sg(1:Ntz, kk) * inv_jac(1:Ntz))
        end do

        gJu(1:Ntz, ii) = gJu(1:Ntz, ii) * inv_jac(1:Ntz)
      end do

      select case (Igeometry)
      case (1)
      case (2)
      case (3)
        B_cartesian(:, 1) = (gBu(1:Ntz, 1, 0) * Rij(1:Ntz, 1, 0) + gBu(1:Ntz, 2, 0) * Rij(1:Ntz, 2, 0) + gBu(1:Ntz, 3, 0) &
            * Rij(1:Ntz, 3, 0)) * inv_jac(1:Ntz)
        B_cartesian(:, 2) = (gBu(1:Ntz, 1, 0) * Zij(1:Ntz, 1, 0) + gBu(1:Ntz, 2, 0) * Zij(1:Ntz, 2, 0) + gBu(1:Ntz, 3, 0) &
            * Zij(1:Ntz, 3, 0)) * inv_jac(1:Ntz)
        B_cartesian(:, 3) = (gBu(1:Ntz, 3, 0) * Rij(1:Ntz, 0, 0)) * inv_jac(1:Ntz)

        J_cartesian(:, 1) = (gJu(1:Ntz, 1) * Rij(1:Ntz, 1, 0) + gJu(1:Ntz, 2) * Rij(1:Ntz, 2, 0) + gJu(1:Ntz, 3) * &
            Rij(1:Ntz, 3, 0)) * inv_jac(1:Ntz)
        J_cartesian(:, 2) = (gJu(1:Ntz, 1) * Zij(1:Ntz, 1, 0) + gJu(1:Ntz, 2) * Zij(1:Ntz, 2, 0) + gJu(1:Ntz, 3) * &
            Zij(1:Ntz, 3, 0)) * inv_jac(1:Ntz)
        J_cartesian(:, 3) = (gJu(1:Ntz, 3) * Rij(1:Ntz, 0, 0)) * inv_jac(1:Ntz)
      end select

      if (Lerrortype == 1 .and. Igeometry == 3) then
        do ii = 1, 3
          jerror(ii) = jerror(ii) + weight(jquad) * sum(sg(1:Ntz, 0) * abs(J_cartesian(1:Ntz, ii) - mu(lvol) *&
              & B_cartesian(1:Ntz, ii)))

          jerrormax(ii) = max(jerrormax(ii), maxval(abs(J_cartesian(1:Ntz, ii) - mu(lvol) * B_cartesian(1:Ntz, ii))))
        end do
      else
        do ii = 1, 3
          jerror(ii) = jerror(ii) + weight(jquad) * sum(abs(gJu(1:Ntz, ii) - mu(lvol) * gBu(1:Ntz, ii, 0)))
          jerrormax(ii) = max(jerrormax(ii), maxval(abs(gJu(1:Ntz, ii) - mu(lvol) * gBu(1:Ntz, ii, 0)) * inv_jac(1:Ntz)))
        end do
      end if
      intvol = intvol + weight(jquad) * sum(sg(1:Ntz, 0))

    end do

    beltramierror(lvol, 1:3) = jerror(1:3) / Ntz ! the 'tranditional' SPECTRE error
    jerror(1:3) = jerror(1:3) / intvol
    beltramierror(lvol, 4:6) = jerror(1:3)! the volume average error
    beltramierror(lvol, 7:9) = jerrormax(1:3)! the max error

    if (Lerrortype == 1 .and. Igeometry == 3) then
      cput = MPI_Wtime()
      write(ounit, 1002)cput - cpus, myid, lvol, Lrad(lvol), jerror(1:3), cput - cpui ! write error to screen
      write(ounit, 1003)cput - cpus, myid, lvol, Lrad(lvol), jerrormax(1:3), cput - cpui ! write error to screen
    else
      cput = MPI_Wtime()
      write(ounit, 1004)cput - cpus, myid, lvol, Lrad(lvol), jerror(1:3), cput - cpui ! write error to screen
      write(ounit, 1005)cput - cpus, myid, lvol, Lrad(lvol), jerrormax(1:3), cput - cpui ! write error to screen
    end if

    1002 format("compute_beltrami_error : ", f10.2, " : myid=", i3, " ; lvol =", i3, " ; lrad =", i3, " ; AVG E^\R=", es23.15, &
        " , E^\Z =", es23.15, " , E^\phi=", es23.15, " ; time=", f8.2, "s ")
    1003 format("compute_beltrami_error : ", f10.2, " : myid=", i3, " ; lvol =", i3, " ; lrad =", i3, " ; MAX E^\R=", es23.15, &
        " , E^\Z =", es23.15, " , E^\phi=", es23.15, " ; time=", f8.2, "s ")
    1004 format("compute_beltrami_error : ", f10.2, " : myid=", i3, " ; lvol =", i3, " ; lrad =", i3, " ; AVG E^\s=", es23.15, &
        " , E^\t =", es23.15, " , E^\z=", es23.15, " ; time=", f8.2, "s ")
    1005 format("compute_beltrami_error : ", f10.2, " : myid=", i3, " ; lvol =", i3, " ; lrad =", i3, " ; MAX E^\s=", es23.15, &
        " , E^\t =", es23.15, " , E^\z=", es23.15, " ; time=", f8.2, "s ")

    ! check boundary condition
    jerror = 0.0_rkind

    do kk = 0, Nz - 1
      zeta = kk * pi2nfp / Nz
      do jj = 0, Nt - 1
        teta = jj * pi2 / Nt
        jk = 1 + jj + kk * Nt
        st(1:2) = [1.0_rkind, teta]

        call compute_magnetic_field(lvol, zeta, st(1:Node), Bst(1:Node), gBzeta)

        jerror(2) = max(jerror(2), abs(Bst(1) * gBzeta))

      end do
    end do

    if (.not.Lcoordinatesingularity) then
      do kk = 0, Nz - 1
        zeta = kk * pi2nfp / Nz
        do jj = 0, Nt - 1
          teta = jj * pi2 / Nt
          jk = 1 + jj + kk * Nt
          st(1:2) = [-1.0_rkind, teta]

          call compute_magnetic_field(lvol, zeta, st(1:Node), Bst(1:Node), gBzeta)

          jerror(1) = max(jerror(1), abs(Bst(1) * gBzeta))
        end do
      end do
    end if
    cput = MPI_Wtime()
    write(ounit, 1006)cput - cpus, myid, lvol, Lrad(lvol), jerror(1:2), cput - cpui ! write error to screen

    ! check fluxes
    Bst = 0.0_rkind

    if (Lcoordinatesingularity) then
      do ll = 0, Lrad(lvol)
        Bst(1) = Bst(1) + Ate(lvol, 0, 1)%s(ll) * RTT(ll, 0, 1, 0)
      end do
      Bst(1) = abs(Bst(1) - dtflux(lvol))
    else
      do ll = 0, Lrad(lvol)
        Bst(1) = Bst(1) + Ate(lvol, 0, 1)%s(ll) * TT(ll, 1, 0)
      end do
      Bst(1) = abs(Bst(1) - dtflux(lvol))
      do ll = 0, Lrad(lvol)
        Bst(2) = Bst(2) - Aze(lvol, 0, 1)%s(ll) * TT(ll, 1, 0)
      end do
      Bst(2) = abs(Bst(2) - dpflux(lvol))
    end if

    cput = MPI_Wtime()
    write(ounit, 1007)cput - cpus, myid, lvol, Lrad(lvol), Bst(1:2), cput - cpui ! write error to screen

    1006 format("compute_beltrami_error : ", f10.2, " : myid=", i3, " ; lvol =", i3, " ; lrad =", i3, " ; MAX gB^s(-1)=", es23.15, &
        ", gB^s(+1)=", es23.15, " ; time=", f8.2, "s ")
    1007 format("compute_beltrami_error : ", f10.2, " : myid=", i3, " ; lvol =", i3, " ; lrad =", i3, " ; dtfluxERR   =", es23.15, &
        ", dpfluxERR=", es23.15, " ; time=", f8.2, "s ")
  end subroutine compute_beltrami_error
end module diagnostics_mod
