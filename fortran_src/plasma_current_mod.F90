! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module plasma_current_mod
  use precisions_mod, only: rkind
  use constants_mod, only: pi2
  use numerical_mod, only: tfft, invfft
  use input_list_mod, only: Wmacros, Wcompute_plasma_current, Lrad, Wlbpol, Igeometry, Lcheck, Lconstraint, Lbdybnzero
  use allglobal_mod, only: im, in, mne, ime, ine, sg, inv_jac, guvij, efmn, ofmn, cfmn, sfmn, build_vector_potential, mn
  use geometry_mod, only: compute_coordinates

  implicit none(type, external)

  private
  public :: compute_plasma_current, lbpol

contains

  subroutine compute_plasma_current(lvol, mn, Nt, Nz, iflag, ldItGp)
    integer, intent(in) :: lvol, mn, Nt, Nz, iflag
    real(rkind), intent(out) :: ldItGp(0:1, -1:2)

    integer :: innout, ideriv, Lcurvature, ifail, ii
    real(rkind) :: lss
    real(rkind) :: Bsupt(1:Nt * Nz, -1:2), Bsupz(1:Nt * Nz, -1:2), Bsups(1:Nt * Nz, -1:2), Bsups_2(1:Nt * Nz, -1:2)

    real(rkind) :: ijreal(1:Nt * Nz), ijimag(1:Nt * Nz)

    integer :: Ntz

    Ntz = Nt * Nz

    innout = 0
    lss = 2.0_rkind * innout - 1.0_rkind ! lvol = Mvol ! internal shorthand/variables

    if (Lconstraint == -2) then
      innout = 1
      lss = 1.0_rkind
    end if

    do ideriv = -1, 2 ! labels derivative of magnetic field wrt enclosed fluxes

      if (iflag == 1 .and. ideriv /= 0) cycle ! derivatives of currents are not required
      if (iflag == 2 .and. ideriv < 0) cycle ! derivatives of currents w.r.t. geometry is  not required
      if (iflag == -1 .and. ideriv > 0) cycle ! derivatives of currents w.r.t. enclosed toroidal and enclosed poloidal flux
      ! are not required

      call build_vector_potential(lvol, innout, ideriv)

      call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), Bsupz(1:Ntz, ideriv), Bsupt(1:Ntz&
          , ideriv))! map to real space

      if (Lconstraint == -2) then
        if (Lbdybnzero) then
          Bsups = 0.0_rkind
        else
          call build_vector_potential(lvol, innout, ideriv)

          do ii = 1, mn
            efmn(ii) = -in(ii) * efmn(ii)! zeta  derivative of Ate
            ofmn(ii) = in(ii) * ofmn(ii)! zeta  derivative of Ato
            cfmn(ii) = -im(ii) * cfmn(ii)! theta derivative of Aze
            sfmn(ii) = im(ii) * sfmn(ii)! theta derivative of Azo
          end do

          call invfft(mn, im(1:mn), in(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), efmn(1:mn), Bsups(1:Ntz, ideriv), Bsups_2&
              (1:Ntz, ideriv))
          Bsups(1:Ntz, ideriv) = Bsups(1:Ntz, ideriv) + Bsups_2(1:ntz, ideriv)
        end if
      end if

    end do

    select case (iflag)
    case (-1)
      Lcurvature = 3 ! will need derivatives of metrics w.r.t. interface deformation
    case (1)
      Lcurvature = 1
    case (2)
      Lcurvature = 1
    end select

    call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn)! get "lower" metric elements evaluated on innout interface

    do ideriv = -1, 2 ! labels derivative of magnetic field wrt enclosed fluxes

      if (iflag == 1 .and. ideriv /= 0) cycle ! derivatives of currents
      ! are not required
      if (iflag == 2 .and. ideriv < 0) cycle ! derivatives of currents  wrt geometry
      ! is  not required
      if (iflag == -1 .and. ideriv > 0) cycle ! derivatives of currents  wrt enclosed toroidal and enclosed poloidal flux
      ! are not required

      if (Lbdybnzero) then
        ijreal(1:Ntz) = (-Bsupt(1:Ntz, ideriv) * guvij(1:Ntz, 2, 2, 0) + Bsupz(1:Ntz, ideriv) * guvij(1:Ntz, 2, 3, 0)) &
            * inv_jac(1:Ntz)
        ijimag(1:Ntz) = (-Bsupt(1:Ntz, ideriv) * guvij(1:Ntz, 2, 3, 0) + Bsupz(1:Ntz, ideriv) * guvij(1:Ntz, 3, 3, 0)) &
            * inv_jac(1:Ntz)
      else
        ijreal(1:Ntz) = (-Bsupt(1:Ntz, ideriv) * guvij(1:Ntz, 2, 2, 0) + Bsupz(1:Ntz, ideriv) * guvij(1:Ntz, 2, 3, 0) &
            + Bsups(1:Ntz, ideriv) * guvij(1:Ntz, 2, 1, 0)) * inv_jac(1:Ntz)
        ijimag(1:Ntz) = (-Bsupt(1:Ntz, ideriv) * guvij(1:Ntz, 2, 3, 0) + Bsupz(1:Ntz, ideriv) * guvij(1:Ntz, 3, 3, 0) &
            + Bsups(1:Ntz, ideriv) * guvij(1:Ntz, 1, 3, 0)) * inv_jac(1:Ntz)
      end if

      if (ideriv == -1) then ! add derivatives of metrics with respect to interface geometry
        if (Lbdybnzero) then
          ijreal(1:Ntz) = ijreal(1:Ntz) &
              + (-Bsupt(1:Ntz, 0) * guvij(1:Ntz, 2, 2, 1) + Bsupz(1:Ntz, 0) * guvij(1:Ntz, 2, 3, 1)) * inv_jac(1:Ntz)
          ijimag(1:Ntz) = ijimag(1:Ntz) &
              + (-Bsupt(1:Ntz, 0) * guvij(1:Ntz, 2, 3, 1) + Bsupz(1:Ntz, 0) * guvij(1:Ntz, 3, 3, 1)) * inv_jac(1:Ntz)
        else
          ijreal(1:Ntz) = ijreal(1:Ntz) &
              + (-Bsupt(1:Ntz, 0) * guvij(1:Ntz, 2, 2, 1) + Bsupz(1:Ntz, 0) * guvij(1:Ntz, 2, 3, 1) &
              + Bsups(1:Ntz, 0) * guvij(1:Ntz, 2, 1, 1)) * inv_jac(1:Ntz)
          ijimag(1:Ntz) = ijimag(1:Ntz) &
              + (-Bsupt(1:Ntz, 0) * guvij(1:Ntz, 2, 3, 1) + Bsupz(1:Ntz, 0) * guvij(1:Ntz, 3, 3, 1) &
              + Bsups(1:Ntz, 0) * guvij(1:Ntz, 1, 3, 1)) * inv_jac(1:Ntz)
        end if
      end if

      ifail = 0
      call tfft(ijreal(1:Ntz), ijimag(1:Ntz), mne, ime(1:mne), ine(1:mne), efmn(1:mne), ofmn(1:mne), cfmn(1:mne),&
          & sfmn(1:mne), ifail)

      ldItGp(0, ideriv) = efmn(1) * pi2 ! "toroidal", plasma  current
      ldItGp(1, ideriv) = cfmn(1) * pi2 ! "poloidal", linking current
    end do

  end subroutine compute_plasma_current

  !> @todo This subroutine is very similar to compute_plasma_current.f90 - maybe merge both in a single subroutine to simplify?
  subroutine lbpol(lvol, Mvol, Nt, Nz, Bt00, ideriv, iocons)
    integer, intent(in) :: lvol
    integer, intent(in) :: Mvol
    integer, intent(in) :: Nt
    integer, intent(in) :: Nz
    real(rkind), intent(inout) :: Bt00(1:Mvol, 0:1, -1:2)
    integer, intent(in) :: ideriv
    integer, intent(in) :: iocons

    integer :: Lcurvature, ifail
    real(rkind) :: lss
    real(rkind) :: dAt(1:(Nt*Nz)), dAz(1:(Nt*Nz)), Bt(1:(Nt*Nz)), Bz(1:(Nt*Nz)), dAt0(1:(Nt*Nz)), dAz0(1:(Nt*Nz))

    integer :: Ntz

    Ntz = Nt * Nz

    ! Lcurvature:             Controls what the routine compute_coordinates computes.
    ! lint:                   Interface number
    ! vvol, ideriv, ii, ll:   Some iteration variables
    ! lAte, lAze, lAto, lAzo: Reconstructed vector potential

    ! iocons=0 -> inner boundary of volume (s=-1) and iocons=1 -> outer boundary (s=1)

    lss = 2.0_rkind * iocons - 1.0_rkind

    ! First get the metric component and jacobian
    Lcurvature = 1
    call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn)! get guvij and sg

    ! Then compute the vector potential and its derivatives.
    call build_vector_potential(lvol, iocons, ideriv)

    ! Inverse Fourier transform to map to real space
    call invfft(mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), dAt(1:Ntz), dAz(1:Ntz))!
    !get covariant component of dA / contravariant of B

    ! Construct covariant Fourier components of B
    Bt(1:Ntz) = (-dAz(1:Ntz) * guvij(1:Ntz, 2, 2, 0) + dAt(1:Ntz) * guvij(1:Ntz, 2, 3, 0)) * inv_jac(1:Ntz)
    Bz(1:Ntz) = (-dAz(1:Ntz) * guvij(1:Ntz, 2, 3, 0) + dAt(1:Ntz) * guvij(1:Ntz, 3, 3, 0)) * inv_jac(1:Ntz)

    if (ideriv == -1) then

      ! Get derivatives of metric element
      Lcurvature = 3
      call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn)! get sg times d/dx (g_mu,nu / sg)

      ! Compute vector potential without taking derivatives
      call build_vector_potential(lvol, iocons, 0)

      ! And now add variation of metric contribution
      call invfft(mn, im, in, efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), dAt0(1:Ntz), dAz0(1:Ntz))! get
      ! covariant component of dA without derivatives

      Bt(1:Ntz) = Bt(1:Ntz) + (-dAz0(1:Ntz) * guvij(1:Ntz, 2, 2, 1) + dAt0(1:Ntz) * guvij(1:Ntz, 2, 3, 1)) * inv_jac(1:Ntz)&
          & ! Add
      ! metric derivatives
      Bz(1:Ntz) = Bz(1:Ntz) + (-dAz0(1:Ntz) * guvij(1:Ntz, 2, 3, 1) + dAt0(1:Ntz) * guvij(1:Ntz, 3, 3, 1)) * inv_jac(1:Ntz)

    end if

    ! Fourier transform, map to Fourier space
    ifail = 0
    call tfft(Bt(1:Ntz), Bz(1:Ntz), mn, im(1:mn), in(1:mn), efmn(1:mn), ofmn(1:mn), cfmn(1:mn), sfmn(1:mn), ifail)

    Bt00(lvol, iocons, ideriv) = efmn(1)
  end subroutine lbpol

end module plasma_current_mod
