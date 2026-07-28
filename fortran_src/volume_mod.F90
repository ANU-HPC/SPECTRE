! SPDX-FileCopyrightText: 2026 Proxima Fusion
! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module volume_mod
#include "mpi_precomp.h"
  USE_MPI

  use precisions_mod, only: rkind, vsmall, small
  use io_mod, only: ounit
  use input_list_mod, only: Wvolume, Igeometry, Nvol, pscale, rpol, rtor, enforce_stell_sym
  use cputiming_mod
  ! Rij/Zij/sg/inv_jac/guvij are NOT imported: volume() uses its own local scratch instead of the
  ! module globals; Lcoordinatesingularity is likewise an argument, not an import.
  use allglobal_mod, only: myid, cpus, mn, im, in, iRbc, iRbs, djkp, djkm, vvolume, dvolume, cosi,&
      & sini, dBdX, pi2pi2nfp, pi2pi2nfpquart
  use errors_mod, only: fatal
  use force_jvp_context_mod, only: jvp_context_borrow_direction, jvp_context_is_active, &
      jvp_context_raise_status
  use geometry_mod, only: compute_coordinates
  use numerical_mod, only: invfft

  implicit none

  private
  public :: volume, lregion

contains

  subroutine volume(lvol, vflag, Nt, Nz, Lcoordinatesingularity)
    real(rkind) :: cput

    integer, intent(in) :: lvol, Nt, Nz
    integer, intent(inout) :: vflag
    logical, intent(in) :: Lcoordinatesingularity

    integer :: Lcurvature
    integer :: jvol, ii, jj, mi, ni, mj, nj, innout

    real(rkind) :: vol(0:1), vint(1:(Nt*Nz))

    ! General-direction scratch: displacement fields of the current boundary and their
    ! theta derivatives, and the per-boundary sign of the enclosed-volume difference
    real(rkind) :: DRg(1:(Nt*Nz)), DZg(1:(Nt*Nz)), DRgt(1:(Nt*Nz)), DZgt(1:(Nt*Nz)), dsgn

    real(rkind) :: lss
    real(rkind), pointer, contiguous :: dirRc(:, :), dirRs(:, :), dirZc(:, :), dirZs(:, :)

    integer :: Ntz
    logical :: general_direction

    ! local metric scratch; only Rij/Zij are read here, the rest are required arguments of compute_coordinates
    real(rkind) :: Rij(1:(Nt*Nz), 0:3, 0:3), Zij(1:(Nt*Nz), 0:3, 0:3)
    real(rkind) :: sg(1:(Nt*Nz), 0:3), inv_jac(1:(Nt*Nz))
    real(rkind) :: guvij(1:(Nt*Nz), 0:3, 0:3, -1:3)

    Ntz = Nt * Nz
    general_direction = jvp_context_is_active()
    if (general_direction) call jvp_context_borrow_direction(dirRc, dirRs, dirZc, dirZs)

    if (lvol > Nvol) then
      ! this can only be the vacuum region: provide default value
      vvolume(lvol) = 1.0_rkind
      dvolume = 0.0_rkind
      return
    end if

    vol(0:1) = 0.0_rkind ! initialize
    dvolume = 0.0_rkind ! derivatives of volume wrt interface inner/outer, R/Z harmonic

    do innout = 0, 1 ! will subtract inner volume from outer volume to obtain enclosed volume

      jvol = lvol - 1 + innout ! labels inner or outer interface
      dsgn = 2.0_rkind * innout - 1.0_rkind ! sign of this boundary in vol(1) - vol(0); the
      ! general-direction mode sums both boundaries so it applies the sign per contribution
      ! instead of via the trailing dBdX%innout flip

      select case (Igeometry)

      case (1)

        vol(innout) = iRbc(1, jvol)

        if (general_direction) then
          if (dBdX%L) dvolume = dvolume + dsgn * dirRc(1, innout)
        else if (dBdX%L .and. dBdX%innout == innout .and. dBdX%ii == 1) then ! compute derivative of volume
          if (dBdX%issym == 0) dvolume = 1.0_rkind ! note that the sign factor for the lower interface is included below
        end if

        ! respecting rpol, rtor sizes of slab
        vol(innout) = vol(innout) * rpol * rtor

      case (2)

        if (enforce_stell_sym) then

          do ii = 1, mn
            mi = im(ii)
            ni = in(ii)

            do jj = 1, mn
              mj = im(jj)
              nj = in(jj)

              vol(innout) = vol(innout) + iRbc(ii, jvol) * iRbc(jj, jvol) * (djkp(ii, jj) + djkm(ii, jj))

              if (general_direction) then
                if (dBdX%L) then
                  dvolume = dvolume + dsgn * dirRc(ii, innout) * iRbc(jj, jvol) &
                      * (djkp(jj, ii) + djkm(jj, ii) + djkp(ii, jj) + djkm(ii, jj))
                end if
              else if (dBdX%L .and. dBdX%innout == innout .and. dBdX%ii == ii) then ! compute derivative of volume
                dvolume = dvolume + iRbc(jj, jvol) * (djkp(jj, ii) + djkm(jj, ii) + djkp(ii, jj) + djkm(ii, jj))
              end if

            end do

          end do

        else ! .not.enforce_stell_sym

          do ii = 1, mn
            mi = im(ii)
            ni = in(ii)
            do jj = 1, mn
              mj = im(jj)
              nj = in(jj)

              vol(innout) = vol(innout) + iRbc(ii, jvol) * iRbc(jj, jvol) * (djkp(ii, jj) + djkm(ii, jj)) + iRbs(ii,&
                  & jvol) * iRbs(jj, jvol) * (djkp(ii, jj) - djkm(ii, jj))

              if (general_direction) then
                if (dBdX%L) then
                  if (dirRs(ii, innout) /= 0.0_rkind) call jvp_context_raise_status(lvol, 6)
                  dvolume = dvolume + dsgn * dirRc(ii, innout) * iRbc(jj, jvol) &
                      * (djkp(jj, ii) + djkm(jj, ii) + djkp(ii, jj) + djkm(ii, jj))
                end if
              else if (dBdX%L .and. dBdX%innout == innout .and. dBdX%ii == ii) then ! compute derivative of volume
                if (dBdX%issym == 0) then !     stellarator-symmetric harmonic: dV/dRei
                  dvolume = dvolume + iRbc(jj, jvol) * (djkp(jj, ii) + djkm(jj, ii) + djkp(ii, jj) + djkm(ii, jj))
                else
                  call fatal(.true., __FILE__, "derivatives of volume under construction ")

                  dvolume = dvolume + iRbs(jj, jvol) * (djkp(jj, ii) - djkm(jj, ii) + djkp(ii, jj) - djkm(ii, jj)) !
                  !needs to be checked
                end if
              end if

            end do
          end do

        end if

      case (3)

        ! Replacing the trigonometric formula by fast Fourier transform

        if (lvol == 1 .and. innout == 0) then
          vol(1) = 0.0_rkind
          dvolume = 0.0_rkind
        else

          Lcurvature = 1

          lss = innout * 2.0_rkind - 1.0_rkind
          call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn, Lcoordinatesingularity, Rij, Zij, sg, inv_jac, guvij)

          vint = Rij(1:Ntz, 0, 0) * (Zij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) - Zij(1:Ntz, 2, 0) * Rij(1:Ntz, 0, 0))
          vol(innout) = 4.0_rkind * sum(vint) / float(Ntz)

          if (general_direction) then

            if (dBdX%L) then
              ! general-direction derivative of the enclosed volume: first variation of
              ! vint = R * (Z * R_theta - Z_theta * R) under the boundary displacement
              ! (DRg, DZg); the per-harmonic irz/issym cases below are its special cases.
              call invfft(mn, im(1:mn), in(1:mn), dirRc(1:mn, innout), dirRs(1:mn, innout), &
                  dirZc(1:mn, innout), dirZs(1:mn, innout), DRg(1:Ntz), DZg(1:Ntz))
              call invfft(mn, im(1:mn), in(1:mn), im(1:mn) * dirRs(1:mn, innout), -im(1:mn) * dirRc(1:mn, innout), &
                  im(1:mn) * dirZs(1:mn, innout), -im(1:mn) * dirZc(1:mn, innout), DRgt(1:Ntz), DZgt(1:Ntz))

              vint = DRg(1:Ntz) * (Zij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) - Zij(1:Ntz, 2, 0) * Rij(1:Ntz, 0, 0)) &
                  + Rij(1:Ntz, 0, 0) * (DZg(1:Ntz) * Rij(1:Ntz, 2, 0) + Zij(1:Ntz, 0, 0) * DRgt(1:Ntz) &
                  - DZgt(1:Ntz) * Rij(1:Ntz, 0, 0) - Zij(1:Ntz, 2, 0) * DRg(1:Ntz))

              dvolume = dvolume + dsgn * 4.0_rkind * sum(vint) / float(Ntz)
            end if

          else if (dBdX%L .and. dBdX%innout == innout) then ! compute derivative of volume

            ii = dBdX%ii ! shorthand

            if (dBdX%irz == 0) then ! compute derivatives wrt R

              if (dBdX%issym == 0) then
                vint = cosi(1:Ntz, ii) * (Zij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) - Zij(1:Ntz, 2, 0) * Rij(1:Ntz, 0, 0)) +&
                    & Rij(1:Ntz, 0, 0) * (-im(ii) * Zij(1:Ntz, 0, 0) * sini(1:Ntz, ii)) + Rij(1:Ntz, 0, 0) * (&
                    &-Zij(1:Ntz, 2, 0) * cosi(1:Ntz, ii))
              else
                vint = sini(1:Ntz, ii) * (Zij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) - Zij(1:Ntz, 2, 0) * Rij(1:Ntz, 0, 0)) +&
                    & Rij(1:Ntz, 0, 0) * (+im(ii) * Zij(1:Ntz, 0, 0) * cosi(1:Ntz, ii)) + Rij(1:Ntz, 0, 0) * (&
                    &-Zij(1:Ntz, 2, 0) * sini(1:Ntz, ii))
              end if

            else

              if (dBdX%issym == 0) then
                vint = Rij(1:Ntz, 0, 0) * (sini(1:Ntz, ii) * Rij(1:Ntz, 2, 0)) + Rij(1:Ntz, 0, 0) * (-im(ii) *&
                    & cosi(1:Ntz, ii) * Rij(1:Ntz, 0, 0))
              else
                vint = Rij(1:Ntz, 0, 0) * (cosi(1:Ntz, ii) * Rij(1:Ntz, 2, 0)) + Rij(1:Ntz, 0, 0) * (+im(ii) *&
                    & sini(1:Ntz, ii) * Rij(1:Ntz, 0, 0))
              end if

            end if

            dvolume = 4.0_rkind * sum(vint) / float(Ntz)

          else

            dvolume = 0.0_rkind

          end if
        end if

      end select
    end do

    select case (Igeometry)
    case (1)
      vvolume(lvol) = (vol(1) - vol(0)) * pi2pi2nfp
      dvolume = dvolume * pi2pi2nfp
    case (2)
      vvolume(lvol) = (vol(1) - vol(0)) * pi2pi2nfpquart
      dvolume = dvolume * pi2pi2nfpquart
    case (3)
      vvolume(lvol) = (vol(1) - vol(0)) * pi2pi2nfpquart * 1.0_rkind / 3.0_rkind
      dvolume = dvolume * pi2pi2nfpquart * 1.0_rkind / 3.0_rkind
    case (4)
      vvolume(lvol) = 1.0_rkind
      dvolume = 0.0_rkind ! this is under construction
      call fatal(abs(pscale) > vsmall, __FILE__, "need to compute volume ")

    case default
      call fatal(.true., __FILE__, "invalid Igeometry ")

    end select

    ! General-direction mode applies each boundary's sign while accumulating it above.
    if (dBdX%innout == 0 .and. .not. general_direction) dvolume = -dvolume

    if (Wvolume) then
      cput = MPI_Wtime()
      write(ounit, '("volume : ",f10.2," : myid=",i3," ; Igeometry=",i2," ; vvolume(",i3," ) =",es23.15" ")')cput -&
          & cpus, myid, Igeometry, lvol, vvolume(lvol)
    end if

    call fatal(vflag == 0 .and. vvolume(lvol) < small, __FILE__, "volume cannot be zero or negative. Check that theta is oriented correctly and R, phi, Z coordinates are right-handed (set flag Lchangeangle)")

    if (vvolume(lvol) < small) then
      write(ounit, '("volume : ", 10x ," : myid=",i3," ; lvol=",i3,' // &
          '" ; vvolume=",es13.5,"  volume cannot be zero or negative ")') &
          myid, lvol, vvolume(lvol)
      vvolume(lvol) = +9.9e+09_rkind
      vflag = 1
    else
      vflag = 0
    end if
  end subroutine volume

  subroutine lregion(vol, Igeometry, Nvol, Lcoordinatesingularity, Lplasmaregion, Lvacuumregion)
    integer, intent(in) :: vol
    integer, intent(in) :: Igeometry
    integer, intent(in) :: Nvol
    logical, intent(inout) :: Lcoordinatesingularity
    logical, intent(inout) :: Lplasmaregion
    logical, intent(inout) :: Lvacuumregion

    if (Igeometry == 1 .or. vol > 1) then
      Lcoordinatesingularity = .false.
    else
      Lcoordinatesingularity = .true.
    end if

    if (vol <= Nvol) then
      Lplasmaregion = .true.
    else
      Lplasmaregion = .false.
    end if

    Lvacuumregion = .not.Lplasmaregion
  end subroutine lregion
end module volume_mod
