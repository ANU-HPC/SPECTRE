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
  use allglobal_mod, only: myid, cpus, mn, im, in, iRbc, iRbs, djkp, djkm, vvolume, dvolume, Rij, Zij, cosi,&
      & sini, dBdX, pi2pi2nfp, pi2pi2nfpquart, sg, inv_jac, guvij
  use errors_mod, only: fatal
  use geometry_mod, only: compute_coordinates

  implicit none

  private
  public :: volume, lregion

contains

  subroutine volume(lvol, vflag, Nt, Nz)
    real(rkind) :: cput

    integer, intent(in) :: lvol, Nt, Nz
    integer, intent(inout) :: vflag

    integer :: Lcurvature
    integer :: jvol, ii, jj, mi, ni, mj, nj, innout

    real(rkind) :: vol(0:1), vint(1:(Nt*Nz))

    real(rkind) :: lss

    integer :: Ntz

    Ntz = Nt * Nz

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

      select case (Igeometry)

      case (1)

        vol(innout) = iRbc(1, jvol)

        if (dBdX%L .and. dBdX%innout == innout .and. dBdX%ii == 1) then ! compute derivative of volume
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

              if (dBdX%L .and. dBdX%innout == innout .and. dBdX%ii == ii) then ! compute derivative of volume
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

              if (dBdX%L .and. dBdX%innout == innout .and. dBdX%ii == ii) then ! compute derivative of volume
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
          call compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn, Rij, Zij, sg, inv_jac, guvij)

          vint = Rij(1:Ntz, 0, 0) * (Zij(1:Ntz, 0, 0) * Rij(1:Ntz, 2, 0) - Zij(1:Ntz, 2, 0) * Rij(1:Ntz, 0, 0))
          vol(innout) = 4.0_rkind * sum(vint) / float(Ntz)

          if (dBdX%L .and. dBdX%innout == innout) then ! compute derivative of volume

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

    if (dBdX%innout == 0) dvolume = -dvolume

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
