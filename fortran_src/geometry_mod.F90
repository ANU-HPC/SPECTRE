! SPDX-FileCopyrightText: 2026 Proxima Fusion
! SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
!
! SPDX-License-Identifier: MIT

module geometry_mod
  use io_mod, only: stderr
  use precisions_mod, only: rkind
  use input_list_mod, only: Wcompute_coordinates, Igeometry, Ntor, rpol, rtor, enforce_stell_sym
  ! The metric outputs (Rij/Zij/sg/inv_jac/guvij) and Lcoordinatesingularity are passed in as
  ! arguments, not read from module globals: serial callers pass the module globals, the
  ! perturbed/threaded Jacobian path passes its per-thread/per-DOF values.
  use allglobal_mod, only: im, in, iRbc, iZbs, iRbs, iZbc, cosi, sini, &
      dBdX, dRodR, dRodZ, dZodR, dZodZ, gaussianabscissae, use_ext_mesh, Remn_ext, Romn_ext, Zemn_ext, Zomn_ext, &
      Ldirgen, dirRc, dirRs, dirZs, dirZc, dir_axR, dir_axRz, dir_axZ, dir_axZz
  use errors_mod, only: fatal
  use numerical_mod, only: invfft

  implicit none(type, external)

  private
  public :: compute_coordinates

contains

  subroutine compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn, Lcoordinatesingularity, &
      Rij, Zij, sg, inv_jac, guvij)
    integer, intent(in) :: lvol, Lcurvature, Nt, Nz, mn
    real(rkind), intent(in) :: lss
    logical, intent(in) :: Lcoordinatesingularity
    ! Metric outputs; intent(inout): the Lcurvature==0 early return leaves higher-derivative slots at their incoming values.
    real(rkind), intent(inout) :: Rij(1:(Nt*Nz), 0:3, 0:3), Zij(1:(Nt*Nz), 0:3, 0:3)
    real(rkind), intent(inout) :: sg(1:(Nt*Nz), 0:3), inv_jac(1:(Nt*Nz))
    real(rkind), intent(inout) :: guvij(1:(Nt*Nz), 0:3, 0:3, -1:3)

    integer :: ii, jj, kk, irz, innout, issym, signlss, radial_index
    real(rkind) :: Remn(1:mn, 0:2), Zomn(1:mn, 0:2), Romn(1:mn, 0:2), Zemn(1:mn, 0:2), alss, blss, sbar, fj(1:mn, 0:2)

    real(rkind) :: Dij(1:(Nt*Nz), 0:3), dguvij(1:(Nt*Nz), 1:3, 1:3), DRxij(1:(Nt*Nz), 0:3), DZxij(1:(Nt*Nz), 0:3)

    integer :: Ntz

    Ntz = Nt * Nz

    ! Zero only the slots that are returned as 0 (never overwritten below) on the current
    ! Lcurvature path; slots that are unconditionally overwritten before any read are not zeroed.
    ! Mapping (cross-checked slot by slot against the writes further down):
    !   Rij/Zij(:,0,0)          : always overwritten (invfft at Lcurvature>=0)         -> never zero here
    !   Rij/Zij(:,1:3,0),(0,1:3): overwritten once Lcurvature>=1                        -> zero only if Lcurvature==0
    !   Rij/Zij(:,1:3,1:3)      : overwritten only at Lcurvature==2                     -> zero unless Lcurvature==2
    !   sg(:,0), inv_jac        : overwritten at Lcurvature>=1 before any read          -> never zero here
    !   guvij(:,1:3,1:3,0)      : fully assigned at Lcurvature>=1 before any read       -> never zero here
    !   sg(:,1:3)               : (:,1) always set at Lcurvature>=2; (:,2:3) only at ==2 -> zero (:,2:3) unless ==2
    !   guvij(:,1:3,1:3,1:3)    : all set at ==2; only kk=1 set at ==3/4; none at ==5   -> zero the returned-0 slices
    if (Lcurvature == 0) then
      Rij(1:Ntz, 0:3, 0:3) = 0.0_rkind
      Zij(1:Ntz, 0:3, 0:3) = 0.0_rkind
    else if (Lcurvature /= 2) then
      Rij(1:Ntz, 1:3, 1:3) = 0.0_rkind
      Zij(1:Ntz, 1:3, 1:3) = 0.0_rkind
    end if
    if (Lcurvature > 1) then
      if (Lcurvature /= 2) sg(1:Ntz, 2:3) = 0.0_rkind
      if (Lcurvature == 5) then
        guvij(1:Ntz, 1:3, 1:3, 1:3) = 0.0_rkind
      else if (Lcurvature /= 2) then
        guvij(1:Ntz, 1:3, 1:3, 2:3) = 0.0_rkind
      end if
    end if
    
    Remn(1:mn, 0:2) = 0.0_rkind ! interpolated coordinate harmonics
    Zomn(1:mn, 0:2) = 0.0_rkind
    Romn(1:mn, 0:2) = 0.0_rkind
    Zemn(1:mn, 0:2) = 0.0_rkind

    if (Lcoordinatesingularity) then
      if (use_ext_mesh) then
        if (.not.allocated(Remn_ext)) write(stderr, *)"WARNING: Remn_ext not allocated"

        radial_index = minloc(abs(gaussianabscissae(:, 1) - lss), DIM=1)

        Remn(1:mn, 0:2) = Remn_ext(1:mn, radial_index, 0:2)
        Romn(1:mn, 0:2) = Romn_ext(1:mn, radial_index, 0:2)
        Zemn(1:mn, 0:2) = Zemn_ext(1:mn, radial_index, 0:2)
        Zomn(1:mn, 0:2) = Zomn_ext(1:mn, radial_index, 0:2)
      else
        sbar = (lss + 1.0_rkind) * 1.0_rkind / 2.0_rkind

        select case (Igeometry)
        case (2)
          fj(1:Ntor + 1, 0) = sbar ! these are the mj.eq.0 harmonics
          fj(Ntor + 2:mn, 0) = sbar ** (im(Ntor + 2:mn) + 1) ! these are the me.ne.0 harmonics
        case (3)
          fj(1:Ntor + 1, 0) = sbar ** 2 ! switch to sbar=r
          fj(Ntor + 2:mn, 0) = sbar ** im(Ntor + 2:mn)! these are the me.ne.0 harmonics
        case default
          call fatal(.true., __FILE__, "invalid Igeometry for Lcoordinatesingularity=T ")

        end select

        Remn(1:mn, 0) = iRbc(1:mn, 0) + (iRbc(1:mn, 1) - iRbc(1:mn, 0)) * fj(1:mn, 0)
        if (.not.enforce_stell_sym) then
          Romn(1:mn, 0) = iRbs(1:mn, 0) + (iRbs(1:mn, 1) - iRbs(1:mn, 0)) * fj(1:mn, 0)
        end if
        if (Igeometry == 3) then
          Zomn(1:mn, 0) = iZbs(1:mn, 0) + (iZbs(1:mn, 1) - iZbs(1:mn, 0)) * fj(1:mn, 0)
          if (.not.enforce_stell_sym) then
            Zemn(1:mn, 0) = iZbc(1:mn, 0) + (iZbc(1:mn, 1) - iZbc(1:mn, 0)) * fj(1:mn, 0)
          end if
        end if
      end if

    else

      alss = 1.0_rkind / 2.0_rkind * (1.0_rkind - lss)
      blss = 1.0_rkind / 2.0_rkind * (1.0_rkind + lss)

      Remn(1:mn, 0) = alss * iRbc(1:mn, lvol - 1) + blss * iRbc(1:mn, lvol)
      if (.not.enforce_stell_sym) then
        Romn(1:mn, 0) = alss * iRbs(1:mn, lvol - 1) + blss * iRbs(1:mn, lvol)
      end if
      if (Igeometry == 3) then
        Zomn(1:mn, 0) = alss * iZbs(1:mn, lvol - 1) + blss * iZbs(1:mn, lvol)
        if (.not.enforce_stell_sym) then
          Zemn(1:mn, 0) = alss * iZbc(1:mn, lvol - 1) + blss * iZbc(1:mn, lvol)
        end if
      end if

    end if

    call invfft(mn, im(1:mn), in(1:mn), Remn(1:mn, 0), Romn(1:mn, 0), Zemn(1:mn, 0), Zomn(1:mn, 0), Rij(1:Ntz, 0,&
        & 0), Zij(1:Ntz, 0, 0))! maps to real space

    if (Lcurvature == 0) return

    if (Lcoordinatesingularity) then
      if (.not.use_ext_mesh) then
        select case (Igeometry)
        case (2)
          fj(1:Ntor + 1, 1) = 1.0_rkind / 2.0_rkind ! these are the mj.eq.0 harmonics
          fj(Ntor + 2:mn, 1) = 1.0_rkind / 2.0_rkind * (im(Ntor + 2:mn) + 1.0_rkind) * fj(Ntor + 2:mn, 0) / sbar ! these
          ! are the
          ! me.ne.0 harmonics
        case (3)
          fj(1:Ntor + 1, 1) = sbar ! these are the mj.eq.0 harmonics
          fj(Ntor + 2:mn, 1) = 1.0_rkind / 2.0_rkind * im(Ntor + 2:mn) * fj(Ntor + 2:mn, 0) / sbar ! these are the me.ne.0
          ! harmonics
        case default
          call fatal(.true., __FILE__, "invalid Igeometry for Lcoordinatesingularity=T and Lcurvature.ne.0 ")

        end select

        Remn(1:mn, 1) = (iRbc(1:mn, 1) - iRbc(1:mn, 0)) * fj(1:mn, 1)
        if (.not.enforce_stell_sym) then
          Romn(1:mn, 1) = (iRbs(1:mn, 1) - iRbs(1:mn, 0)) * fj(1:mn, 1)
        end if
        if (Igeometry == 3) then
          Zomn(1:mn, 1) = (iZbs(1:mn, 1) - iZbs(1:mn, 0)) * fj(1:mn, 1)
          if (.not.enforce_stell_sym) then
            Zemn(1:mn, 1) = (iZbc(1:mn, 1) - iZbc(1:mn, 0)) * fj(1:mn, 1)
          end if
        end if
      end if

    else ! matches if( Lcoordinatesingularity )

      Remn(1:mn, 1) = (-iRbc(1:mn, lvol - 1) + iRbc(1:mn, lvol)) * 1.0_rkind / 2.0_rkind
      if (.not.enforce_stell_sym) then
        Romn(1:mn, 1) = (-iRbs(1:mn, lvol - 1) + iRbs(1:mn, lvol)) * 1.0_rkind / 2.0_rkind
      end if
      if (Igeometry == 3) then
        Zomn(1:mn, 1) = (-iZbs(1:mn, lvol - 1) + iZbs(1:mn, lvol)) * 1.0_rkind / 2.0_rkind
        if (.not.enforce_stell_sym) then
          Zemn(1:mn, 1) = (-iZbc(1:mn, lvol - 1) + iZbc(1:mn, lvol)) * 1.0_rkind / 2.0_rkind
        end if
      end if

    end if

    ! Derivative w.r.t s
    call invfft(mn, im(1:mn), in(1:mn), Remn(1:mn, 1), Romn(1:mn, 1), Zemn(1:mn, 1), Zomn(1:mn, 1), Rij(1:Ntz, 1,&
        & 0), Zij(1:Ntz, 1, 0))! maps to real space

    ! Derivative w.r.t theta
    call invfft(mn, im(1:mn), in(1:mn), im(1:mn) * Romn(1:mn, 0), -im(1:mn) * Remn(1:mn, 0), im(1:mn) * Zomn(1:mn, 0), &
        &-im(1:mn) * Zemn(1:mn, 0), Rij(1:Ntz, 2, 0), Zij(1:Ntz, 2, 0))! maps to real space

    ! Derivative w.r.t zeta
    call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * Romn(1:mn, 0), in(1:mn) * Remn(1:mn, 0), -in(1:mn) * Zomn(1:mn, 0),&
        & in(1:mn) * Zemn(1:mn, 0), Rij(1:Ntz, 3, 0), Zij(1:Ntz, 3, 0))! maps to real space

    do ii = 1, 3
      Rij(1:Ntz, 0, ii) = Rij(1:Ntz, ii, 0)! just to complete workspace arrays
      Zij(1:Ntz, 0, ii) = Zij(1:Ntz, ii, 0)! just to complete workspace arrays
    end do

    guvij(1:Ntz, 0, 0, 0) = 1.0_rkind ! this is (only) required for the helicity integral

    select case (Igeometry)

    case (1) ! Igeometry=1 Cartesian

      sg(1:Ntz, 0) = Rij(1:Ntz, 1, 0) * rpol * rtor

      do ii = 1, 3
        do jj = ii, 3
          guvij(1:Ntz, ii, jj, 0) = Rij(1:Ntz, ii, 0) * Rij(1:Ntz, jj, 0)
        end do
      end do

      guvij(1:Ntz, 2, 2, 0) = guvij(1:Ntz, 2, 2, 0) + rpol * rpol
      guvij(1:Ntz, 3, 3, 0) = guvij(1:Ntz, 3, 3, 0) + rtor * rtor

    case (2) ! Igeometry=2 cylindrical

      sg(1:Ntz, 0) = Rij(1:Ntz, 1, 0) * Rij(1:Ntz, 0, 0)

      do ii = 1, 3
        do jj = ii, 3
          guvij(1:Ntz, ii, jj, 0) = Rij(1:Ntz, ii, 0) * Rij(1:Ntz, jj, 0)
        end do
      end do

      guvij(1:Ntz, 2, 2, 0) = guvij(1:Ntz, 2, 2, 0) + Rij(1:Ntz, 0, 0) * Rij(1:Ntz, 0, 0)
      guvij(1:Ntz, 3, 3, 0) = guvij(1:Ntz, 3, 3, 0) + 1.0_rkind

    case (3) ! Igeometry=3 toroidal

      sg(1:Ntz, 0) = Rij(1:Ntz, 0, 0) * (Zij(1:Ntz, 1, 0) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) * Zij(1:Ntz, 2, 0))

      do ii = 1, 3
        do jj = ii, 3
          guvij(1:Ntz, ii, jj, 0) = Rij(1:Ntz, ii, 0) * Rij(1:Ntz, jj, 0) + Zij(1:Ntz, ii, 0) * Zij(1:Ntz, jj, 0)
        end do
      end do

      guvij(1:Ntz, 3, 3, 0) = guvij(1:Ntz, 3, 3, 0) + (Rij(1:Ntz, 0, 0)) ** 2

    case default ! Igeometry

      call fatal(.true., __FILE__, "selected Igeometry not supported ")

    end select

    inv_jac(1:Ntz) = 1.0_rkind / sg(1:Ntz, 0)

    do ii = 2, 3
      do jj = 1, ii - 1
        guvij(1:Ntz, ii, jj, 0) = guvij(1:Ntz, jj, ii, 0)! complete metric array
      end do
    end do

    if (Lcurvature <= 1) return

    select case (Lcurvature)

    case (2) ! Lcurvature=2 get second derivatives of position wrt \s, \t & \z

      if (Lcoordinatesingularity) then
        select case (Igeometry)
        case (2)
          fj(1:Ntor + 1, 2) = 0.0_rkind ! these are the mj.eq.0 harmonics
          fj(Ntor + 2:mn, 2) = 1.0_rkind / 2.0_rkind * (im(Ntor + 2:mn)) * fj(Ntor + 2:mn, 1) / sbar ! these are the
          ! me.ne.0
          ! harmonics
        case (3)
          fj(1:Ntor + 1, 2) = 1.0_rkind / 2.0_rkind ! these are the mj.eq.0 harmonics
          fj(Ntor + 2:mn, 2) = 1.0_rkind / 2.0_rkind * (im(Ntor + 2:mn) - 1.0_rkind) * fj(Ntor + 2:mn, 1) / sbar ! these
          ! are the
          ! me.ne.0 harmonics
        case default
          call fatal(.true., __FILE__, "invalid Igeometry for Lcoordinatesingularity=T and Lcurvature=2 ")

        end select
        Remn(1:mn, 2) = (iRbc(1:mn, 1) - iRbc(1:mn, 0)) * fj(1:mn, 2)
        if (.not.enforce_stell_sym) then
          Romn(1:mn, 2) = (iRbs(1:mn, 1) - iRbs(1:mn, 0)) * fj(1:mn, 2)
        end if
        if (Igeometry == 3) then
          Zomn(1:mn, 2) = (iZbs(1:mn, 1) - iZbs(1:mn, 0)) * fj(1:mn, 2)
          if (.not.enforce_stell_sym) then
            Zemn(1:mn, 2) = (iZbc(1:mn, 1) - iZbc(1:mn, 0)) * fj(1:mn, 2)
          end if
        end if

      else

        Remn(1:mn, 2) = 0.0_rkind
        if (.not.enforce_stell_sym) then
          Romn(1:mn, 2) = 0.0_rkind
        end if
        if (Igeometry == 3) then
          Zomn(1:mn, 2) = 0.0_rkind
          if (.not.enforce_stell_sym) then
            Zemn(1:mn, 2) = 0.0_rkind
          end if
        end if

      end if

      call invfft(mn, im(1:mn), in(1:mn), Remn(1:mn, 2), Romn(1:mn, 2), Zemn(1:mn, 2), Zomn(1:mn, 2), Rij(1:Ntz,&
          & 1, 1), Zij(1:Ntz, 1, 1))! maps to real space

      call invfft(mn, im(1:mn), in(1:mn), +im(1:mn) * Romn(1:mn, 1), -im(1:mn) * Remn(1:mn, 1), im(1:mn) * Zomn(1:mn, 1),&
          & -im(1:mn) * Zemn(1:mn, 1), Rij(1:Ntz, 1, 2), Zij(1:Ntz, 1, 2))! maps to real space

      call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * Romn(1:mn, 1), in(1:mn) * Remn(1:mn, 1), -in(1:mn) * Zomn(1:mn, 1),&
          & in(1:mn) * Zemn(1:mn, 1), Rij(1:Ntz, 1, 3), Zij(1:Ntz, 1, 3))! maps to real space

      call invfft(mn, im(1:mn), in(1:mn), -im(1:mn) * im(1:mn) * Remn(1:mn, 0), -im(1:mn) * im(1:mn) * Romn(1:mn, 0), &
          &-im(1:mn) * im(1:mn) * Zemn(1:mn, 0), -im(1:mn) * im(1:mn) * Zomn(1:mn, 0), Rij(1:Ntz, 2, 2),&
          & Zij(1:Ntz, 2, 2))! maps
      ! to real space

      call invfft(mn, im(1:mn), in(1:mn), +im(1:mn) * in(1:mn) * Remn(1:mn, 0), im(1:mn) * in(1:mn) * Romn(1:mn, 0),&
          & im(1:mn) * in(1:mn) * Zemn(1:mn, 0), im(1:mn) * in(1:mn) * Zomn(1:mn, 0), Rij(1:Ntz, 2, 3), Zij(1:Ntz&
          &, 2, 3))! maps
      ! to real space

      call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * in(1:mn) * Remn(1:mn, 0), -in(1:mn) * in(1:mn) * Romn(1:mn, 0), &
          &-in(1:mn) * in(1:mn) * Zemn(1:mn, 0), -in(1:mn) * in(1:mn) * Zomn(1:mn, 0), Rij(1:Ntz, 3, 3),&
          & Zij(1:Ntz, 3, 3))! maps
      ! to real space

      do ii = 2, 3
        do jj = 1, ii - 1
          Rij(1:Ntz, ii, jj) = Rij(1:Ntz, jj, ii)
          Zij(1:Ntz, ii, jj) = Zij(1:Ntz, jj, ii)
        end do
      end do

      select case (Igeometry)

      case (1) ! Lcurvature=2 Igeometry=1  Cartesian

        do kk = 1, 3 ! kk labels derivative

          !    sg(1:Ntz, 0) = Rij(1:Ntz,1, 0)*rpol*rtor
          sg(1:Ntz, kk) = Rij(1:Ntz, 1, kk) * rpol * rtor

          do ii = 1, 3
            do jj = ii, 3
              guvij(1:Ntz, ii, jj, kk) = Rij(1:Ntz, ii, kk) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * Rij(1:Ntz, jj, kk)
            end do
          end do

        end do

      case (2) ! Lcurvature=2 Igeometry=2 cylindrical

        do kk = 1, 3 ! kk labels derivative

          !    sg(1:Ntz, 0) = Rij(1:Ntz, 1, 0) * Rij(1:Ntz, 0,0)
          sg(1:Ntz, kk) = Rij(1:Ntz, 1, kk) * Rij(1:Ntz, 0, 0) + Rij(1:Ntz, 1, 0) * Rij(1:Ntz, 0, kk)

          do ii = 1, 3
            do jj = ii, 3
              guvij(1:Ntz, ii, jj, kk) = Rij(1:Ntz, ii, kk) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * Rij(1:Ntz, jj, kk)
            end do
          end do

          guvij(1:Ntz, 2, 2, kk) = guvij(1:Ntz, 2, 2, kk) + Rij(1:Ntz, 0, kk) * Rij(1:Ntz, 0, 0) + Rij(1:Ntz, 0, 0) *&
              & Rij(1:Ntz, 0, kk)
          !guvij(1:Ntz,2,2,kk) = guvij(1:Ntz,2,2,kk) + Rij(1:Ntz,kk,0) * Rij(1:Ntz,0,0) + Rij(1:Ntz,0,0) * Rij(1:Ntz,kk,0)
          !guvij(1:Ntz,3,3,kk) = additional term is unity

        end do

      case (3) ! Lcurvature=2 Igeometry=3 toroidal

        do kk = 1, 3 ! kk labels derivative

          !    sg(1:Ntz, 0) = Rij(1:Ntz, 0,0) * ( Zij(1:Ntz,1, 0)*Rij(1:Ntz,2, 0) - Rij(1:Ntz,1, 0)*Zij(1:Ntz,2, 0) )
          sg(1:Ntz, kk) = Rij(1:Ntz, kk, 0) * (Zij(1:Ntz, 1, 0) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) * Zij(1:Ntz, 2, 0))&
              & + Rij(1:Ntz, 0, 0) * (Zij(1:Ntz, 1, kk) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, kk) * Zij(1:Ntz, 2, 0)) +&
              & Rij(1:Ntz, 0, 0) * (Zij(1:Ntz, 1, 0) * Rij(1:Ntz, 2, kk) - Rij(1:Ntz, 1, 0) * Zij(1:Ntz, 2, kk))

          sg(1:Ntz, kk) = sg(1:Ntz, kk)

          do ii = 1, 3
            do jj = ii, 3
              guvij(1:Ntz, ii, jj, kk) = Rij(1:Ntz, ii, kk) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * Rij(1:Ntz, jj, kk) &
                  &+ Zij(1:Ntz, ii, kk) * Zij(1:Ntz, jj, 0) + Zij(1:Ntz, ii, 0) * Zij(1:Ntz, jj, kk)
            end do
          end do

          guvij(1:Ntz, 3, 3, kk) = guvij(1:Ntz, 3, 3, kk) + (Rij(1:Ntz, 0, kk) * Rij(1:Ntz, 0, 0) + Rij(1:Ntz, 0, 0) *&
              & Rij(1:Ntz, 0, kk))

        end do

      case default

        call fatal(.true., __FILE__, "selected Igeometry not supported for Lcurvature.eq.2 ")

      end select

      do ii = 2, 3
        do jj = 1, ii - 1
          guvij(1:Ntz, ii, jj, 1:3) = guvij(1:Ntz, jj, ii, 1:3)
        end do
      end do

    case (3, 4, 5) ! Lcurvature=3,4,5  get derivatives wrt R_j and Z_j

      if (Ldirgen) then
        ! General-direction (JVP) mode: metric response to a full interface deformation
        ! instead of a single harmonic; see direction_metric_response.
        call direction_metric_response(lss, Lcurvature, Nt, Nz, mn, Lcoordinatesingularity, sbar, fj, &
            Rij, Zij, sg, inv_jac, guvij)
        return
      end if

      ii = dBdX%ii
      innout = dBdX%innout
      irz = dBdX%irz
      issym = dBdX%issym ! shorthand

      if (Lcoordinatesingularity) then
        if ( (irz == 0 .and. issym == 0) .or. (irz == 1 .and. issym == 1) ) then ! cosine harmonics

          Dij(1:Ntz, 0) = fj(ii, 0) * cosi(1:Ntz, ii)
          Dij(1:Ntz, 1) = fj(ii, 1) * cosi(1:Ntz, ii)
          Dij(1:Ntz, 2) = fj(ii, 0) * sini(1:Ntz, ii) * (-im(ii))
          Dij(1:Ntz, 3) = fj(ii, 0) * sini(1:Ntz, ii) * (+in(ii))

        else !   sine harmonics

          Dij(1:Ntz, 0) = fj(ii, 0) * sini(1:Ntz, ii)
          Dij(1:Ntz, 1) = fj(ii, 1) * sini(1:Ntz, ii)
          Dij(1:Ntz, 2) = fj(ii, 0) * cosi(1:Ntz, ii) * (+im(ii))
          Dij(1:Ntz, 3) = fj(ii, 0) * cosi(1:Ntz, ii) * (-in(ii))

        end if ! if( ( irz.eq.0 .and. issym.eq.1 ) .or. ...

        if (Lcurvature == 3 .or. Lcurvature == 4) then
          if (Igeometry == 3) then
            ! add the terms due to the moving coordinate axis
            if (irz == 0) then
              DRxij(1:Ntz, 0) = (1.0_rkind - sbar ** 2) * dRodR(1:Ntz, issym, ii)! dRx/dR
              DRxij(1:Ntz, 1) = -sbar * dRodR(1:Ntz, issym, ii)
              DRxij(1:Ntz, 2) = 0.0_rkind
              DRxij(1:Ntz, 3) = (1.0_rkind - sbar ** 2) * dRodR(1:Ntz, issym + 2, ii)! dRx/dR, zeta derivative

              DZxij(1:Ntz, 0) = (1.0_rkind - sbar ** 2) * dZodR(1:Ntz, issym, ii)! dZx/dR
              DZxij(1:Ntz, 1) = -sbar * dZodR(1:Ntz, issym, ii)
              DZxij(1:Ntz, 2) = 0.0_rkind
              DZxij(1:Ntz, 3) = (1.0_rkind - sbar ** 2) * dZodR(1:Ntz, issym + 2, ii)! dZx/dR, zeta derivative
            else
              DRxij(1:Ntz, 0) = (1.0_rkind - sbar ** 2) * dRodZ(1:Ntz, 1 - issym, ii)! dRx/dZ
              DRxij(1:Ntz, 1) = -sbar * dRodZ(1:Ntz, 1 - issym, ii)
              DRxij(1:Ntz, 2) = 0.0_rkind
              DRxij(1:Ntz, 3) = (1.0_rkind - sbar ** 2) * dRodZ(1:Ntz, 1 - issym + 2, ii)! dRx/dZ, zeta derivative

              DZxij(1:Ntz, 0) = (1.0_rkind - sbar ** 2) * dZodZ(1:Ntz, 1 - issym, ii)! dZx/dZ
              DZxij(1:Ntz, 1) = -sbar * dZodZ(1:Ntz, 1 - issym, ii)
              DZxij(1:Ntz, 2) = 0.0_rkind
              DZxij(1:Ntz, 3) = (1.0_rkind - sbar ** 2) * dZodZ(1:Ntz, 1 - issym + 2, ii)! dZx/dZ, zeta derivative
            end if
          end if
        end if

      else ! matches if( Lcoordinatesingularity )

        if (innout == 0) signlss = -1
        if (innout == 1) signlss = +1

        if ( (irz == 0 .and. issym == 0) .or. (irz == 1 .and. issym == 1) ) then ! cosine
          Dij(1:Ntz, 0) = (1.0_rkind + signlss * lss) * 1.0_rkind / 2.0_rkind * cosi(1:Ntz, ii)
          Dij(1:Ntz, 1) = (signlss) * 1.0_rkind / 2.0_rkind * cosi(1:Ntz, ii)
          Dij(1:Ntz, 2) = (1.0_rkind + signlss * lss) * 1.0_rkind / 2.0_rkind * sini(1:Ntz, ii) * (-im(ii))
          Dij(1:Ntz, 3) = (1.0_rkind + signlss * lss) * 1.0_rkind / 2.0_rkind * sini(1:Ntz, ii) * (+in(ii))
        else !   sine
          Dij(1:Ntz, 0) = (1.0_rkind + signlss * lss) * 1.0_rkind / 2.0_rkind * sini(1:Ntz, ii)
          Dij(1:Ntz, 1) = (signlss) * 1.0_rkind / 2.0_rkind * sini(1:Ntz, ii)
          Dij(1:Ntz, 2) = (1.0_rkind + signlss * lss) * 1.0_rkind / 2.0_rkind * cosi(1:Ntz, ii) * (+im(ii))
          Dij(1:Ntz, 3) = (1.0_rkind + signlss * lss) * 1.0_rkind / 2.0_rkind * cosi(1:Ntz, ii) * (-in(ii))
        end if

      end if

      if (Lcurvature == 5) then ! we only need the 2D Jacobian
        if (Igeometry == 3) then ! only works for toroidal

          if (irz == 0) sg(1:Ntz, 1) = (Zij(1:Ntz, 1, 0) * Dij(1:Ntz, 2) - Dij(1:Ntz, 1) * Zij(1:Ntz, 2, 0))
          if (irz == 1) sg(1:Ntz, 1) = (Dij(1:Ntz, 1) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) * Dij(1:Ntz, 2))

        else
          call fatal(Lcurvature == 5 .and. Igeometry /= 3, __FILE__, "Lcurvature.eq.5 can only be combined with          &
              &   Igeometry.ne.3 ")

        end if ! if (Igeometry .eq. 3)

      else ! we need more for Lcurvature=3,4

        select case (Igeometry)

        case (1) ! Lcurvature=3,4  Igeometry=1  Cartesian
          !                  sg(1:Ntz,0) = Rij(1:Ntz,1,0)*rpol*rtor
          sg(1:Ntz, 1) = Dij(1:Ntz, 1) * rpol * rtor
          !   if( irz.eq.1 ) sg(1:Ntz,1) =

          do ii = 1, 3 ! careful: ii was used with a different definition above
            do jj = ii, 3
              dguvij(1:Ntz, ii, jj) = Dij(1:Ntz, ii) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * Dij(1:Ntz, jj)
              !     if( irz.eq.1 ) dguvij(1:Ntz,ii,jj) =
            end do
          end do

        case (2) ! Lcurvature=3,4,5 Igeometry=2 cylindrical
          !                  sg(1:Ntz,0) = Rij(1:Ntz,1,0) * Rij(1:Ntz,0,0)
          if (irz == 0) sg(1:Ntz, 1) = Dij(1:Ntz, 1) * Rij(1:Ntz, 0, 0) + Rij(1:Ntz, 1, 0) * Dij(1:Ntz, 0)

          do ii = 1, 3 ! careful: ii was used with a different definition above
            do jj = ii, 3
              if (irz == 0) dguvij(1:Ntz, ii, jj) = Dij(1:Ntz, ii) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * Dij(1:Ntz,&
                  & jj)
              if (irz == 1) then
                call fatal(.true., __FILE__, "No Z-geometrical degree of freedom when Igeometry=2")
                !dguvij(1:Ntz,ii,jj) = Dij(1:Ntz,ii) * Zij(1:Ntz,jj,0) + Zij(1:Ntz,ii,0) * Dij(1:Ntz,jj) ! TODO REMOVE
              end if
            end do
          end do

          dguvij(1:Ntz, 2, 2) = dguvij(1:Ntz, 2, 2) + 2.0_rkind * Dij(1:Ntz, 0) * Rij(1:Ntz, 0, 0)
          !dguvij(1:Ntz,3,3) = additional term is unity

        case (3) ! Lcurvature=3,4,5 Igeometry=3 toroidal

          if (LcoordinateSingularity) then
            !                  sg(1:Ntz,0) = Rij(1:Ntz,0,0) * ( Zij(1:Ntz,1,0)*Rij(1:Ntz,2,0) -
            !                  Rij(1:Ntz,1,0)*Zij(1:Ntz,2,0) )
            if (irz == 0) sg(1:Ntz, 1) = (Dij(1:Ntz, 0) + DRxij(1:Ntz, 0)) * (Zij(1:Ntz, 1, 0) * Rij(1:Ntz, 2, 0) -&
                & Rij(1:Ntz, 1, 0) * Zij(1:Ntz, 2, 0)) + Rij(1:Ntz, 0, 0) * (Zij(1:Ntz, 1, 0) * Dij(1:Ntz, 2) -&
                & (Dij(1:Ntz, 1) + DRxij(1:Ntz, 1)) * Zij(1:Ntz, 2, 0)) + Rij(1:Ntz, 0, 0) * (DZxij(1:Ntz, 1) * Rij(1:Ntz&
                &, 2, 0))
            if (irz == 1) sg(1:Ntz, 1) = DRxij(1:Ntz, 0) * (Zij(1:Ntz, 1, 0) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) *&
                & Zij(1:Ntz, 2, 0)) + Rij(1:Ntz, 0, 0) * ( (Dij(1:Ntz, 1) + DZxij(1:Ntz, 1)) * Rij(1:Ntz, 2, 0) -&
                & Rij(1:Ntz, 1, 0) * Dij(1:Ntz, 2)) + Rij(1:Ntz, 0, 0) * (-DRxij(1:Ntz, 1) * Zij(1:Ntz, 2, 0))

            do ii = 1, 3 ! careful: ii was used with a different definition above
              do jj = ii, 3
                if (irz == 0) dguvij(1:Ntz, ii, jj) = (Dij(1:Ntz, ii) + DRxij(1:Ntz, ii)) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz&
                    &, ii, 0) * (Dij(1:Ntz, jj) + DRxij(1:Ntz, jj)) + DZxij(1:Ntz, ii) * Zij(1:Ntz, jj, 0) + Zij(1:Ntz,&
                    & ii, 0) * DZxij(1:Ntz, jj)
                if (irz == 1) dguvij(1:Ntz, ii, jj) = (Dij(1:Ntz, ii) + DZxij(1:Ntz, ii)) * Zij(1:Ntz, jj, 0) + Zij(1:Ntz&
                    &, ii, 0) * (Dij(1:Ntz, jj) + DZxij(1:Ntz, jj)) + DRxij(1:Ntz, ii) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz,&
                    & ii, 0) * DRxij(1:Ntz, jj)
              end do
            end do

            if (irz == 0) dguvij(1:Ntz, 3, 3) = dguvij(1:Ntz, 3, 3) + 2.0_rkind * (Dij(1:Ntz, 0) + DRxij(1:Ntz, 0)) *&
                & Rij(1:Ntz, 0, 0)
            if (irz == 1) dguvij(1:Ntz, 3, 3) = dguvij(1:Ntz, 3, 3) + 2.0_rkind * (DRxij(1:Ntz, 0)) * Rij(1:Ntz, 0, 0)

          else

            if (irz == 0) sg(1:Ntz, 1) = Dij(1:Ntz, 0) * (Zij(1:Ntz, 1, 0) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) *&
                & Zij(1:Ntz, 2, 0)) + Rij(1:Ntz, 0, 0) * (Zij(1:Ntz, 1, 0) * Dij(1:Ntz, 2) - Dij(1:Ntz, 1) * Zij(1:Ntz,&
                & 2, 0))
            if (irz == 1) sg(1:Ntz, 1) = Rij(1:Ntz, 0, 0) * (Dij(1:Ntz, 1) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) *&
                & Dij(1:Ntz, 2))

            do ii = 1, 3 ! careful: ii was used with a different definition above
              do jj = ii, 3
                if (irz == 0) dguvij(1:Ntz, ii, jj) = Dij(1:Ntz, ii) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * Dij(1:Ntz,&
                    & jj)
                if (irz == 1) dguvij(1:Ntz, ii, jj) = Dij(1:Ntz, ii) * Zij(1:Ntz, jj, 0) + Zij(1:Ntz, ii, 0) * Dij(1:Ntz,&
                    & jj)
              end do
            end do

            if (irz == 0) dguvij(1:Ntz, 3, 3) = dguvij(1:Ntz, 3, 3) + 2.0_rkind * Dij(1:Ntz, 0) * Rij(1:Ntz, 0, 0)
          end if

        case default

          call fatal(.true., __FILE__, "supplied Igeometry is not yet supported for Lcurvature = 3 or Lcurvature = 4 ")

        end select

        do ii = 2, 3
          do jj = 1, ii - 1
            dguvij(1:Ntz, ii, jj) = dguvij(1:Ntz, jj, ii)! symmetry of metrics
          end do
        end do

        guvij(1:Ntz, 0, 0, 1) = 0.0_rkind ! this "metric" does not depend on geometry: helicity matrix does not depend on
        ! geometry

        if (Lcurvature == 3) then

          do ii = 1, 3
            do jj = 1, 3
              guvij(1:Ntz, ii, jj, 1) = dguvij(1:Ntz, ii, jj) - guvij(1:Ntz, ii, jj, 0) * sg(1:Ntz, 1) * inv_jac(1:Ntz)!
              !differentiated metric elements
            end do
          end do

        else

          do ii = 1, 3
            do jj = 1, 3
              guvij(1:Ntz, ii, jj, 1) = dguvij(1:Ntz, ii, jj)! differentiated metric elements
            end do
          end do

        end if

      end if
    end select ! matches select case( Lcurvature )
  end subroutine compute_coordinates

  subroutine direction_metric_response(lss, Lcurvature, Nt, Nz, mn, Lcoordinatesingularity, sbar, fj, &
      Rij, Zij, sg, inv_jac, guvij)

    ! General-direction (Ldirgen) counterpart of the single-harmonic Lcurvature=3/4/5 branch of
    ! compute_coordinates: the metric response to a full interface deformation given by the
    ! direction coefficients dirRc/dirRs/dirZs/dirZc (both boundaries of the current volume, R and
    ! Z, cosine and sine) plus the precomputed axis-motion fields dir_ax*. Works with the TOTAL
    ! displacement fields dR(:,0:3)/dZ(:,0:3) (value, d/ds, d/dtheta, d/dzeta): the per-harmonic
    ! irz/issym branches of compute_coordinates are the special cases of the universal first-order
    ! variations evaluated here, dguvij(i,j) = dRi*Rj + Ri*dRj + dZi*Zj + Zi*dZj and
    ! dsg = the bilinear expansion of sg in (dR, dZ).

    integer, intent(in) :: Lcurvature, Nt, Nz, mn
    real(rkind), intent(in) :: lss, sbar
    logical, intent(in) :: Lcoordinatesingularity
    real(rkind), intent(in) :: fj(1:mn, 0:2)
    real(rkind), intent(in) :: Rij(1:(Nt*Nz), 0:3, 0:3), Zij(1:(Nt*Nz), 0:3, 0:3)
    real(rkind), intent(in) :: inv_jac(1:(Nt*Nz))
    real(rkind), intent(inout) :: sg(1:(Nt*Nz), 0:3)
    real(rkind), intent(inout) :: guvij(1:(Nt*Nz), 0:3, 0:3, -1:3)

    integer :: ii, jj, Ntz
    real(rkind) :: alss, blss
    real(rkind) :: ce0(1:mn, 1:2), co0(1:mn, 1:2), ce1(1:mn, 1:2), co1(1:mn, 1:2)! 1 = R, 2 = Z
    real(rkind) :: dR(1:(Nt*Nz), 0:3), dZ(1:(Nt*Nz), 0:3), dguvij(1:(Nt*Nz), 1:3, 1:3)

    Ntz = Nt * Nz

    ! Effective cosine/sine coefficients of the displacement fields and their s-derivatives, with
    ! the radial interpolation factors folded in: per-mode fj at the coordinate singularity (outer
    ! boundary only; the axis carries no direct degrees of freedom), linear (1 -+ lss)/2 elsewhere.
    if (Lcoordinatesingularity) then
      call fatal(use_ext_mesh, __FILE__, "general-direction mode does not support use_ext_mesh")
      ce0(1:mn, 1) = fj(1:mn, 0) * dirRc(1:mn, 1)
      co0(1:mn, 1) = fj(1:mn, 0) * dirRs(1:mn, 1)
      ce0(1:mn, 2) = fj(1:mn, 0) * dirZc(1:mn, 1)
      co0(1:mn, 2) = fj(1:mn, 0) * dirZs(1:mn, 1)
      ce1(1:mn, 1) = fj(1:mn, 1) * dirRc(1:mn, 1)
      co1(1:mn, 1) = fj(1:mn, 1) * dirRs(1:mn, 1)
      ce1(1:mn, 2) = fj(1:mn, 1) * dirZc(1:mn, 1)
      co1(1:mn, 2) = fj(1:mn, 1) * dirZs(1:mn, 1)
    else
      alss = 1.0_rkind / 2.0_rkind * (1.0_rkind - lss)
      blss = 1.0_rkind / 2.0_rkind * (1.0_rkind + lss)
      ce0(1:mn, 1) = alss * dirRc(1:mn, 0) + blss * dirRc(1:mn, 1)
      co0(1:mn, 1) = alss * dirRs(1:mn, 0) + blss * dirRs(1:mn, 1)
      ce0(1:mn, 2) = alss * dirZc(1:mn, 0) + blss * dirZc(1:mn, 1)
      co0(1:mn, 2) = alss * dirZs(1:mn, 0) + blss * dirZs(1:mn, 1)
      ce1(1:mn, 1) = (dirRc(1:mn, 1) - dirRc(1:mn, 0)) * 1.0_rkind / 2.0_rkind
      co1(1:mn, 1) = (dirRs(1:mn, 1) - dirRs(1:mn, 0)) * 1.0_rkind / 2.0_rkind
      ce1(1:mn, 2) = (dirZc(1:mn, 1) - dirZc(1:mn, 0)) * 1.0_rkind / 2.0_rkind
      co1(1:mn, 2) = (dirZs(1:mn, 1) - dirZs(1:mn, 0)) * 1.0_rkind / 2.0_rkind
    end if

    ! Displacement fields and their derivatives; same invfft conventions as the base coordinates.
    call invfft(mn, im(1:mn), in(1:mn), ce0(1:mn, 1), co0(1:mn, 1), ce0(1:mn, 2), co0(1:mn, 2), &
        dR(1:Ntz, 0), dZ(1:Ntz, 0))
    call invfft(mn, im(1:mn), in(1:mn), ce1(1:mn, 1), co1(1:mn, 1), ce1(1:mn, 2), co1(1:mn, 2), &
        dR(1:Ntz, 1), dZ(1:Ntz, 1))
    call invfft(mn, im(1:mn), in(1:mn), im(1:mn) * co0(1:mn, 1), -im(1:mn) * ce0(1:mn, 1), &
        im(1:mn) * co0(1:mn, 2), -im(1:mn) * ce0(1:mn, 2), dR(1:Ntz, 2), dZ(1:Ntz, 2))
    call invfft(mn, im(1:mn), in(1:mn), -in(1:mn) * co0(1:mn, 1), in(1:mn) * ce0(1:mn, 1), &
        -in(1:mn) * co0(1:mn, 2), in(1:mn) * ce0(1:mn, 2), dR(1:Ntz, 3), dZ(1:Ntz, 3))

    ! Moving-coordinate-axis response (toroidal axis volume, Lcurvature=3/4 only, matching the
    ! per-harmonic branch: Lcurvature=5 excludes axis motion by convention).
    if (Lcoordinatesingularity .and. Igeometry == 3 .and. (Lcurvature == 3 .or. Lcurvature == 4)) then
      dR(1:Ntz, 0) = dR(1:Ntz, 0) + (1.0_rkind - sbar ** 2) * dir_axR(1:Ntz)
      dR(1:Ntz, 1) = dR(1:Ntz, 1) - sbar * dir_axR(1:Ntz)
      dR(1:Ntz, 3) = dR(1:Ntz, 3) + (1.0_rkind - sbar ** 2) * dir_axRz(1:Ntz)
      dZ(1:Ntz, 0) = dZ(1:Ntz, 0) + (1.0_rkind - sbar ** 2) * dir_axZ(1:Ntz)
      dZ(1:Ntz, 1) = dZ(1:Ntz, 1) - sbar * dir_axZ(1:Ntz)
      dZ(1:Ntz, 3) = dZ(1:Ntz, 3) + (1.0_rkind - sbar ** 2) * dir_axZz(1:Ntz)
    end if

    if (Lcurvature == 5) then ! only the 2D Jacobian is needed
      call fatal(Igeometry /= 3, __FILE__, "Lcurvature.eq.5 can only be combined with Igeometry.eq.3 ")
      sg(1:Ntz, 1) = dZ(1:Ntz, 1) * Rij(1:Ntz, 2, 0) + Zij(1:Ntz, 1, 0) * dR(1:Ntz, 2) &
          - dR(1:Ntz, 1) * Zij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) * dZ(1:Ntz, 2)
      return
    end if

    select case (Igeometry)

    case (1) ! Cartesian
      sg(1:Ntz, 1) = dR(1:Ntz, 1) * rpol * rtor

      do ii = 1, 3
        do jj = ii, 3
          dguvij(1:Ntz, ii, jj) = dR(1:Ntz, ii) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * dR(1:Ntz, jj)
        end do
      end do

    case (2) ! cylindrical
      sg(1:Ntz, 1) = dR(1:Ntz, 1) * Rij(1:Ntz, 0, 0) + Rij(1:Ntz, 1, 0) * dR(1:Ntz, 0)

      do ii = 1, 3
        do jj = ii, 3
          dguvij(1:Ntz, ii, jj) = dR(1:Ntz, ii) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * dR(1:Ntz, jj)
        end do
      end do

      dguvij(1:Ntz, 2, 2) = dguvij(1:Ntz, 2, 2) + 2.0_rkind * dR(1:Ntz, 0) * Rij(1:Ntz, 0, 0)

    case (3) ! toroidal
      sg(1:Ntz, 1) = dR(1:Ntz, 0) * (Zij(1:Ntz, 1, 0) * Rij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) * Zij(1:Ntz, 2, 0)) &
          + Rij(1:Ntz, 0, 0) * (dZ(1:Ntz, 1) * Rij(1:Ntz, 2, 0) + Zij(1:Ntz, 1, 0) * dR(1:Ntz, 2) &
          - dR(1:Ntz, 1) * Zij(1:Ntz, 2, 0) - Rij(1:Ntz, 1, 0) * dZ(1:Ntz, 2))

      do ii = 1, 3
        do jj = ii, 3
          dguvij(1:Ntz, ii, jj) = dR(1:Ntz, ii) * Rij(1:Ntz, jj, 0) + Rij(1:Ntz, ii, 0) * dR(1:Ntz, jj) &
              + dZ(1:Ntz, ii) * Zij(1:Ntz, jj, 0) + Zij(1:Ntz, ii, 0) * dZ(1:Ntz, jj)
        end do
      end do

      dguvij(1:Ntz, 3, 3) = dguvij(1:Ntz, 3, 3) + 2.0_rkind * dR(1:Ntz, 0) * Rij(1:Ntz, 0, 0)

    case default
      call fatal(.true., __FILE__, "supplied Igeometry is not yet supported for Lcurvature = 3 or Lcurvature = 4 ")

    end select

    do ii = 2, 3
      do jj = 1, ii - 1
        dguvij(1:Ntz, ii, jj) = dguvij(1:Ntz, jj, ii)! symmetry of metrics
      end do
    end do

    guvij(1:Ntz, 0, 0, 1) = 0.0_rkind ! the helicity "metric" does not depend on geometry

    if (Lcurvature == 3) then
      do ii = 1, 3
        do jj = 1, 3
          guvij(1:Ntz, ii, jj, 1) = dguvij(1:Ntz, ii, jj) - guvij(1:Ntz, ii, jj, 0) * sg(1:Ntz, 1) * inv_jac(1:Ntz)
        end do
      end do
    else
      do ii = 1, 3
        do jj = 1, 3
          guvij(1:Ntz, ii, jj, 1) = dguvij(1:Ntz, ii, jj)
        end do
      end do
    end if

  end subroutine direction_metric_response
end module geometry_mod
