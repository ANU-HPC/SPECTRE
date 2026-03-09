module geometry_mod
  use io_mod, only: stderr
  use precisions_mod, only: rkind
  use input_list_mod, only: Wcompute_coordinates, Igeometry, Ntor, rpol, rtor, enforce_stell_sym
  use allglobal_mod, only: im, in, iRbc, iZbs, iRbs, iZbc, Lcoordinatesingularity, Rij, Zij, cosi, sini, sg, inv_jac, &
      guvij, dBdX, dRodR, dRodZ, dZodR, dZodZ, gaussianabscissae, use_ext_mesh, Remn_ext, Romn_ext, Zemn_ext, Zomn_ext
  use errors_mod, only: fatal
  use numerical_mod, only: invfft

  implicit none(type, external)

  private
  public :: compute_coordinates

contains

  subroutine compute_coordinates(lvol, lss, Lcurvature, Nt, Nz, mn)
    integer, intent(in) :: lvol, Lcurvature, Nt, Nz, mn
    real(rkind), intent(in) :: lss

    integer :: ii, jj, kk, irz, innout, issym, signlss, radial_index
    real(rkind) :: Remn(1:mn, 0:2), Zomn(1:mn, 0:2), Romn(1:mn, 0:2), Zemn(1:mn, 0:2), alss, blss, sbar, fj(1:mn, 0:2)

    real(rkind) :: Dij(1:(Nt*Nz), 0:3), dguvij(1:(Nt*Nz), 1:3, 1:3), DRxij(1:(Nt*Nz), 0:3), DZxij(1:(Nt*Nz), 0:3)

    integer :: Ntz

    Ntz = Nt * Nz

    Rij(1:Ntz, 0:3, 0:3) = 0.0_rkind
    Zij(1:Ntz, 0:3, 0:3) = 0.0_rkind
    if (Lcurvature > 0) then
      sg(1:Ntz, 0) = 0.0_rkind
      inv_jac(1:Ntz) = 0.0_rkind
      guvij(1:Ntz, 1:3, 1:3, 0) = 0.0_rkind
    end if
    if (Lcurvature > 1) then
      sg(1:Ntz, 1:3) = 0.0_rkind
      guvij(1:Ntz, 1:3, 1:3, 1:3) = 0.0_rkind
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
end module geometry_mod
